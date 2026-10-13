"""Живой скан: телефон шлёт кадры по одному, компьютер сразу считает глубину и подсказывает, что доснять.

На каждый кадр:
  - глубина (кладётся в тот же кеш out/<скан>/depth/, что читает cloud.py — второй раз не считается);
  - проверка: темно, смазано, не совпал со стенами соседних кадров (лицо, пол, стол вплотную);
  - пересборка грубого плана сверху (без уточнения поз) — его телефон показывает во время съёмки.
"""
from __future__ import annotations

import base64
import io
import json
import queue
import struct
import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from build_room import MODEL, focal_px
from cloud import (cached_depth, clean_mask, floor_height, grid_score, occupancy, render_plan, to_cam,
                   wall_ring, world)

HFOV = 69.0
DARK = 45       # средняя яркость 0..255: ниже — просим фонарик
BLUR = 12       # дисперсия лапласиана на 360×640: ниже — кадр смазан (резкие кадры 30–300)
MIN_OVERLAP = 0.2
MAX_FRAMES = 400  # кадров на один скан


def best_overlap(item: dict, near: list[dict], cam_h: float, ceiling: float) -> float:
    """Лучшая доля совпадения стен кадра с соседями при сдвиге курса ±8° (гироскоп уплывает)."""
    cell, size = 0.08, 200
    origin = np.array([-size * cell / 2, -size * cell / 2])
    walls = [wall_ring(world(f["cam"][f["ok"]], f["yaw"], f["pitch"]), cam_h, ceiling) for f in near]
    occ = occupancy(np.concatenate(walls), origin, size, cell, dilate=2)
    best = 0.0
    for dy in range(-8, 9, 2):
        mine = wall_ring(world(item["cam"][item["ok"]], item["yaw"] + dy, item["pitch"]), cam_h, ceiling)
        if len(mine) < 150:
            return 0.0
        best = max(best, grid_score(mine, occ, origin, cell) / len(mine))
    return best


def sharpness(g: np.ndarray) -> float:
    return float((g[1:-1, :-2] + g[1:-1, 2:] + g[:-2, 1:-1] + g[2:, 1:-1] - 4 * g[1:-1, 1:-1]).var())


class LiveScan:
    def __init__(self, sid: str, out: Path, meta: dict, sess: ort.InferenceSession):
        self.sid, self.out, self.meta, self.sess = sid, out, meta, sess
        self.frames: list[dict] = []     # метаданные кадров в порядке прихода
        self.cloud: dict[int, dict] = {}  # i → точки кадра в осях камеры
        self.plan_png = b""
        self.cloud_bin = b""  # живое облако для страницы статуса: uint32 n, float32 xyz×n, uint8 rgb×n
        self.plan_ver = 0
        self.size = None
        self.lock = threading.Lock()
        self.q: queue.Queue = queue.Queue()
        self.closed = False
        self.capture: Path | None = None
        self._since_plan = 0
        (out / "frames").mkdir(parents=True, exist_ok=True)
        (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        threading.Thread(target=self._worker, daemon=True).start()

    def add(self, fr: dict) -> int:
        """Кадр с телефона: на диск и в очередь — атомарно, чтобы finish() не проскочил между ними."""
        jpeg = base64.b64decode(fr.pop("jpeg").split(",", 1)[1])
        with self.lock:
            if self.closed:
                raise ScanClosed("съёмка уже закончена")
            if len(self.frames) >= MAX_FRAMES:
                raise ScanClosed(f"больше {MAX_FRAMES} кадров на скан не принимаем")
            i = len(self.frames)
            (self.out / "frames" / f"{i:03d}.jpg").write_bytes(jpeg)
            fr.update(i=i, state="queued")
            self.frames.append(fr)
            self.q.put(i)
        return i

    def _set(self, i: int, **kw):
        with self.lock:
            self.frames[i].update(kw)

    def state(self) -> dict:
        with self.lock:
            return {"id": self.sid, "room": self.meta.get("room"), "started": self.meta.get("created"),
                    "frames": [{k: f.get(k) for k in ("i", "yaw", "pitch", "row", "state", "reason", "tip")} for f in self.frames],
                    "plan_ver": self.plan_ver, "size": self.size, "queued": self.q.qsize()}

    def _worker(self):
        while True:
            i = self.q.get()
            if i is None:  # finish(): поток выходит, тяжёлые массивы освобождаем
                self.q.task_done()
                with self.lock:
                    self.cloud.clear()
                return
            try:
                self._process(i)
            except Exception as e:  # кадр не должен ронять сессию
                self._set(i, state="bad", reason=f"ошибка: {e}")
            finally:
                self.q.task_done()

    def _process(self, i: int):
        fr = self.frames[i]
        img = Image.open(self.out / "frames" / f"{i:03d}.jpg").convert("RGB")
        g = np.asarray(img.convert("L").resize((360, 640) if img.height > img.width else (640, 360)), np.float32)
        tip = "темновато — включи фонарик" if g.mean() < DARK else None
        if sharpness(g) < BLUR:  # дешёвая проверка — до нейросети
            self._set(i, state="bad", reason="смазано — замри перед кадром", tip=tip)
            return
        d = cached_depth(self.sess, img, self.out / "depth" / f"{i:03d}.npy")
        cam, vs, us = to_cam(d, focal_px(*img.size, HFOV), step=6)
        item = {"i": i, "yaw": fr["yaw"], "pitch": fr["pitch"], "scale": 1.0, "cam": cam,
                "ok": clean_mask(d)[vs, us], "rgb": np.asarray(img)[vs, us]}
        self._set(i, state="ok", reason=None, tip=tip)
        with self.lock:
            self.cloud[i] = item
        cam_h = self._check_all()
        # план и облако для страницы — когда очередь разобрана или раз в 4 кадра, а не после каждого
        self._since_plan += 1
        if self.q.qsize() == 0 or self._since_plan >= 4:
            self._since_plan = 0
            self._replan(cam_h)

    def _check_all(self) -> float:
        """Кадр сверяем со стенами принятых соседей (±30° по курсу). Мало соседей — кадр пока принят;
        забракованный перепроверяется, когда соседей прибавилось, и может снова стать принятым."""
        ceiling = self.meta.get("ceiling_m") or 2.7
        with self.lock:
            items = [self.cloud[k] for k in sorted(self.cloud)]
        good = lambda: [f for f in items if self.frames[f["i"]]["state"] == "ok"]
        cam_h = floor_height(good()) if len(good()) >= 4 else 1.2
        if len(items) < 6:  # высота телефона над полом ещё не устоялась — сверять рано
            return cam_h
        for it in items:
            if it.get("confirmed"):
                continue
            if abs(it["pitch"]) > 15:  # вниз/вверх стен почти не видно — сверять не с чем
                it["confirmed"] = True
                continue
            near = [f for f in good() if f is not it and abs(f["pitch"]) <= 15
                    and abs((f["yaw"] - it["yaw"] + 540) % 360 - 180) < 30]
            if len(near) < 2 or len(near) == it.get("seen"):
                continue
            it["seen"] = len(near)
            if best_overlap(it, near, cam_h, ceiling) >= MIN_OVERLAP:
                self._set(it["i"], state="ok", reason=None)
                it["confirmed"] = len(near) >= 3
            else:
                self._set(it["i"], state="bad", reason="не совпал с соседними кадрами (человек, пол, предмет вплотную?) — сними это направление ещё раз")
        return cam_h

    def _replan(self, cam_h: float):
        ceiling = self.meta.get("ceiling_m") or 2.7
        with self.lock:
            frames = [self.cloud[k] for k in sorted(self.cloud) if self.frames[k]["state"] == "ok"]
        if not frames:
            return
        pts, cols = [], []
        for f in frames:
            p = world(f["cam"][f["ok"]], f["yaw"], f["pitch"])
            p[:, 2] += cam_h
            keep = (p[:, 2] > -0.1) & (p[:, 2] < ceiling + 0.2)
            pts.append(p[keep]); cols.append(f["rgb"][f["ok"]][keep])
        xyz, rgb = np.concatenate(pts), np.concatenate(cols)
        sub = slice(None, None, max(1, len(xyz) // 80000))
        cloud = struct.pack("<I", len(xyz[sub])) + xyz[sub].astype("<f4").tobytes() + rgb[sub].astype(np.uint8).tobytes()
        img = render_plan(xyz, [f["yaw"] for f in frames], ceiling, px_per_m=60)
        buf = io.BytesIO()
        img.save(buf, "PNG", optimize=False)
        walls = xyz[(xyz[:, 2] > 1.0) & (xyz[:, 2] < ceiling - 0.15)][:, :2]
        size = None
        if len(walls) > 2000:
            lo, hi = np.percentile(walls, 3, axis=0), np.percentile(walls, 97, axis=0)
            size = [round(float(hi[0] - lo[0]), 1), round(float(hi[1] - lo[1]), 1)]
        with self.lock:
            self.plan_png, self.size, self.cloud_bin = buf.getvalue(), size, cloud
            self.plan_ver += 1

    def finish(self, captures: Path) -> tuple[Path, bool]:
        """Закрывает приём, дожидается обработки и пишет обычный файл скана. Повторный вызов —
        тот же файл и False (сборку второй раз не запускать)."""
        with self.lock:
            if self.capture:
                return self.capture, False
            self.closed = True
        self.q.join()
        frames = []
        for f in self.frames:
            jpeg = base64.b64encode((self.out / "frames" / f"{f['i']:03d}.jpg").read_bytes()).decode()
            frames.append({**f, "jpeg": "data:image/jpeg;base64," + jpeg})
        cap = {**self.meta, "format": "room-twin-capture/1", "frames": frames,
               "live_bad": [f["i"] for f in self.frames if f.get("state") == "bad"]}
        path = captures / f"room-capture-{self.sid}.json"
        path.write_text(json.dumps(cap, ensure_ascii=False))
        self.capture = path
        self.q.put(None)  # отпускаем поток обработки
        return path, True


class ScanClosed(Exception):
    pass


_SESS = None
_SESS_LOCK = threading.Lock()


def session() -> ort.InferenceSession:
    """Одна сессия ONNX на процесс: два одновременных старта не грузят модель дважды."""
    global _SESS
    with _SESS_LOCK:
        if _SESS is None:
            _SESS = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    return _SESS
