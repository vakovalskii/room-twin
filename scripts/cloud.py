"""Скан → очищенное облако точек комнаты + план сверху + 3D-вьюер облака.

  uv run --with numpy --with onnxruntime --with pillow \
    python scripts/cloud.py captures/room-capture-....json --out out/<комната> [--drop 0,11,12]

Шаги:
  1. глубина каждого кадра в метрах (кеш в out/<комната>/depth/);
  2. чистка: края кадра, «летающие» пиксели на границах предметов, слишком близко и далеко;
  3. высота телефона над полом — по плоскости пола в облаке;
  4. уточнение курса и масштаба глубины каждого кадра по перекрытию с остальными
     (гироскоп уплывает на градусы, масштаб глубины гуляет от кадра к кадру на проценты);
  5. склейка, прореживание сеткой, удаление одиночных выбросов;
  6. cloud.ply, plan.png (вид сверху с сеткой в метрах), poses.json, cloud.html.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_room import MODEL, ROOT, depth_map, fit_rectangle, focal_px, rotation, to_room  # noqa: E402


def load_frames(cap: dict, out: Path, drop: set[int]):
    (out / "frames").mkdir(parents=True, exist_ok=True)
    for fr in cap["frames"]:
        if fr["i"] in drop:
            continue
        path = out / "frames" / f"{fr['i']:03d}.jpg"
        if not path.exists():
            Image.open(io.BytesIO(base64.b64decode(fr["jpeg"].split(",", 1)[1]))).save(path, quality=88)
        yield fr, Image.open(path).convert("RGB")


def cached_depth(sess, img: Image.Image, path: Path) -> np.ndarray:
    if path.exists():
        return np.load(path).astype(np.float32)
    d = depth_map(sess, img)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, d.astype(np.float16))
    return d


def clean_mask(d: np.ndarray, border: float = 0.03, jump: float = 0.04, near: float = 0.35, far: float = 7.0) -> np.ndarray:
    """Годные пиксели: не у края кадра, без скачка глубины к соседям, в разумной дальности."""
    h, w = d.shape
    ok = (d > near) & (d < far)
    by, bx = int(h * border), int(w * border)
    ok[:by] = ok[-by:] = False
    ok[:, :bx] = ok[:, -bx:] = False
    pad = np.pad(d, 1, mode="edge")
    neigh = np.stack([pad[1:-1, :-2], pad[1:-1, 2:], pad[:-2, 1:-1], pad[2:, 1:-1]])
    ok &= np.max(np.abs(neigh - d), axis=0) / d < jump  # граница предмета: глубина «летит» между планами
    return ok


def to_cam(d: np.ndarray, f: float, step: int):
    h, w = d.shape
    vs, us = np.mgrid[0:h:step, 0:w:step]
    z = d[vs, us]
    return np.stack([(us - w / 2) * z / f, (vs - h / 2) * z / f, z], axis=-1), vs, us


def world(pts_cam: np.ndarray, yaw: float, pitch: float, scale: float = 1.0) -> np.ndarray:
    return (pts_cam * scale) @ rotation(yaw, pitch).T


def wall_ring(p: np.ndarray, cam_h: float, ceiling: float) -> np.ndarray:
    """Точки на высоте стен (над мебелью, под потолком) → 2D для выравнивания кадров между собой."""
    z = p[..., 2] + cam_h
    sel = (z > 1.0) & (z < ceiling - 0.15)
    return p[sel][:, :2]


def grid_score(xy: np.ndarray, occ: np.ndarray, origin: np.ndarray, cell: float) -> int:
    ij = np.floor((xy - origin) / cell).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < occ.shape[0]) & (ij[:, 1] < occ.shape[1])
    return int(occ[ij[ok, 0], ij[ok, 1]].sum())


def occupancy(xy: np.ndarray, origin: np.ndarray, size: int, cell: float, dilate: int = 1) -> np.ndarray:
    occ = np.zeros((size, size), bool)
    ij = np.floor((xy - origin) / cell).astype(int)
    ok = (ij >= 0).all(1) & (ij < size).all(1)
    occ[ij[ok, 0], ij[ok, 1]] = True
    for _ in range(dilate):
        o = occ.copy()
        o[1:] |= occ[:-1]; o[:-1] |= occ[1:]; o[:, 1:] |= occ[:, :-1]; o[:, :-1] |= occ[:, 1:]
        occ = o
    return occ


def refine_poses(frames, cam_h: float, ceiling: float, passes: int = 3, max_pts: int = 20000):
    """Каждый кадр по очереди двигаем по курсу (±8°) и масштабу (±12 %) к остальным: максимум совпадений стен.

    Поворот по курсу — вращение вокруг вертикали, поэтому точки кадра один раз переводим в мир при курсе 0
    и дальше крутим только 2D-точки стен из фиксированной выборки max_pts — в разы быстрее пересчёта всего кадра."""
    cell, size = 0.06, 260
    origin = np.array([-size * cell / 2, -size * cell / 2])
    rng = np.random.default_rng(0)
    base = []  # точки кадра в мире при курсе 0 и масштабе 1
    for fr in frames:
        # без маски чистки: она режет стены под острым углом (скачок глубины > 4 %), а для выравнивания
        # они самые ценные; проверено на реальном скане — с маской разброс стен 0.22 м, без — 0.16 м
        p = world(fr["cam"].reshape(-1, 3), 0.0, fr["pitch"])
        if len(p) > max_pts:  # одна выборка на кадр: от масштаба и курса она не зависит, оценки сравнимы
            p = p[rng.choice(len(p), max_pts, replace=False)]
        base.append(p)

    def ring(k: int, yaw: float, scale: float) -> np.ndarray:
        p = base[k] * scale
        z = p[:, 2] + cam_h
        xy = p[(z > 1.0) & (z < ceiling - 0.15), :2]
        t = math.radians(yaw)  # world(): курс поворачивает по часовой, если смотреть сверху
        c, s_ = math.cos(t), math.sin(t)
        return np.stack([xy[:, 0] * c + xy[:, 1] * s_, -xy[:, 0] * s_ + xy[:, 1] * c], 1)

    rings = [ring(k, f["yaw"], f["scale"]) for k, f in enumerate(frames)]
    for _ in range(passes):
        for k, fr in enumerate(frames):
            others = np.concatenate([r for j, r in enumerate(rings) if j != k] or [np.zeros((0, 2))])
            occ = occupancy(others, origin, size, cell)
            best = (-1, fr["yaw"], fr["scale"])
            for s in (0.88, 0.92, 0.96, 1.0, 1.04, 1.08, 1.12):
                scale = fr["scale"] * s
                for dy in np.arange(-8, 8.01, 1.0):
                    sc = grid_score(ring(k, fr["yaw"] + dy, scale), occ, origin, cell)
                    if sc > best[0]:
                        best = (sc, fr["yaw"] + dy, scale)
            fr["yaw"], fr["scale"] = best[1], float(np.clip(best[2], 0.75, 1.3))
            rings[k] = ring(k, fr["yaw"], fr["scale"])


def floor_height(frames) -> float:
    """Высота телефона над полом: самая плотная горизонтальная полка точек ниже камеры."""
    zs = np.concatenate([world(f["cam"], f["yaw"], f["pitch"], f["scale"])[..., 2][f["ok"]] for f in frames])
    low = zs[(zs < -0.6) & (zs > -2.2)]
    if len(low) < 500:
        return 1.4
    hist, edges = np.histogram(low, bins=np.arange(-2.2, -0.6, 0.03))
    return float(-(edges[np.argmax(hist)] + 0.015))


def voxel_down(xyz: np.ndarray, rgb: np.ndarray, cell: float, min_pts: int = 2):
    """Сетка cell: точка на ячейку (среднее), ячейки с одной случайной точкой — выбросы."""
    key = np.floor(xyz / cell).astype(np.int64)
    key = (key[:, 0] + 100000) * 10_000_000_000 + (key[:, 1] + 100000) * 100000 + (key[:, 2] + 50000)
    uniq, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    sums = np.zeros((len(uniq), 3)); np.add.at(sums, inv, xyz)
    cols = np.zeros((len(uniq), 3)); np.add.at(cols, inv, rgb.astype(np.float64))
    keep = counts >= min_pts
    return (sums[keep] / counts[keep, None]).astype(np.float32), (cols[keep] / counts[keep, None]).astype(np.uint8)


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray):
    head = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode()
    rec = np.zeros(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec["r"], rec["g"], rec["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    path.write_bytes(head + rec.tobytes())


FONT = next((f for f in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
             if Path(f).exists()), None)


def font(size: int):
    return ImageFont.truetype(FONT, size) if FONT else ImageFont.load_default()


def render_plan(xyz: np.ndarray, yaws, ceiling: float, px_per_m: int = 90, labels=None) -> Image.Image:
    """Вид сверху: точки по высоте (синий пол → красный верх стен), сетка 0.5 м, лучи кадров из точки съёмки."""
    if len(xyz) < 10:
        xyz = np.array([[-2.0, -2.0, 0.0], [2.0, 2.0, 0.0]])
    lo = np.minimum(np.percentile(xyz[:, :2], 0.5, axis=0), -1.0) - 0.5
    hi = np.maximum(np.percentile(xyz[:, :2], 99.5, axis=0), 1.0) + 0.5
    W, H = ((hi - lo) * px_per_m).astype(int) + 1
    canvas = np.full((H, W, 3), (250, 250, 247), np.uint8)
    img = Image.fromarray(canvas)
    dr = ImageDraw.Draw(img)
    for gx in np.arange(math.floor(lo[0] * 2) / 2, hi[0], 0.5):
        x = (gx - lo[0]) * px_per_m
        dr.line([(x, 0), (x, H)], fill=(225, 225, 220) if gx % 1 else (200, 200, 195))
    for gy in np.arange(math.floor(lo[1] * 2) / 2, hi[1], 0.5):
        y = H - (gy - lo[1]) * px_per_m
        dr.line([(0, y), (W, y)], fill=(225, 225, 220) if gy % 1 else (200, 200, 195))
    canvas = np.asarray(img).copy()
    p = xyz[np.argsort(xyz[:, 2])]
    t = np.clip(p[:, 2] / ceiling, 0, 1)
    col = np.stack([40 + 200 * t, 70 + 120 * (1 - np.abs(t - 0.5) * 2), 200 - 160 * t], 1).astype(np.uint8)
    u = ((p[:, 0] - lo[0]) * px_per_m).astype(int)
    v = (H - (p[:, 1] - lo[1]) * px_per_m).astype(int)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    canvas[v[ok], u[ok]] = col[ok]
    img = Image.fromarray(canvas)
    dr = ImageDraw.Draw(img)
    c = (-lo[0] * px_per_m, H + lo[1] * px_per_m)
    for k, yaw in enumerate(yaws):
        a = math.radians(yaw)
        dr.line([c, (c[0] + math.sin(a) * 40, c[1] - math.cos(a) * 40)], fill=(255, 120, 0))
        if labels is not None:
            dr.text((c[0] + math.sin(a) * 50 - 6, c[1] - math.cos(a) * 50 - 6), str(labels[k]), fill=(200, 80, 0), font=font(11))
    dr.ellipse([c[0] - 5, c[1] - 5, c[0] + 5, c[1] + 5], fill=(255, 80, 0))
    return img


def plan_png(path: Path, xyz: np.ndarray, frames, ceiling: float):
    img = render_plan(xyz, [f["yaw"] for f in frames], ceiling, labels=[f["i"] for f in frames])
    ImageDraw.Draw(img).text((8, 6), "сетка 0.5 м · цвет — высота: синий пол → красный верх · оранжевое — точка съёмки и кадры",
                             fill=(60, 60, 60), font=font(13))
    img.save(path)


def plan_room_png(path: Path, xyz: np.ndarray, rect: dict, cams: list[tuple[int, float, float, float]], px_per_m: int = 120):
    """План в координатах комнаты — по нему агент снимает x, y, w, d предметов.
    Серые — стены (выше 1.2 м), цветные — всё ниже 1.2 м (мебель и пол раскрашены по высоте).
    Оси подписаны в метрах от угла: x вдоль стены S (низ), y вдоль стены W (лево)."""
    W, D = rect["x1"] - rect["x0"], rect["y1"] - rect["y0"]
    pad = 60
    w, h = int(W * px_per_m) + 2 * pad, int(D * px_per_m) + 2 * pad
    canvas = np.full((h, w, 3), 252, np.uint8)
    u = (pad + xyz[:, 0] * px_per_m).astype(int)
    v = (h - pad - xyz[:, 1] * px_per_m).astype(int)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    z = xyz[:, 2]
    high = ok & (z > 1.2)
    canvas[v[high], u[high]] = (200, 200, 200)
    low = ok & (z <= 1.2) & (z > 0.06)
    order = np.where(low)[0][np.argsort(z[low])]
    t = np.clip(z[order] / 1.2, 0, 1)
    canvas[v[order], u[order]] = np.stack([30 + 210 * t, 90 + 60 * t, 220 - 190 * t], 1).astype(np.uint8)
    img = Image.fromarray(canvas)
    dr = ImageDraw.Draw(img)
    for gx in np.arange(0, W + 0.01, 0.5):
        x = pad + gx * px_per_m
        dr.line([(x, pad), (x, h - pad)], fill=(215, 215, 210) if gx % 1 else (170, 170, 165))
        if gx % 1 == 0:
            dr.text((x - 4, h - pad + 6), f"{gx:.0f}", fill=(80, 80, 80), font=font(13))
    for gy in np.arange(0, D + 0.01, 0.5):
        y = h - pad - gy * px_per_m
        dr.line([(pad, y), (w - pad, y)], fill=(215, 215, 210) if gy % 1 else (170, 170, 165))
        if gy % 1 == 0:
            dr.text((pad - 22, y - 7), f"{gy:.0f}", fill=(80, 80, 80), font=font(13))
    dr.rectangle([pad, pad, w - pad, h - pad], outline=(40, 40, 40), width=2)
    for label, (x, y) in {"S": (w / 2, h - pad + 24), "N": (w / 2, pad - 26), "W": (pad - 40, h / 2), "E": (w - pad + 26, h / 2)}.items():
        dr.text((x, y), label, fill=(20, 20, 20), font=font(18))
    for i, cx, cy, ang in cams:
        c = (pad + cx * px_per_m, h - pad - cy * px_per_m)
        a = math.radians(ang)
        dr.line([c, (c[0] + math.sin(a) * 46, c[1] - math.cos(a) * 46)], fill=(255, 120, 0), width=2)
        dr.text((c[0] + math.sin(a) * 58 - 7, c[1] - math.cos(a) * 58 - 7), str(i), fill=(200, 70, 0), font=font(12))
    if cams:
        c = (pad + cams[0][1] * px_per_m, h - pad - cams[0][2] * px_per_m)
        dr.ellipse([c[0] - 6, c[1] - 6, c[0] + 6, c[1] + 6], fill=(255, 80, 0))
    dr.text((pad, 10), f"комната {W:.2f} × {D:.2f} м · сетка 0.5 м · серое — стены выше 1.2 м · цвет — ниже 1.2 м (синий пол → оранжевый 1.2 м) · лучи — кадры",
            fill=(60, 60, 60), font=font(13))
    img.save(path)


def overlap(frames, k: int, cam_h: float, ceiling: float) -> float:
    """Доля точек стен кадра k, попавших на стены остальных кадров. Мусорный кадр (лицо, пол, стол) — около нуля."""
    cell, size = 0.08, 200
    origin = np.array([-size * cell / 2, -size * cell / 2])
    others = [wall_ring(world(o["cam"][o["ok"]], o["yaw"], o["pitch"], o["scale"]), cam_h, ceiling)
              for j, o in enumerate(frames) if j != k]
    mine = wall_ring(world(frames[k]["cam"][frames[k]["ok"]], frames[k]["yaw"], frames[k]["pitch"], frames[k]["scale"]), cam_h, ceiling)
    if not others or len(mine) < 200:
        return 0.0
    occ = occupancy(np.concatenate(others), origin, size, cell, dilate=2)
    return grid_score(mine, occ, origin, cell) / len(mine)


def cloud_html(path: Path, xyz: np.ndarray, rgb: np.ndarray, title: str):
    tpl = (ROOT / "web" / "cloud.html").read_text()
    data = {"title": title, "n": len(xyz),
            "xyz": base64.b64encode(xyz.astype("<f4").tobytes()).decode(),
            "rgb": base64.b64encode(rgb.astype(np.uint8).tobytes()).decode()}
    path.write_text(tpl.replace("__CLOUD_JSON__", json.dumps(data)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--out", required=True)
    ap.add_argument("--drop", default="", help="номера кадров через запятую: брак, люди в кадре")
    ap.add_argument("--hfov", type=float, default=69.0)
    ap.add_argument("--cell", type=float, default=0.025, help="шаг прореживания облака, м")
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--keep-all", action="store_true", help="не выкидывать кадры, не совпавшие с соседями")
    ap.add_argument("--min-overlap", type=float, default=0.3)
    args = ap.parse_args()

    cap = json.load(open(args.capture))
    out = Path(args.out)
    drop = {int(x) for x in args.drop.split(",") if x.strip()} | set(cap.get("live_bad", []))
    ceiling = cap.get("ceiling_m") or 2.7
    sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])

    frames = []
    for fr, img in load_frames(cap, out, drop):
        d = cached_depth(sess, img, out / "depth" / f"{fr['i']:03d}.npy")
        f = focal_px(*img.size, args.hfov)
        ok_full = clean_mask(d)
        cam, vs, us = to_cam(d, f, step=4)
        rgb = np.asarray(img)[vs, us]
        frames.append({"i": fr["i"], "yaw": fr["yaw"], "yaw_gyro": fr["yaw"], "pitch": fr["pitch"], "scale": 1.0,
                       "cam": cam, "rgb": rgb, "ok": ok_full[vs, us]})
    print(f"кадров в работе: {len(frames)} (выкинуты: {sorted(drop) or 'нет'})")

    cam_h = floor_height(frames)
    print(f"телефон над полом: {cam_h:.2f} м")
    if not args.no_refine:
        refine_poses(frames, cam_h, ceiling)
        cam_h = floor_height(frames)
        for fr in frames:
            print(f"  кадр {fr['i']:02d}: курс {fr['yaw_gyro']:6.1f}° → {fr['yaw']:6.1f}°, масштаб глубины ×{fr['scale']:.2f}")
        print(f"телефон над полом после уточнения: {cam_h:.2f} м")
    if not args.keep_all:
        scores = [overlap(frames, k, cam_h, ceiling) for k in range(len(frames))]
        bad = [fr["i"] for fr, sc in zip(frames, scores) if sc < args.min_overlap]
        for fr, sc in zip(frames, scores):
            fr["overlap"] = round(sc, 2)
        if bad:
            print(f"не совпали с соседями (<{args.min_overlap:.0%}), выкинуты: {bad}")
            drop |= set(bad)
            frames = [fr for fr in frames if fr["i"] not in bad]

    xyz, rgb = [], []
    for fr in frames:
        p = world(fr["cam"], fr["yaw"], fr["pitch"], fr["scale"])
        p[..., 2] += cam_h
        sel = fr["ok"] & (p[..., 2] > -0.1) & (p[..., 2] < ceiling + 0.2)
        xyz.append(p[sel]); rgb.append(fr["rgb"][sel])
    xyz, rgb = np.concatenate(xyz), np.concatenate(rgb)
    raw = len(xyz)
    xyz, rgb = voxel_down(xyz, rgb, args.cell)
    print(f"облако: {raw} точек → {len(xyz)} после прореживания {args.cell * 100:.1f} см и выбросов")

    # прямоугольник комнаты по точкам стен → облако в координатах комнаты (угол SW в начале, как в room.json)
    walls = xyz[(xyz[:, 2] > 1.0) & (xyz[:, 2] < ceiling - 0.15)][:, :2]
    rect = fit_rectangle(walls[:: max(1, len(walls) // 20000)])
    print(f"комната {rect['x1'] - rect['x0']:.2f} × {rect['y1'] - rect['y0']:.2f} м, разворот {rect['angle_deg']:.1f}°, "
          f"разброс стен {rect['fit_error_m']:.2f} м")
    plan_png(out / "plan.png", xyz, frames, ceiling)
    xyz = np.concatenate([to_room(xyz[:, :2], rect), xyz[:, 2:]], axis=1).astype(np.float32)
    cam_xy = to_room(np.zeros((1, 2)), rect)[0]
    cams = [(f["i"], float(cam_xy[0]), float(cam_xy[1]), (f["yaw"] + rect["angle_deg"]) % 360) for f in frames]
    plan_room_png(out / "plan_room.png", xyz, rect, cams)
    write_ply(out / "cloud.ply", xyz, rgb)
    cloud_html(out / "cloud.html", xyz, rgb, cap.get("room") or "Комната")
    (out / "poses.json").write_text(json.dumps({
        "cam_height": round(cam_h, 3), "hfov": args.hfov,
        "room": {"width": round(rect["x1"] - rect["x0"], 2), "depth": round(rect["y1"] - rect["y0"], 2),
                 "angle_deg": rect["angle_deg"], "fit_error_m": round(rect["fit_error_m"], 3),
                 "camera_xy": [round(float(cam_xy[0]), 2), round(float(cam_xy[1]), 2)]}, "ceiling": ceiling, "dropped": sorted(drop),
        "frames": [{"i": f["i"], "yaw": round(f["yaw"], 2), "yaw_gyro": f["yaw_gyro"], "pitch": f["pitch"],
                    "scale": round(f["scale"], 3), "overlap": f.get("overlap"),
                    "yaw_room": round((f["yaw"] + rect["angle_deg"]) % 360, 1)} for f in frames]}, ensure_ascii=False, indent=2))
    print(f"готово: {out / 'cloud.html'}, {out / 'plan_room.png'}, {out / 'cloud.ply'}")


if __name__ == "__main__":
    main()
