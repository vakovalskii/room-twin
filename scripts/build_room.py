"""Скан комнаты (room-capture-*.json со страницы capture.html) → модель комнаты room.json.

Шаги:
  1. метрическая глубина на каждый кадр (Depth Anything V2 Metric Indoor Small, ONNX);
  2. кадр + направление камеры → точки в комнате; дальние точки каждого столбца — стены;
  3. прямоугольник комнаты по точкам стен (комнаты почти всегда прямоугольные);
  4. vision-модель на NeuralDeep размечает предметы и проёмы рамками на кадрах;
  5. центр рамки + глубина под ней + направление камеры → место на плане, размер рамки → размер.

Запуск:
  uv run --with numpy --with onnxruntime --with pillow python scripts/build_room.py capture.json --out out/
Ключ NeuralDeep: NEURALDEEP_API_KEY, иначе берётся из ~/.config/opencode/opencode.json.
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import io
import json
import math
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "da2-metric-indoor-small.onnx"
ND_BASE = os.getenv("ND_BASE_URL", "https://api.neuraldeep.ru/v1").rstrip("/")
VISION_MODEL = os.getenv("NDT_VISION_MODEL", "qwen3.6-fp8-noreason")
WORKERS = int(os.getenv("ANALYZE_WORKERS", "4"))

CATEGORIES = [
    "sofa", "sofa_corner", "armchair", "chair", "stool", "table", "coffee_table", "desk", "bed",
    "wardrobe", "cabinet", "wall_cabinet", "shelf", "bookcase", "tv", "tv_stand", "rug",
    "kitchen_counter", "kitchen_island", "bar_counter", "fridge", "oven_column", "stove", "sink",
    "lamp_pendant", "lamp_floor", "lamp_table", "plant", "radiator", "column", "ac_unit",
    "curtain", "mirror", "picture", "speaker", "pc", "other",
]

# Типичная глубина предмета (вдоль взгляда), её по кадру не видно, м
DEPTH_GUESS = {
    "sofa": 0.95, "sofa_corner": 1.6, "armchair": 0.85, "chair": 0.5, "stool": 0.4, "table": 0.9,
    "coffee_table": 0.6, "desk": 0.7, "bed": 2.0, "wardrobe": 0.6, "cabinet": 0.45, "wall_cabinet": 0.35,
    "shelf": 0.25, "bookcase": 0.35, "tv": 0.08, "tv_stand": 0.4, "rug": 1.6, "kitchen_counter": 0.6,
    "kitchen_island": 0.9, "bar_counter": 0.5, "fridge": 0.65, "oven_column": 0.6, "stove": 0.6,
    "sink": 0.5, "lamp_pendant": 0.25, "lamp_floor": 0.35, "lamp_table": 0.25, "plant": 0.45,
    "radiator": 0.1, "column": 0.4, "ac_unit": 0.25, "curtain": 0.1, "mirror": 0.05, "picture": 0.04,
    "speaker": 0.3, "pc": 0.45, "other": 0.5,
}


# Перечень (enum) в json_schema на шлюзе выбирает категорию криво («диван» с категорией fridge),
# поэтому категория — свободная строка. Вне списка — сводим по русскому названию.
NAME_RULES = [
    ("углов", "sofa_corner"), ("диван", "sofa"), ("кресл", "armchair"), ("табур", "stool"), ("стул", "chair"),
    ("журнальн", "coffee_table"), ("письменн", "desk"), ("рабочий стол", "desk"), ("стол", "table"),
    ("кроват", "bed"), ("гардероб", "wardrobe"), ("навесн", "wall_cabinet"), ("книжн", "bookcase"),
    ("стеллаж", "bookcase"), ("полк", "shelf"), ("тумба под", "tv_stand"), ("тв-тумб", "tv_stand"),
    ("телевиз", "tv"), ("ковёр", "rug"), ("ковер", "rug"), ("остров", "kitchen_island"),
    ("барн", "bar_counter"), ("столешниц", "kitchen_counter"), ("кухонн", "kitchen_counter"),
    ("холодил", "fridge"), ("духов", "oven_column"), ("плит", "stove"), ("раковин", "sink"),
    ("мойк", "sink"), ("подвесн", "lamp_pendant"), ("люстр", "lamp_pendant"), ("торшер", "lamp_floor"),
    ("настольн", "lamp_table"), ("светильник", "lamp_pendant"), ("растен", "plant"), ("цвет", "plant"),
    ("радиатор", "radiator"), ("батаре", "radiator"), ("колонн", "column"), ("кондиц", "ac_unit"),
    ("штор", "curtain"), ("зеркал", "mirror"), ("картин", "picture"), ("колонк", "speaker"),
    ("компьют", "pc"), ("шкаф", "cabinet"), ("тумб", "cabinet"),
]


def normalize_category(name: str, category: str) -> str:
    category = category.strip().lower().replace(" ", "_")
    if category in CATEGORIES and category != "other":
        return category
    low = name.lower()
    for key, cat in NAME_RULES:
        if key in low:
            return cat
    return "other"


# ── ключ и вызов vision-модели ───────────────────────────────────────────────
def api_key() -> str:
    key = os.getenv("NEURALDEEP_API_KEY", "")
    if key:
        return key
    try:
        cfg = json.load(open(Path.home() / ".config/opencode/opencode.json"))
        return cfg["provider"]["neuraldeep"]["options"]["apiKey"]
    except Exception:
        sys.exit("Нет ключа: задай NEURALDEEP_API_KEY")


DETECT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {"name": "RoomFrame", "strict": True, "schema": {
        "type": "object", "additionalProperties": False,
        "properties": {
            "objects": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                # порядок полей = порядок генерации: сначала название, потом категория по нему,
                # иначе модель выбирает категорию вслепую («диван» с категорией fridge)
                "properties": {
                    "name": {"type": "string"},
                    "category": {"type": "string"},
                    "color": {"type": "string"},
                    "material": {"type": "string"},
                    "mount": {"type": "string", "enum": ["floor", "wall", "ceiling"]},
                    "is_light": {"type": "boolean"},
                    "bbox": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["name", "category", "color", "material", "mount", "is_light", "bbox"]}},
            "openings": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "type": {"type": "string", "enum": ["door", "window", "opening"]},
                    "leads_to": {"type": "string"},
                    "bbox": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["type", "leads_to", "bbox"]}},
        },
        "required": ["objects", "openings"]}},
}

DETECT_PROMPT = (
    "Это кадр из скана комнаты: человек стоит в центре и поворачивается. Найди на кадре мебель, "
    "технику, светильники, ковры, а также двери, окна и проёмы. Для каждого дай рамку bbox = "
    "[x1, y1, x2, y2] в координатах от 0 до 1000 относительно ширины и высоты кадра, по видимой "
    "части предмета. name — короткое русское название («угловой диван», «подвесной светильник»), "
    "color — цвет в HEX (#3a3f3c), material — материал по-русски. mount: floor — стоит на полу, "
    "wall — висит на стене, ceiling — подвешен к потолку. Мелочи (книги, кружки, провода) не нужны. "
    "Предмет, обрезанный краем кадра, всё равно отмечай. category — одно из: " + ", ".join(CATEGORIES) + "."
)


def detect(jpeg_data_url: str, key: str) -> dict:
    body = json.dumps({
        "model": VISION_MODEL, "temperature": 0, "max_tokens": 2500,
        "response_format": DETECT_SCHEMA,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": jpeg_data_url}},
            {"type": "text", "text": DETECT_PROMPT}]}],
    }).encode()
    req = urllib.request.Request(f"{ND_BASE}/chat/completions", data=body, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    for attempt in range(3):
        try:
            data = json.load(urllib.request.urlopen(req, timeout=180))
            return json.loads(data["choices"][0]["message"]["content"])
        except Exception as err:  # сеть или кривой JSON: ещё раз
            last = err
    print(f"  разметка кадра не удалась: {last}", file=sys.stderr)
    return {"objects": [], "openings": []}


# ── глубина ─────────────────────────────────────────────────────────────────
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def depth_map(sess: ort.InferenceSession, img: Image.Image, long_side: int = 518) -> np.ndarray:
    """Глубина в метрах с разрешением кадра (по оси камеры, z-depth)."""
    w, h = img.size
    k = long_side / max(w, h)
    nw, nh = max(14, round(w * k / 14) * 14), max(14, round(h * k / 14) * 14)
    x = (np.asarray(img.convert("RGB").resize((nw, nh), Image.BICUBIC), dtype=np.float32) / 255 - MEAN) / STD
    out = sess.run(None, {"pixel_values": x.transpose(2, 0, 1)[None]})[0][0]
    return np.asarray(Image.fromarray(out).resize((w, h), Image.BILINEAR), dtype=np.float32)


# ── геометрия ───────────────────────────────────────────────────────────────
def rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Камера (x вправо, y вниз, z вперёд) → комната (x восток, y север, z вверх)."""
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    fwd = np.array([math.sin(y) * math.cos(p), math.cos(y) * math.cos(p), math.sin(p)])
    right = np.array([math.cos(y), -math.sin(y), 0.0])
    down = np.cross(fwd, right)  # fwd × right смотрит вниз для камеры без крена
    return np.stack([right, down, fwd], axis=1)


def focal_px(w: int, h: int, hfov_long_deg: float) -> float:
    return (max(w, h) / 2) / math.tan(math.radians(hfov_long_deg) / 2)


def frame_points(depth: np.ndarray, f: float, yaw: float, pitch: float, cam_h: float, step: int = 6) -> np.ndarray:
    """Точки кадра в координатах комнаты (камера в начале, пол на z = -cam_h)."""
    h, w = depth.shape
    vs, us = np.mgrid[0:h:step, 0:w:step]
    z = depth[vs, us]
    pts_cam = np.stack([(us - w / 2) * z / f, (vs - h / 2) * z / f, z], axis=-1)
    pts = pts_cam @ rotation(yaw, pitch).T
    pts[..., 2] += cam_h
    return pts  # [rows, cols, 3]; z — высота над полом


def wall_samples(pts: np.ndarray, ceiling: float) -> np.ndarray:
    """По столбцу кадра: точка стены = самая дальняя по горизонтали между 0.3 м и потолком."""
    out = []
    for col in range(pts.shape[1]):
        p = pts[:, col]
        ok = (p[:, 2] > 0.3) & (p[:, 2] < ceiling - 0.1)
        if ok.sum() < 3:
            continue
        r = np.hypot(p[ok, 0], p[ok, 1])
        i = int(np.argsort(r)[int(len(r) * 0.9) - 1 if len(r) > 1 else 0])
        if 0.4 < r[i] < 15:
            out.append(p[ok][i, :2])
    return np.array(out)


def fit_rectangle(xy: np.ndarray) -> dict:
    """Прямоугольник комнаты: угол поворота и стороны по перцентилям, лучший по сумме расстояний до сторон."""
    best = None
    for deg in np.arange(0, 90, 0.5):
        t = math.radians(deg)
        rot = xy @ np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
        x0, x1 = np.percentile(rot[:, 0], [3, 97])
        y0, y1 = np.percentile(rot[:, 1], [3, 97])
        d = np.minimum.reduce([np.abs(rot[:, 0] - x0), np.abs(rot[:, 0] - x1), np.abs(rot[:, 1] - y0), np.abs(rot[:, 1] - y1)])
        score = float(np.median(d))
        if best is None or score < best[0]:
            best = (score, deg, x0, x1, y0, y1)
    score, deg, x0, x1, y0, y1 = best
    return {"angle_deg": float(deg), "x0": float(x0), "x1": float(x1), "y0": float(y0), "y1": float(y1), "fit_error_m": score}


def to_room(xy: np.ndarray, rect: dict) -> np.ndarray:
    """Точки плана → координаты комнаты: начало в углу, x вдоль ширины, y вдоль глубины."""
    t = math.radians(rect["angle_deg"])
    rot = xy @ np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    return rot - np.array([rect["x0"], rect["y0"]])


# ── предметы ────────────────────────────────────────────────────────────────
def lift(det: dict, depth: np.ndarray, pts_full_f: float, frame: dict, cam_h: float) -> dict | None:
    h, w = depth.shape
    x1, y1, x2, y2 = [max(0, min(1000, int(v))) for v in (det["bbox"] + [0, 0, 0, 0])[:4]]
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    u1, u2, v1, v2 = x1 * w // 1000, x2 * w // 1000, y1 * h // 1000, y2 * h // 1000
    # глубина по внутренней части рамки: края часто ловят фон
    cu1, cu2 = u1 + (u2 - u1) // 4, u2 - (u2 - u1) // 4
    cv1, cv2 = v1 + (v2 - v1) // 4, v2 - (v2 - v1) // 4
    z = float(np.median(depth[cv1:max(cv1 + 1, cv2), cu1:max(cu1 + 1, cu2)]))
    f = pts_full_f
    uc, vc = (u1 + u2) / 2, (v1 + v2) / 2
    centre_cam = np.array([(uc - w / 2) * z / f, (vc - h / 2) * z / f, z])
    rot = rotation(frame["yaw"], frame["pitch"])
    c = rot @ centre_cam
    top = rot @ np.array([(uc - w / 2) * z / f, (v1 - h / 2) * z / f, z])
    bottom = rot @ np.array([(uc - w / 2) * z / f, (v2 - h / 2) * z / f, z])
    width = (u2 - u1) * z / f
    return {
        "category": det["category"], "name": det["name"], "color": det["color"], "material": det["material"],
        "mount": det["mount"], "is_light": det["is_light"],
        "xy": c[:2], "z_top": float(top[2] + cam_h), "z_bottom": float(bottom[2] + cam_h),
        "width": float(width), "seen_from": frame["i"],
        "cut": x1 < 15 or x2 > 985,  # обрезан краем кадра: ширина занижена
    }


def merge(items: list[dict]) -> list[dict]:
    """Один предмет виден с нескольких кадров: склеиваем по категории и расстоянию."""
    groups: list[list[dict]] = []
    for it in items:
        for g in groups:
            ref = g[0]
            same = it["category"] == ref["category"] or {it["category"], ref["category"]} <= {"sofa", "sofa_corner"}
            near = np.linalg.norm(np.mean([x["xy"] for x in g], axis=0) - it["xy"]) < max(0.7, ref["width"] / 2)
            if same and near:
                g.append(it)
                break
        else:
            groups.append([it])
    out = []
    for g in groups:
        whole = [x for x in g if not x["cut"]] or g
        best = max(whole, key=lambda x: x["width"])
        out.append({**best, "xy": np.mean([x["xy"] for x in g], axis=0), "views": len(g),
                    "z_top": float(np.median([x["z_top"] for x in g])),
                    "z_bottom": float(np.median([x["z_bottom"] for x in g]))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--out", default="out")
    ap.add_argument("--hfov", type=float, default=float(os.getenv("HFOV_LONG", "69")),
                    help="угол обзора камеры по длинной стороне кадра, градусы (iPhone 1x ≈ 69)")
    ap.add_argument("--cam-height", type=float, default=1.4, help="высота телефона над полом, м")
    ap.add_argument("--no-vlm", action="store_true", help="только стены, без разметки предметов")
    args = ap.parse_args()

    cap = json.load(open(args.capture))
    out = Path(args.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    ceiling = cap.get("ceiling_m") or 2.7
    frames = cap["frames"]
    print(f"{cap.get('room')}: {len(frames)} кадров, потолок {ceiling} м{' (по умолчанию)' if not cap.get('ceiling_m') else ''}")

    if not MODEL.exists():
        sys.exit(f"Нет модели глубины {MODEL}. Один раз сконвертируй официальную:\n"
                 "  uv run --python 3.12 --with torch --with transformers --with onnx --with onnxruntime "
                 "python tools/export_depth.py models/da2-metric-indoor-small.onnx")
    sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    walls, depths, images = [], [], []
    for fr in frames:
        img = Image.open(io.BytesIO(base64.b64decode(fr["jpeg"].split(",", 1)[1])))
        img.save(out / "frames" / f"{fr['i']:03d}.jpg", quality=85)
        d = depth_map(sess, img)
        f = focal_px(*img.size, args.hfov)
        pts = frame_points(d, f, fr["yaw"], fr["pitch"], args.cam_height)
        ws = wall_samples(pts, ceiling)
        if len(ws):
            walls.append(ws)
        depths.append((d, f))
        images.append(img)
        print(f"  кадр {fr['i']:02d}: курс {fr['yaw']:6.1f}°, наклон {fr['pitch']:5.1f}°, глубина {np.percentile(d, 5):.1f}–{np.percentile(d, 95):.1f} м")

    xy = np.concatenate(walls)
    rect = fit_rectangle(xy)
    width, depth_m = rect["x1"] - rect["x0"], rect["y1"] - rect["y0"]
    print(f"комната {width:.2f} × {depth_m:.2f} м, разворот {rect['angle_deg']:.1f}°, разброс стен {rect['fit_error_m']:.2f} м")

    objects, openings = [], []
    if not args.no_vlm:
        key = api_key()
        print(f"разметка предметов ({VISION_MODEL}, {len(frames)} кадров)…")
        with cf.ThreadPoolExecutor(WORKERS) as pool:
            dets = list(pool.map(lambda fr: detect(fr["jpeg"], key), frames))
        lifted, opened = [], []
        for fr, det, (d, f) in zip(frames, dets, depths):
            for o in det["objects"]:
                if any(w in o["name"].lower() for w in ("двер", "окно", "проём", "проем")):
                    det["openings"].append({"type": "window" if "окно" in o["name"].lower() else "door",
                                            "leads_to": "", "bbox": o["bbox"]})
                    continue
                o = {**o, "category": normalize_category(o["name"], o["category"])}
                it = lift(o, d, f, fr, args.cam_height)
                if it:
                    lifted.append(it)
            for o in det["openings"]:
                it = lift({**o, "category": o["type"], "name": o["leads_to"], "color": "", "material": "",
                           "mount": "wall", "is_light": False}, d, f, fr, args.cam_height)
                if it:
                    opened.append(it)
        objects, openings = merge(lifted), merge(opened)

    def place(it: dict) -> dict:
        x, y = to_room(np.array([it["xy"]]), rect)[0]
        x, y = float(np.clip(x, 0, width)), float(np.clip(y, 0, depth_m))
        return {"x": round(x, 2), "y": round(y, 2)}

    def wall_of(pos: dict) -> str:
        d = {"S": pos["y"], "N": depth_m - pos["y"], "W": pos["x"], "E": width - pos["x"]}
        return min(d, key=d.get)

    room = {
        "format": "room-twin/1",
        "name": cap.get("room") or "Комната",
        "size": {"width": round(width, 2), "depth": round(depth_m, 2), "height": round(ceiling, 2)},
        "fit_error_m": round(rect["fit_error_m"], 3),
        "objects": [],
        "openings": [],
        "frames": [{"i": fr["i"], "yaw": fr["yaw"], "pitch": fr["pitch"]} for fr in frames],
    }
    for n, it in enumerate(sorted(objects, key=lambda o: -o["width"]), 1):
        pos = place(it)
        h = max(0.05, it["z_top"] - (0 if it["mount"] == "floor" else it["z_bottom"]))
        room["objects"].append({
            "id": f"O{n:02d}", "category": it["category"], "name": it["name"], "color": it["color"] or "#8a8a80",
            "material": it["material"], "mount": it["mount"], "is_light": it["is_light"], **pos,
            "w": round(it["width"], 2), "d": DEPTH_GUESS.get(it["category"], 0.5),
            "h": round(min(h, ceiling), 2),
            "elevation": round(0 if it["mount"] == "floor" else max(0.0, it["z_bottom"]), 2),
            "wall": wall_of(pos), "views": it["views"],
        })
    for n, it in enumerate(openings, 1):
        pos = place(it)
        room["openings"].append({
            "id": f"P{n:02d}", "type": it["category"], "leads_to": it["name"], "wall": wall_of(pos), **pos,
            "w": round(min(it["width"], 3.0), 2), "h": round(min(it["z_top"], ceiling), 2),
            "elevation": round(max(0.0, it["z_bottom"]) if it["category"] == "window" else 0.0, 2),
        })
    (out / "room.json").write_text(json.dumps(room, ensure_ascii=False, indent=2))
    viewer = (ROOT / "web" / "viewer.html").read_text()
    (out / "room.html").write_text(viewer.replace("__ROOM_JSON__", json.dumps(room, ensure_ascii=False)))
    np.save(out / "wall_points.npy", to_room(xy, rect))
    print(f"готово: {out / 'room.html'} (данные в {out / 'room.json'}) — предметов {len(room['objects'])}, проёмов {len(room['openings'])}")


if __name__ == "__main__":
    main()
