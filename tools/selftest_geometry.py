"""Проверка геометрии без модели глубины: точная глубина для известной комнаты → размеры назад.

Комната W×D×H, камера в точке (cx, cy) на высоте cam_h, поворот на 360° с шагом 15°,
комната развёрнута на rot градусов относительно севера. Ожидаем те же W и D.
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from build_room import fit_rectangle, focal_px, frame_points, rotation, wall_samples  # noqa: E402

W, D, H = 7.0, 5.0, 2.7
CX, CY, CAM_H, ROT = 3.1, 2.2, 1.4, 23.0
w, h, hfov = 720, 1280, 69.0
f = focal_px(w, h, hfov)


def analytic_depth(yaw: float, pitch: float) -> np.ndarray:
    vs, us = np.mgrid[0:h, 0:w]
    rays = np.stack([(us - w / 2) / f, (vs - h / 2) / f, np.ones_like(us, dtype=float)], axis=-1)
    world = rays @ rotation(yaw, pitch).T  # z-компонента в камере = 1, значит t = z-depth
    # в систему комнаты: повернуть на ROT
    t = math.radians(ROT)
    dx = world[..., 0] * math.cos(t) - world[..., 1] * math.sin(t)
    dy = world[..., 0] * math.sin(t) + world[..., 1] * math.cos(t)
    dz = world[..., 2]
    ts = []
    with np.errstate(divide="ignore", invalid="ignore"):
        for num, den in [(0 - CX, dx), (W - CX, dx), (0 - CY, dy), (D - CY, dy), (-CAM_H, dz), (H - CAM_H, dz)]:
            tt = num / den
            ts.append(np.where(tt > 0, tt, np.inf))
    return np.min(ts, axis=0).astype(np.float32)


walls = []
for yaw in range(0, 360, 15):
    d = analytic_depth(yaw, -5.0)
    pts = frame_points(d, f, yaw, -5.0, CAM_H)
    walls.append(wall_samples(pts, H))
rect = fit_rectangle(np.concatenate(walls))
got_w, got_d = sorted([rect["x1"] - rect["x0"], rect["y1"] - rect["y0"]], reverse=True)
print(f"ждём {W} × {D}, получили {got_w:.2f} × {got_d:.2f}, угол {rect['angle_deg']:.1f}° (ждём {(90 - ROT) % 90:.1f} или {ROT:.1f}), разброс {rect['fit_error_m']:.3f} м")
ok = abs(got_w - W) < 0.15 and abs(got_d - D) < 0.15
print("OK" if ok else "FAIL")
sys.exit(0 if ok else 1)
