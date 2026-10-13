"""Контактный лист кадров для разметки агентом: номер кадра, курс в координатах комнаты, наклон, брак.

  python scripts/sheet.py out/<скан>        # → out/<скан>/sheet.jpg

Курс 0° смотрит на стену N, 90° — на E, 180° — на S, 270° — на W (как лучи на plan_room.png).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cloud import font  # noqa: E402


def wall_of(yaw: float) -> str:
    return "NESW"[int(((yaw + 45) % 360) // 90)]


def main():
    out = Path(sys.argv[1])
    poses = json.loads((out / "poses.json").read_text())
    used = {f["i"]: f for f in poses["frames"]}
    paths = sorted((out / "frames").glob("*.jpg"))
    tw, th, cols = 240, 427, 6
    rows = (len(paths) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * (th + 26)), (20, 20, 20))
    dr = ImageDraw.Draw(sheet)
    for k, p in enumerate(paths):
        i = int(p.stem)
        im = Image.open(p).convert("RGB")
        im.thumbnail((tw, th))
        x, y = (k % cols) * tw, (k // cols) * (th + 26)
        sheet.paste(im, (x, y))
        f = used.get(i)
        if f:
            text, col = f"#{i}  курс {f['yaw_room']:.0f}° → {wall_of(f['yaw_room'])}  наклон {f['pitch']:+.0f}°", (230, 230, 230)
        else:
            text, col = f"#{i}  выкинут", (255, 110, 90)
            dr.line([(x, y), (x + im.width, y + im.height)], fill=col, width=3)
        dr.text((x + 6, y + th + 5), text, fill=col, font=font(14))
    sheet.save(out / "sheet.jpg", quality=85)
    print(out / "sheet.jpg")


if __name__ == "__main__":
    main()
