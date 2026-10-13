"""room.json → room.html рядом с ним (после ручной правки room.json, без повторной разметки).

  python scripts/render_html.py out/гостиная/room.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

src = Path(sys.argv[1])
room = json.loads(src.read_text())
viewer = (ROOT / "web" / "viewer.html").read_text()
dst = src.with_suffix(".html")
dst.write_text(viewer.replace("__ROOM_JSON__", json.dumps(room, ensure_ascii=False)))
print(dst)
