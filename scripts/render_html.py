"""room.json → room.html рядом с ним (после ручной правки room.json, без повторной разметки).

  python scripts/render_html.py out/гостиная/room.json
"""
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def embed_json(data) -> str:
    """JSON для вставки внутрь <script type="application/json">: «<» экранируем, иначе название
    вида «</script><img onerror=…>» закроет блок и выполнится как разметка."""
    return (json.dumps(data, ensure_ascii=False)
            .replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def write_atomic(path: Path, text: str):
    """Пишем во временный файл рядом и подменяем: читатель никогда не увидит обрезанный файл."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def render(room: dict, dst: Path):
    viewer = (ROOT / "web" / "viewer.html").read_text()
    write_atomic(dst, viewer.replace("__ROOM_JSON__", embed_json(room)))


if __name__ == "__main__":
    src = Path(sys.argv[1])
    dst = src.with_suffix(".html")
    render(json.loads(src.read_text()), dst)
    print(dst)
