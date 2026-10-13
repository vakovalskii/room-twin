"""HTTPS-сервер для съёмки с телефона в той же Wi-Fi сети. Телефон — только камера, всё считает компьютер.

Живой скан: POST /live/start → кадры по одному POST /live/<id>/frame → компьютер сразу считает глубину,
бракует кадры и рисует план; телефон опрашивает GET /live/<id>/state и /live/<id>/plan.png.
POST /live/<id>/finish → captures/room-capture-<id>.json и сборка облака (scripts/cloud.py) в out/<id>/.
Комнату по облаку и кадрам дальше собирает агент (SKILL.md).

Старый путь тоже жив: POST /upload целым файлом.
Камера и гироскоп в браузере работают только по https, поэтому сертификат
самоподписанный: на телефоне один раз «Всё равно перейти».

  uv run --with numpy --with onnxruntime --with pillow python scripts/serve_capture.py   # https://<ip>:8443/capture.html
"""
from __future__ import annotations

import datetime as dt
import http.server
import json
import socket
import ssl
import subprocess
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from live import LiveScan, session  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
CAPTURES = ROOT / "captures"
CERT_DIR = Path.home() / ".cache" / "room-twin"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8443
MAX_BYTES = 200 * 1024 * 1024


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.0.2.1", 9))  # пакет не уходит, просто выбирается интерфейс
        return s.getsockname()[0]
    finally:
        s.close()


def ensure_cert(ip: str) -> tuple[Path, Path]:
    CERT_DIR.mkdir(parents=True, exist_ok=True)
    cert, key, made_for = CERT_DIR / "cert.pem", CERT_DIR / "key.pem", CERT_DIR / "cert.ip"
    # сертификат выписан на IP мака: сменилась сеть — выписываем заново
    if not cert.exists() or not made_for.exists() or made_for.read_text() != ip:
        subprocess.run([
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "365",
            "-keyout", str(key), "-out", str(cert), "-subj", "/CN=room-twin",
            "-addext", f"subjectAltName=IP:{ip},DNS:localhost",
        ], check=True, capture_output=True)
        made_for.write_text(ip)
    return cert, key


OUT = ROOT / "out"
BUILDS: dict[str, str] = {}  # имя скана → building / done / error
LIVE: dict[str, LiveScan] = {}


def build(capture: Path):
    """Облако точек и план на этом компьютере сразу после съёмки. Комнату по ним собирает агент."""
    stem = capture.stem.replace("room-capture-", "")
    dst = OUT / stem
    dst.mkdir(parents=True, exist_ok=True)
    BUILDS[stem] = "building"
    with open(dst / "build.log", "w") as log:
        code = subprocess.run(
            [sys.executable, "-u", str(ROOT / "scripts" / "cloud.py"), str(capture), "--out", str(dst)],
            stdout=log, stderr=subprocess.STDOUT, cwd=ROOT).returncode
    BUILDS[stem] = "done" if code == 0 and (dst / "cloud.html").exists() else "error"
    print(f"облако {stem}: {BUILDS[stem]} — https://localhost:{PORT}/out/{stem}/cloud.html", flush=True)


def sessions() -> dict:
    """Для страницы статуса на компьютере: живые сканы и всё, что уже собрано в out/."""
    live = [{**scan.state(), "build": BUILDS.get(sid)} for sid, scan in LIVE.items()]
    done = []
    for d in sorted(OUT.iterdir(), reverse=True) if OUT.exists() else []:
        if d.is_dir() and d.name not in LIVE:
            log = d / "build.log"
            done.append({"id": d.name, "build": BUILDS.get(d.name) or ("done" if (d / "cloud.html").exists() else None),
                         "files": [f for f in ("cloud.html", "room.html", "plan_room.png", "sheet.jpg") if (d / f).exists()],
                         "last": log.read_text(errors="ignore").strip().splitlines()[-1:] if log.exists() else []})
    return {"live": live[::-1], "done": done[:12]}


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(WEB), **kw)

    def translate_path(self, path):
        # /out/<скан>/room.html — готовые двойники, остальное — страницы из web/
        if path.startswith("/out/"):
            rel = Path(path.split("?", 1)[0][len("/out/"):])
            target = (OUT / rel).resolve()
            return str(target) if OUT.resolve() in target.parents else str(OUT / "missing")
        return super().translate_path(path)

    def reply(self, code: int, body: bytes, ctype: str = "application/json; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def live(self, sid: str) -> LiveScan | None:
        return LIVE.get(sid)

    def do_GET(self):
        parts = self.path.split("?", 1)[0].strip("/").split("/")
        if len(parts) == 3 and parts[0] == "live" and (scan := self.live(parts[1])):
            if parts[2] == "state":
                return self.reply(200, json.dumps(scan.state(), ensure_ascii=False).encode())
            if parts[2] == "plan.png" and scan.plan_png:
                return self.reply(200, scan.plan_png, "image/png")
            return self.send_error(404)
        if parts == ["api", "sessions"]:
            return self.reply(200, json.dumps(sessions(), ensure_ascii=False).encode())
        if self.path.startswith("/status/"):
            stem = self.path[len("/status/"):]
            log = OUT / stem / "build.log"
            tail = log.read_text(errors="ignore").strip().splitlines()[-1:] if log.exists() else []
            reply = json.dumps({"state": BUILDS.get(stem, "unknown"), "last": tail[0] if tail else ""},
                               ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)
            return
        super().do_GET()

    def do_POST(self):
        size = int(self.headers.get("Content-Length", 0))
        if not 0 <= size <= MAX_BYTES:
            return self.send_error(413)
        body = self.rfile.read(size) if size else b"{}"
        parts = self.path.strip("/").split("/")
        if parts[0] == "live":
            return self.post_live(parts, body)
        if self.path != "/upload" or not size:
            return self.send_error(404)
        try:
            cap = json.loads(body)
            assert cap.get("format", "").startswith("room-twin-capture/")
        except Exception:
            return self.send_error(400, "не скан room-twin")
        CAPTURES.mkdir(exist_ok=True)
        name = f"room-capture-{dt.datetime.now():%Y-%m-%d-%H%M%S}.json"
        (CAPTURES / name).write_bytes(body)
        print(f"принят скан: {CAPTURES / name} — {len(cap.get('frames', []))} кадров, «{cap.get('room')}»", flush=True)
        stem = Path(name).stem.replace("room-capture-", "")
        threading.Thread(target=build, args=(CAPTURES / name,), daemon=True).start()
        reply = json.dumps({"ok": True, "file": name, "id": stem}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)


    def post_live(self, parts: list[str], body: bytes):
        try:
            data = json.loads(body)
        except Exception:
            return self.send_error(400)
        if parts == ["live", "start"]:
            sid = f"{dt.datetime.now():%Y-%m-%d-%H%M%S}"
            meta = {k: data.get(k) for k in ("room", "ceiling_m", "created", "user_agent", "camera")}
            LIVE[sid] = LiveScan(sid, OUT / sid, meta, session())
            print(f"живой скан {sid}: «{meta.get('room')}»", flush=True)
            return self.reply(200, json.dumps({"id": sid}).encode())
        scan = self.live(parts[1]) if len(parts) == 3 else None
        if not scan:
            return self.send_error(404)
        if parts[2] == "frame":
            if not str(data.get("jpeg", "")).startswith("data:image/jpeg;base64,"):
                return self.send_error(400)
            i = scan.add({k: data.get(k) for k in ("t", "auto", "yaw", "pitch", "alpha", "beta", "gamma",
                                                    "screen_angle", "width", "height", "jpeg", "row")})
            return self.reply(200, json.dumps({"i": i}).encode())
        if parts[2] == "camera":
            scan.meta["camera"] = data
            return self.reply(200, b"{}")
        if parts[2] == "finish":
            CAPTURES.mkdir(exist_ok=True)
            cap = scan.finish(CAPTURES)
            print(f"живой скан {scan.sid} закончен: {len(scan.frames)} кадров → {cap}", flush=True)
            threading.Thread(target=build, args=(cap,), daemon=True).start()
            return self.reply(200, json.dumps({"id": scan.sid}).encode())
        return self.send_error(404)


def main():
    ip = lan_ip()
    cert, key = ensure_cert(ip)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert, key)
    srv = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
    print(f"Открой на телефоне: https://{ip}:{PORT}/capture.html")
    print(f"Статус на компьютере: https://localhost:{PORT}/monitor.html")
    print("Сертификат самоподписанный: «Подробнее» → «Перейти на сайт». Сканы падают в", CAPTURES)
    srv.serve_forever()


if __name__ == "__main__":
    main()
