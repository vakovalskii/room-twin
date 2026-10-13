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
import os
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
PHONES: dict[str, dict] = {}  # IP телефона → последний пинг страницы съёмки


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
    for sid in [sid for sid in LIVE if not (OUT / sid).exists()]:  # папку удалили — сессия больше не живая
        LIVE.pop(sid)
    live = [{**scan.state(), "build": BUILDS.get(sid)} for sid, scan in LIVE.items()]
    now = dt.datetime.now().timestamp()
    phones = [{**p, "ip": ip, "age": round(now - p["seen"], 1)} for ip, p in PHONES.items() if now - p["seen"] < 120]
    done = []
    for d in sorted(OUT.iterdir(), reverse=True) if OUT.exists() else []:
        if d.is_dir() and d.name not in LIVE:
            log = d / "build.log"
            done.append({"id": d.name, "build": BUILDS.get(d.name) or ("done" if (d / "cloud.html").exists() else None),
                         "files": [f for f in ("cloud.html", "room.html", "plan_room.png", "sheet.jpg") if (d / f).exists()],
                         "room": json.loads((d / "room.json").read_text()).get("name") if (d / "room.json").exists() else
                                 (json.loads((d / "meta.json").read_text()).get("room") if (d / "meta.json").exists() else None),
                         "last": log.read_text(errors="ignore").strip().splitlines()[-1:] if log.exists() else []})
    return {"phones": phones, "live": live[::-1], "done": done[:12]}


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(WEB), **kw)

    def log_message(self, fmt, *args):
        # опросы статуса и пинги телефона — каждую секунду, в лог не пишем
        if any(x in self.path for x in ("/state", "/ping", "/api/sessions", "/status/", "plan.png", "cloud.bin", "/frames/")):
            return
        super().log_message(fmt, *args)

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
            if parts[2] == "cloud.bin" and scan.cloud_bin:
                return self.reply(200, scan.cloud_bin, "application/octet-stream")
            return self.send_error(404)
        if parts == ["api", "info"]:
            return self.reply(200, json.dumps({"phone_url": f"https://{lan_ip()}:{PORT}/capture.html"}).encode())
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
        if parts == ["phone", "ping"]:
            try:
                data = json.loads(body)
            except Exception:
                return self.send_error(400)
            ip = self.client_address[0]
            first = ip not in PHONES or dt.datetime.now().timestamp() - PHONES[ip]["seen"] > 30
            PHONES[ip] = {"seen": dt.datetime.now().timestamp(), "stage": str(data.get("stage", ""))[:20],
                          "live": data.get("live"), "frames": data.get("frames"), "device": str(data.get("device", ""))[:80],
                          "torch": bool(data.get("torch"))}
            if first:
                print(f"телефон подключился: {ip} · {PHONES[ip]['device']}", flush=True)
            return self.reply(200, b"{}")
        if parts[0] == "live":
            return self.post_live(parts, body)
        if len(parts) == 3 and parts[0] == "out" and parts[2] == "room.json":
            return self.save_room(parts[1], body)
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


    def save_room(self, stem: str, body: bytes):
        """Редактор во вьюере сохраняет room.json и пересобирает room.html рядом."""
        dst = (OUT / stem).resolve()
        if dst.parent != OUT.resolve() or not dst.is_dir():
            return self.send_error(404)
        try:
            room = json.loads(body)
            assert room.get("format") == "room-twin/1" and isinstance(room.get("objects"), list)
        except Exception:
            return self.send_error(400, "не room.json")
        text = json.dumps(room, ensure_ascii=False, indent=2)
        (dst / "room.json").write_text(text)
        viewer = (WEB / "viewer.html").read_text()
        (dst / "room.html").write_text(viewer.replace("__ROOM_JSON__", json.dumps(room, ensure_ascii=False)))
        print(f"комната {stem}: сохранена из редактора ({len(room['objects'])} предметов)", flush=True)
        return self.reply(200, b'{"ok": true}')

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
    if os.getenv("ROOM_TWIN_HTTP") == "1":  # для автотестов в браузере без сертификата; телефону нужен https
        print(f"ВНИМАНИЕ: http без шифрования, камера на телефоне так не заработает — http://localhost:{PORT}/")
    else:
        srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
    print(f"Открой на телефоне: https://{ip}:{PORT}/capture.html")
    print(f"На компьютере: https://localhost:{PORT}/  (статус, комнаты, редактор)")
    print("Сертификат самоподписанный: «Подробнее» → «Перейти на сайт». Сканы падают в", CAPTURES)
    srv.serve_forever()


if __name__ == "__main__":
    main()
