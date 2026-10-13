"""HTTPS-сервер для съёмки с телефона в той же Wi-Fi сети. Телефон — только камера, всё считает компьютер.

Живой скан: POST /live/start → кадры по одному POST /live/<id>/frame → компьютер сразу считает глубину,
бракует кадры и рисует план; телефон опрашивает GET /live/<id>/state и /live/<id>/plan.png.
POST /live/<id>/finish → captures/room-capture-<id>.json и сборка облака (scripts/cloud.py) в out/<id>/.
Комнату по облаку и кадрам дальше собирает агент (SKILL.md).

Доступ: с этого компьютера (localhost) — свободно; из Wi-Fi — только с ключом сопряжения. Ключ
вшит в адрес и QR для телефона (`capture.html?k=…`), при первом заходе ставится cookie.

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
import re
import secrets
import socket
import ssl
import subprocess
import sys
import threading
from http.cookies import SimpleCookie
from urllib.parse import parse_qs, urlsplit
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from live import LiveScan, ScanClosed, session  # noqa: E402
from render_html import render, write_atomic  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
CAPTURES = ROOT / "captures"
CERT_DIR = Path.home() / ".cache" / "room-twin"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8443
MAX_BYTES = 200 * 1024 * 1024
LIMITS = {"frame": 8 << 20, "room": 5 << 20, "upload": MAX_BYTES}  # остальное (пинг, старт, финиш) — 64 КБ
MAX_ACTIVE_SCANS = 2


def pairing_key() -> str:
    """Ключ сопряжения телефона: постоянный, чтобы закладка на телефоне переживала перезапуск."""
    CERT_DIR.mkdir(parents=True, exist_ok=True)
    f = CERT_DIR / "pairing.key"
    if not f.exists():
        f.write_text(secrets.token_urlsafe(12))
        f.chmod(0o600)
    return f.read_text().strip()


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
REG = threading.Lock()  # для BUILDS / LIVE / PHONES: их читают и пишут разные потоки
BUILD_SLOT = threading.Semaphore(1)  # сборки облака по одной: каждая ест все ядра
ROOM_LOCKS: dict[str, threading.Lock] = {}
SAFE_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")  # имена сканов из URL: без точек и слешей
KEY = pairing_key()


def new_id() -> str:
    """Время + случайный хвост: два скана в одну секунду не перепишут друг друга."""
    return f"{dt.datetime.now():%Y-%m-%d-%H%M%S}-{secrets.token_hex(2)}"


def tail_line(path: Path) -> str:
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        f.seek(max(0, path.stat().st_size - 2048))
        lines = f.read().decode(errors="ignore").strip().splitlines()
    return lines[-1] if lines else ""


def build(capture: Path):
    """Облако точек и план на этом компьютере сразу после съёмки. Комнату по ним собирает агент."""
    stem = capture.stem.replace("room-capture-", "")
    dst = OUT / stem
    dst.mkdir(parents=True, exist_ok=True)
    with REG:
        BUILDS[stem] = "building"
    with BUILD_SLOT, open(dst / "build.log", "w") as log:
        code = subprocess.run(
            [sys.executable, "-u", str(ROOT / "scripts" / "cloud.py"), str(capture), "--out", str(dst)],
            stdout=log, stderr=subprocess.STDOUT, cwd=ROOT).returncode
    with REG:
        BUILDS[stem] = "done" if code == 0 and (dst / "cloud.html").exists() else "error"
    print(f"облако {stem}: {BUILDS[stem]} — https://localhost:{PORT}/out/{stem}/cloud.html", flush=True)


def sessions() -> dict:
    """Для страницы статуса на компьютере: живые сканы и последние собранные в out/."""
    with REG:
        for sid in [sid for sid in LIVE if not (OUT / sid).exists()]:  # папку удалили — сессия больше не живая
            LIVE.pop(sid)
        live_items, builds, phones_raw = list(LIVE.items()), dict(BUILDS), dict(PHONES)
    live = [{**scan.state(), "build": builds.get(sid)} for sid, scan in live_items]
    now = dt.datetime.now().timestamp()
    phones = [{**p, "ip": ip, "age": round(now - p["seen"], 1)} for ip, p in phones_raw.items() if now - p["seen"] < 120]
    live_ids = {sid for sid, _ in live_items}
    dirs = [d for d in sorted(OUT.iterdir(), reverse=True) if d.is_dir() and SAFE_ID.fullmatch(d.name)] if OUT.exists() else []
    # живой скан попадает в «Комнаты», как только его облако собрано (или сборка упала)
    dirs = [d for d in dirs if d.name not in live_ids or builds.get(d.name) in ("done", "error")][:12]
    done = []
    for d in dirs:
        room = None
        for name, key in (("room.json", "name"), ("meta.json", "room")):
            try:
                room = room or json.loads((d / name).read_text()).get(key)
            except (OSError, ValueError):
                pass
        done.append({"id": d.name, "build": builds.get(d.name) or ("done" if (d / "cloud.html").exists() else None),
                     "files": [f for f in ("cloud.html", "room.html", "plan_room.png", "sheet.jpg") if (d / f).exists()],
                     "room": room, "last": [t] if (t := tail_line(d / "build.log")) else []})
    return {"phones": phones, "live": live[::-1], "done": done}


class Handler(http.server.SimpleHTTPRequestHandler):
    timeout = 60  # зависшее соединение не держит поток вечно

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(WEB), **kw)

    def setup(self):
        # TLS-рукопожатие — уже в потоке этого соединения, а не в главном цикле accept():
        # иначе один недоделанный клиент останавливает приём всех остальных
        self.request.settimeout(self.timeout)
        if isinstance(self.request, ssl.SSLSocket):
            self.request.do_handshake()
        super().setup()

    def authorized(self) -> bool:
        """localhost — всегда; из сети — cookie или ?k= с ключом сопряжения (тогда ставим cookie)."""
        if self.client_address[0] in ("127.0.0.1", "::1"):
            return True
        cookie = SimpleCookie(self.headers.get("Cookie", ""))
        if "rt" in cookie and secrets.compare_digest(cookie["rt"].value, KEY):
            return True
        k = parse_qs(urlsplit(self.path).query).get("k", [""])[0]
        if k and secrets.compare_digest(k, KEY):
            self._set_cookie = True
            return True
        return False

    def end_headers(self):
        if getattr(self, "_set_cookie", False):
            secure = "; Secure" if isinstance(self.request, ssl.SSLSocket) else ""
            self.send_header("Set-Cookie", f"rt={KEY}; Path=/; Max-Age=2592000; HttpOnly; SameSite=Strict{secure}")
            self._set_cookie = False
        super().end_headers()

    def deny(self):
        body = ("<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width'>"
                "<body style='font:16px system-ui;padding:24px'><h3>Нужен ключ</h3>Открой адрес с компьютера: "
                "вкладка «Телефон» на https://localhost:%d/ — QR-код уже с ключом.</body>" % PORT).encode()
        self.reply(403, body, "text/html; charset=utf-8")

    def same_origin(self) -> bool:
        # изменяющие запросы только со своих страниц: чужой сайт в браузере не сможет ничего записать
        origin = self.headers.get("Origin")
        return not origin or urlsplit(origin).netloc == self.headers.get("Host")

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

    def send_error(self, code, message=None, explain=None):
        # текст ошибки — в тело JSON: строка статуса HTTP только latin-1, кириллица там роняет ответ
        self.close_connection = True
        self.reply(code, json.dumps({"error": message or ""}, ensure_ascii=False).encode())

    def reply(self, code: int, body: bytes, ctype: str = "application/json; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def live(self, sid: str) -> LiveScan | None:
        with REG:
            return LIVE.get(sid)

    def do_GET(self):
        if not self.authorized():
            return self.deny()
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
            return self.reply(200, json.dumps({"phone_url": f"https://{lan_ip()}:{PORT}/capture.html?k={KEY}"}).encode())
        if parts == ["api", "sessions"]:
            return self.reply(200, json.dumps(sessions(), ensure_ascii=False).encode())
        if self.path.startswith("/status/"):
            stem = self.path[len("/status/"):].split("?", 1)[0]
            if not SAFE_ID.fullmatch(stem):
                return self.send_error(404)
            with REG:
                state = BUILDS.get(stem, "unknown")
            return self.reply(200, json.dumps({"state": state, "last": tail_line(OUT / stem / "build.log")},
                                              ensure_ascii=False).encode())
        super().do_GET()

    def do_POST(self):
        if not self.authorized():
            return self.deny()
        if not self.same_origin():
            return self.send_error(403, "чужой Origin")
        parts = self.path.split("?", 1)[0].strip("/").split("/")
        kind = ("frame" if parts[:1] == ["live"] and parts[-1:] == ["frame"] else
                "room" if parts[:1] == ["out"] else "upload" if parts == ["upload"] else "small")
        try:
            size = int(self.headers.get("Content-Length", 0))
        except ValueError:
            return self.send_error(400)
        if not 0 <= size <= LIMITS.get(kind, 64 << 10):
            return self.send_error(413)
        body = self.rfile.read(size) if size else b"{}"
        if parts == ["phone", "ping"]:
            try:
                data = json.loads(body)
            except ValueError:
                return self.send_error(400)
            ip, now = self.client_address[0], dt.datetime.now().timestamp()
            with REG:
                first = ip not in PHONES or now - PHONES[ip]["seen"] > 30
                PHONES[ip] = {"seen": now, "stage": str(data.get("stage", ""))[:20], "live": str(data.get("live") or "")[:64],
                              "frames": int(data.get("frames") or 0), "device": str(data.get("device", ""))[:80],
                              "torch": bool(data.get("torch"))}
            if first:
                print(f"телефон подключился: {ip} · {PHONES[ip]['device']}", flush=True)
            return self.reply(200, b"{}")
        if parts[0] == "live":
            return self.post_live(parts, body)
        if len(parts) == 3 and parts[0] == "out" and parts[2] == "room.json":
            return self.save_room(parts[1], body)
        if parts != ["upload"] or not size:
            return self.send_error(404)
        try:
            cap = json.loads(body)
            assert cap.get("format", "").startswith("room-twin-capture/")
        except (ValueError, AssertionError, AttributeError):
            return self.send_error(400, "не скан room-twin")
        CAPTURES.mkdir(exist_ok=True)
        stem = new_id()
        path = CAPTURES / f"room-capture-{stem}.json"
        with open(path, "xb") as f:  # x — не перезаписывать существующий исходник
            f.write(body)
        print(f"принят скан: {path} — {len(cap.get('frames', []))} кадров, «{cap.get('room')}»", flush=True)
        threading.Thread(target=build, args=(path,), daemon=True).start()
        return self.reply(200, json.dumps({"ok": True, "file": path.name, "id": stem}).encode())

    def save_room(self, stem: str, body: bytes):
        """Редактор во вьюере сохраняет room.json и пересобирает room.html рядом — атомарно и по очереди."""
        if not SAFE_ID.fullmatch(stem) or not (OUT / stem).is_dir():
            return self.send_error(404)
        try:
            room = json.loads(body)
            assert room.get("format") == "room-twin/1" and isinstance(room.get("objects"), list)
        except (ValueError, AssertionError, AttributeError):
            return self.send_error(400, "не room.json")
        with REG:
            lock = ROOM_LOCKS.setdefault(stem, threading.Lock())
        with lock:
            write_atomic(OUT / stem / "room.json", json.dumps(room, ensure_ascii=False, indent=2))
            render(room, OUT / stem / "room.html")
        print(f"комната {stem}: сохранена из редактора ({len(room['objects'])} предметов)", flush=True)
        return self.reply(200, b'{"ok": true}')

    def post_live(self, parts: list[str], body: bytes):
        try:
            data = json.loads(body)
            assert isinstance(data, dict)
        except (ValueError, AssertionError):
            return self.send_error(400)
        if parts == ["live", "start"]:
            with REG:
                active = sum(not scan.closed for scan in LIVE.values())
            if active >= MAX_ACTIVE_SCANS:
                return self.send_error(429, "уже идёт съёмка — закончи её")
            sid = new_id()
            meta = {k: data.get(k) for k in ("room", "ceiling_m", "created", "user_agent", "camera")}
            meta["room"] = str(meta.get("room") or "Комната")[:80]
            (OUT / sid).mkdir(parents=True, exist_ok=False)
            scan = LiveScan(sid, OUT / sid, meta, session())
            with REG:
                LIVE[sid] = scan
            print(f"живой скан {sid}: «{meta['room']}»", flush=True)
            return self.reply(200, json.dumps({"id": sid}).encode())
        scan = self.live(parts[1]) if len(parts) == 3 and SAFE_ID.fullmatch(parts[1]) else None
        if not scan:
            return self.send_error(404)
        if parts[2] == "frame":
            if not str(data.get("jpeg", "")).startswith("data:image/jpeg;base64,"):
                return self.send_error(400)
            try:
                i = scan.add({k: data.get(k) for k in ("t", "auto", "yaw", "pitch", "alpha", "beta", "gamma",
                                                        "screen_angle", "width", "height", "jpeg", "row")})
            except ScanClosed as e:
                return self.send_error(409, str(e))
            except ValueError:
                return self.send_error(400, "битый кадр")
            return self.reply(200, json.dumps({"i": i}).encode())
        if parts[2] == "finish":
            CAPTURES.mkdir(exist_ok=True)
            cap, fresh = scan.finish(CAPTURES)
            if fresh:
                print(f"живой скан {scan.sid} закончен: {len(scan.frames)} кадров → {cap}", flush=True)
                threading.Thread(target=build, args=(cap,), daemon=True).start()
            return self.reply(200, json.dumps({"id": scan.sid, "frames": len(scan.frames)}).encode())
        return self.send_error(404)


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        # телефон отказался от сертификата, ушёл из Wi-Fi или завис — это не ошибка сервера
        if isinstance(sys.exc_info()[1], (ssl.SSLError, ConnectionError, TimeoutError)):
            return
        super().handle_error(request, client_address)


def main():
    ip = lan_ip()
    cert, key = ensure_cert(ip)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert, key)
    srv = Server(("0.0.0.0", PORT), Handler)
    if os.getenv("ROOM_TWIN_HTTP") == "1":  # для автотестов в браузере без сертификата; телефону нужен https
        print(f"ВНИМАНИЕ: http без шифрования, камера на телефоне так не заработает — http://localhost:{PORT}/")
    else:
        srv.socket = ctx.wrap_socket(srv.socket, server_side=True, do_handshake_on_connect=False)
    print(f"Открой на телефоне: https://{ip}:{PORT}/capture.html?k={KEY}")
    print(f"На компьютере: https://localhost:{PORT}/  (статус, комнаты, редактор)")
    print("Сертификат самоподписанный: «Подробнее» → «Перейти на сайт». Сканы падают в", CAPTURES)
    srv.serve_forever()


if __name__ == "__main__":
    main()
