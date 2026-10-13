"""HTTPS-сервер для съёмки с телефона в той же Wi-Fi сети.

Отдаёт web/capture.html и принимает готовый скан (POST /upload) в папку captures/.
Камера и гироскоп в браузере работают только по https, поэтому сертификат
самоподписанный: на телефоне один раз «Всё равно перейти».

  python scripts/serve_capture.py            # https://<ip-мака>:8443/capture.html
"""
from __future__ import annotations

import datetime as dt
import http.server
import json
import socket
import ssl
import subprocess
import sys
from pathlib import Path

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


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(WEB), **kw)

    def do_POST(self):
        if self.path != "/upload":
            return self.send_error(404)
        size = int(self.headers.get("Content-Length", 0))
        if not 0 < size <= MAX_BYTES:
            return self.send_error(413)
        body = self.rfile.read(size)
        try:
            cap = json.loads(body)
            assert cap.get("format", "").startswith("room-twin-capture/")
        except Exception:
            return self.send_error(400, "не скан room-twin")
        CAPTURES.mkdir(exist_ok=True)
        name = f"room-capture-{dt.datetime.now():%Y-%m-%d-%H%M%S}.json"
        (CAPTURES / name).write_bytes(body)
        print(f"принят скан: {CAPTURES / name} — {len(cap.get('frames', []))} кадров, «{cap.get('room')}»")
        reply = json.dumps({"ok": True, "file": name}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)


def main():
    ip = lan_ip()
    cert, key = ensure_cert(ip)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert, key)
    srv = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
    print(f"Открой на телефоне: https://{ip}:{PORT}/capture.html")
    print("Сертификат самоподписанный: «Подробнее» → «Перейти на сайт». Сканы падают в", CAPTURES)
    srv.serve_forever()


if __name__ == "__main__":
    main()
