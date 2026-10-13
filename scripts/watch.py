"""Агент следит за съёмкой: печатает события и выходит, когда облако собрано.

  python scripts/watch.py                 # ждёт телефон, съёмку и сборку на https://localhost:8443
  python scripts/watch.py --port 8444 --http

Строки вывода — события по одному на строку (телефон подключился, кадры, брак, «Готово», облако).
Последняя строка при успехе: `ГОТОВО out/<id>` — дальше агент собирает room.json по SKILL.md.
Код выхода: 0 — облако готово, 1 — сборка упала, 2 — вышло время.
"""
from __future__ import annotations

import argparse
import json
import ssl
import sys
import time
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8443)
    ap.add_argument("--http", action="store_true", help="сервер запущен с ROOM_TWIN_HTTP=1")
    ap.add_argument("--timeout", type=float, default=3600, help="секунд ждать всего")
    args = ap.parse_args()
    base = f"{'http' if args.http else 'https'}://localhost:{args.port}/api/sessions"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}),
                                         urllib.request.HTTPSHandler(context=ssl._create_unverified_context()))
    say = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)  # noqa: E731

    seen_phone, seen_scan, seen_bad, frames_said, finished = set(), set(), set(), {}, set()
    start, down, old = time.time(), 0, None
    say(f"слежу за {base}")
    while time.time() - start < args.timeout:
        try:
            data = json.loads(opener.open(base, timeout=5).read())
            down = 0
            if old is None:  # съёмки, начатые до запуска, — не наши
                old = {s["id"] for s in data.get("live", [])}
        except Exception as e:  # сервер перезапускают — не повод падать
            down += 1
            if down in (1, 30):
                say(f"сервер не отвечает: {e}")
            time.sleep(2)
            continue
        for p in data.get("phones", []):
            if p["ip"] not in seen_phone:
                seen_phone.add(p["ip"])
                say(f"телефон подключился: {p['device']} ({p['ip']}), {p['stage']}")
        for s in data.get("live", []):
            sid = s["id"]
            if sid in old:
                continue
            if sid not in seen_scan:
                seen_scan.add(sid)
                say(f"съёмка началась: «{s['room']}» → out/{sid}")
            ok = sum(f["state"] == "ok" for f in s["frames"])
            if ok // 5 > frames_said.get(sid, 0) // 5:
                rows = {r: sum(f["state"] == "ok" and f.get("row") == r for f in s["frames"]) for r in ("mid", "low", "high")}
                size = f", ≈ {s['size'][0]} × {s['size'][1]} м" if s.get("size") else ""
                say(f"принято {ok} кадров (прямо {rows['mid']}, вниз {rows['low']}, вверх {rows['high']}){size}")
            frames_said[sid] = ok
            for f in s["frames"]:
                key = (sid, f["i"])
                if f["state"] == "bad" and key not in seen_bad:
                    seen_bad.add(key)
                    say(f"брак кадра {f['i']}: {f['reason']}")
            if s.get("build") and sid not in finished:
                finished.add(sid)
                say(f"«Готово» нажато: {len(s['frames'])} кадров, собираю облако")
            if s.get("build") == "done":
                say(f"ГОТОВО out/{sid}")
                return 0
            if s.get("build") == "error":
                say(f"СБОРКА УПАЛА: смотри out/{sid}/build.log")
                return 1
        time.sleep(1.5)
    say("вышло время ожидания")
    return 2


if __name__ == "__main__":
    sys.exit(main())
