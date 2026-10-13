# Room Twin — инструкции для агента (Codex и др.)

Тот же контракт, что в `SKILL.md` (там подробно): телефон снимает → компьютер строит облако →
ты собираешь `room.json` → человек правит в редакторе.

1. Модель глубины `models/da2-metric-indoor-small.onnx`; нет — `tools/export_depth.py` (см. README).
2. Сервер в фоне: `uv run --with numpy --with onnxruntime --with pillow python scripts/serve_capture.py`.
   Открой пользователю `https://localhost:8443/`, адрес для телефона — во вкладке «Телефон» (QR).
3. Следи: `python scripts/watch.py` (в фоне). Пересказывай события; `ГОТОВО out/<id>` — облако собрано.
4. Собери комнату сам: `python scripts/sheet.py out/<id>`, `python scripts/heightmap.py out/<id>`,
   посмотри `plan_room.png`, `height.png`, `sheet.jpg`, `poses.json` и напиши `out/<id>/room.json`
   (формат — README, «Формат room.json»). Потом `python scripts/render_html.py out/<id>/room.json`.
5. Отдай: размеры, `fit_error_m` (> 0.3 м — предложи переснять), список предметов; дальше человек
   правит в «Комнаты» → «Редактор». Перед своей правкой `room.json` перечитай его — человек мог сохранить.

Не трогай `captures/` без просьбы: там исходные сканы пользователя (с его фото).
