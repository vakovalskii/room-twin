# Room Twin — инструкции для агента (Codex и др.)

Тот же контракт, что в `SKILL.md`: скан комнаты с телефона → `room.json` + 3D-вьюер `room.html`.

1. Проверь модель глубины `models/da2-metric-indoor-small.onnx`; нет — `tools/export_depth.py` (см. SKILL.md).
2. Проверь ключ `NEURALDEEP_API_KEY` (sk-…, hub.neuraldeep.ru).
3. Съёмка: `python scripts/serve_capture.py`, ссылку открыть на телефоне в той же Wi-Fi.
   Скан падает в `captures/`. Либо пользователь присылает `room-capture-*.json` сам.
4. Сборка: `python scripts/build_room.py <скан> --out out/<комната>` (зависимости: numpy, onnxruntime, pillow).
5. Отдай `room.html`, размеры и `fit_error_m`; при разбросе стен > 0.3 м предложи переснять.
6. Правки — в `room.json`, затем `python scripts/render_html.py out/<комната>/room.json`.

Не трогай `captures/` без просьбы: там исходные сканы пользователя.
