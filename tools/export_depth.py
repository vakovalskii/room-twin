"""Официальная Depth Anything V2 Metric Indoor Small (HF transformers) → ONNX с динамическими H/W.

Выход: глубина в метрах (z-depth) на каждый пиксель. Вход RGB с нормировкой ImageNet,
стороны кратны 14. Запуск:
  uv run --python 3.12 --with torch --with transformers --with onnx --with onnxruntime \
    python tools/export_depth.py models/da2-metric-indoor-small.onnx
"""
import sys

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForDepthEstimation

REPO = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
OUT = sys.argv[1]

model = AutoModelForDepthEstimation.from_pretrained(REPO).eval()


class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pixel_values):
        return self.m(pixel_values=pixel_values).predicted_depth


x = torch.randn(1, 3, 518, 392)
with torch.no_grad():
    ref = Wrap(model)(x).numpy()
torch.onnx.export(Wrap(model), (x,), OUT, input_names=["pixel_values"], output_names=["depth"],
                  dynamic_axes={"pixel_values": {2: "h", 3: "w"}, "depth": {1: "h", 2: "w"}},
                  opset_version=17, dynamo=False)
got = ort.InferenceSession(OUT).run(None, {"pixel_values": x.numpy()})[0]
print("shape", got.shape, "max abs diff vs torch", float(np.abs(got - ref).max()))
