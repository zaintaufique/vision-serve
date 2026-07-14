"""Inference backends. Both consume an App; neither knows which one."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from vision_serve.core.image import preprocess
from vision_serve.core.interface import App


class Backend(Protocol):
    def predict(self, image_bytes: bytes, top_k: int) -> list[dict[str, float | str]]: ...


class OnnxBackend:
    """Torch-free inference via ONNX Runtime."""

    def __init__(self, app: App) -> None:
        import onnxruntime as ort

        if not app.model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found at {app.model_path}. "
                f"Run: python -m vision_serve.apps.{app.name}.export"
            )
        self.app = app
        self.session = ort.InferenceSession(str(app.model_path))
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, image_bytes: bytes, top_k: int = 5) -> list[dict[str, float | str]]:
        array = preprocess(image_bytes, self.app.input_spec)
        logits = self.session.run([self.output_name], {self.input_name: array})[0]
        return self.app.postprocess(np.asarray(logits), top_k)


class TorchBackend:
    """PyTorch inference. The app supplies the module; core drives it."""

    def __init__(self, app: App) -> None:
        import torch

        from vision_serve.core.registry import load_module

        module = load_module(app.name)
        self.torch = torch
        self.app = app
        self.model = module.load_torch_model()
        self.model.eval()

    def predict(self, image_bytes: bytes, top_k: int = 5) -> list[dict[str, float | str]]:
        array = preprocess(image_bytes, self.app.input_spec)
        with self.torch.no_grad():
            logits = self.model(self.torch.from_numpy(array))
        return self.app.postprocess(logits.numpy(), top_k)


def select_backend(app: App, name: str) -> Backend:
    """Choose a backend by name. Imports are lazy so a torch-free image works."""
    name = name.lower()
    if name == "onnx":
        return OnnxBackend(app)
    if name == "pytorch":
        return TorchBackend(app)
    raise ValueError(f"Unknown backend {name!r}. Use 'pytorch' or 'onnx'.")
