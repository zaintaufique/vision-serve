"""ONNX Runtime-based image classification (torch-free).

Mirrors the interface of `Model` (PyTorch) so they're swappable behind a
common backend protocol. Imports only onnxruntime, numpy, and pillow —
never torch — so it can ship in a slim production image.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from vision_serve.preprocessing import preprocess

# Project root is two levels up from this file (src/vision_serve/onnx_model.py).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "resnet18.onnx"
CLASSES_PATH = Path(__file__).resolve().parent / "imagenet_classes.txt"


def _load_categories() -> list[str]:
    """Read the 1000 ImageNet class names, one per line."""
    text = CLASSES_PATH.read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


class OnnxModel:
    """Wraps an exported ResNet18 ONNX file for top-k image classification."""

    def __init__(self, model_path: Path | str = DEFAULT_MODEL_PATH) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found at {model_path}. "
                "Run `uv run python -m vision_serve.export_onnx` first."
            )

        self.categories = _load_categories()
        self.session = ort.InferenceSession(str(model_path))
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, image_bytes: bytes, top_k: int = 5) -> list[dict[str, float | str]]:
        """Return the top-k predictions for a single image.

        Same signature and output shape as `Model.predict`.
        """
        input_array = preprocess(image_bytes)

        logits = self.session.run([self.output_name], {self.input_name: input_array})[0]

        # Numerically stable softmax in NumPy.
        logits_row = logits[0]
        exps = np.exp(logits_row - logits_row.max())
        probs = exps / exps.sum()

        top_idxs = np.argsort(-probs)[:top_k]
        return [
            {"label": self.categories[int(idx)], "probability": float(probs[idx])}
            for idx in top_idxs
        ]
