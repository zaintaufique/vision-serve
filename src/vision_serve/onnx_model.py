"""ONNX Runtime-based image classification.

Mirrors the interface of `Model` (PyTorch) so they're swappable behind a
common backend protocol.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision.models import ResNet18_Weights

# Default path the export script writes to. Override via constructor if needed.
# Project root is two levels up from this file (src/vision_serve/onnx_model.py).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "resnet18.onnx"


class OnnxModel:
    """Wraps an exported ResNet18 ONNX file for top-k image classification."""

    def __init__(self, model_path: Path | str = DEFAULT_MODEL_PATH) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found at {model_path}. "
                "Run `uv run python -m vision_serve.export_onnx` first."
            )

        # Reuse torchvision's official preprocessing — same pipeline as PyTorch
        # Model, so input tensors are identical across backends.
        weights = ResNet18_Weights.DEFAULT
        self.transform = weights.transforms()
        self.categories: list[str] = list(weights.meta["categories"])

        self.session = ort.InferenceSession(str(model_path))
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, image_bytes: bytes, top_k: int = 5) -> list[dict[str, float | str]]:
        """Return the top-k predictions for a single image.

        Same signature and output shape as `Model.predict`.
        """
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.transform(image).unsqueeze(0)
        input_np = tensor.numpy()

        logits = self.session.run([self.output_name], {self.input_name: input_np})[0]

        # Numerically stable softmax in NumPy.
        logits_row = logits[0]
        exps = np.exp(logits_row - logits_row.max())
        probs = exps / exps.sum()

        top_idxs = np.argsort(-probs)[:top_k]
        return [
            {"label": self.categories[int(idx)], "probability": float(probs[idx])}
            for idx in top_idxs
        ]
