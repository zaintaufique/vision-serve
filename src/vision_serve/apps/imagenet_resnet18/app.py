"""ImageNet ResNet-18. The original vision-serve application."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vision_serve.core.image import softmax_top_k
from vision_serve.core.interface import InputSpec

NAME = "imagenet_resnet18"

PROJECT_ROOT = Path(__file__).resolve().parents[4]
LABELS_PATH = Path(__file__).resolve().parent / "labels.txt"
MODEL_PATH = PROJECT_ROOT / "artifacts" / f"{NAME}.onnx"


def _load_labels() -> list[str]:
    text = LABELS_PATH.read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


class ImagenetResnet18:
    name = NAME
    labels = _load_labels()
    model_path = MODEL_PATH
    input_spec = InputSpec(
        resize_size=256,
        crop_size=224,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        mode="RGB",
    )

    def postprocess(self, logits: np.ndarray, top_k: int) -> list[dict[str, float | str]]:
        return softmax_top_k(logits, self.labels, top_k)


APP = ImagenetResnet18()


def load_torch_model():
    """Used only by the PyTorch backend. Never imported in the ONNX image."""
    from torchvision.models import ResNet18_Weights, resnet18

    return resnet18(weights=ResNet18_Weights.DEFAULT)
