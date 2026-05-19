"""ResNet18-based image classification model."""

from __future__ import annotations

import io

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18


class Model:
    """Wraps a pretrained ResNet18 for top-k image classification."""

    def __init__(self) -> None:
        weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=weights)
        self.model.eval()

        # Preprocessing transforms that match what ResNet18 was trained with.
        self.transform = weights.transforms()

        # Human-readable class names ("tabby cat", "beagle", ...).
        self.categories: list[str] = list(weights.meta["categories"])

    def predict(self, image_bytes: bytes, top_k: int = 5) -> list[dict[str, float | str]]:
        """Return the top-k predictions for a single image.

        Args:
            image_bytes: raw bytes of an image file (JPEG, PNG, etc.).
            top_k: how many top predictions to return.

        Returns:
            A list of dicts shaped like [{"label": "tabby cat", "probability": 0.87}, ...].
        """
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.transform(image).unsqueeze(0)  # add batch dim → (1, 3, 224, 224)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)[0]
            top_probs, top_idxs = probs.topk(top_k)

        return [
            {"label": self.categories[idx], "probability": float(prob)}
            for prob, idx in zip(top_probs.tolist(), top_idxs.tolist(), strict=True)
        ]
