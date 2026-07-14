"""Generic, app-agnostic image preprocessing.

Nothing here knows what it is preprocessing FOR. All values arrive as
arguments; none are hardcoded.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from vision_serve.core.interface import InputSpec


def _resize_shorter_side(image: Image.Image, target: int) -> Image.Image:
    width, height = image.size
    if width <= height:
        new_width, new_height = target, round(target * height / width)
    else:
        new_height, new_width = target, round(target * width / height)
    return image.resize((new_width, new_height), Image.Resampling.BILINEAR)


def _center_crop(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    left = (width - size) // 2
    top = (height - size) // 2
    return image.crop((left, top, left + size, top + size))


def preprocess(image_bytes: bytes, spec: InputSpec) -> np.ndarray:
    """Image bytes -> (1, C, H, W) float32 array, per the given spec."""
    image = Image.open(io.BytesIO(image_bytes)).convert(spec.mode)

    if spec.resize_size is not None:
        image = _resize_shorter_side(image, spec.resize_size)
        image = _center_crop(image, spec.crop_size)
    else:
        image = image.resize((spec.crop_size, spec.crop_size), Image.Resampling.BILINEAR)

    array = np.asarray(image, dtype=np.float32) / 255.0

    if array.ndim == 2:  # grayscale: (H, W) -> (H, W, 1)
        array = array[:, :, None]

    mean = np.array(spec.mean, dtype=np.float32)
    std = np.array(spec.std, dtype=np.float32)
    array = (array - mean) / std

    array = np.transpose(array, (2, 0, 1))  # HWC -> CHW
    array = np.expand_dims(array, axis=0)  # -> (1, C, H, W)
    return np.ascontiguousarray(array, dtype=np.float32)


def softmax_top_k(
    logits: np.ndarray, labels: list[str], top_k: int
) -> list[dict[str, float | str]]:
    """Shared helper: most classifiers postprocess exactly this way."""
    row = logits[0]
    exps = np.exp(row - row.max())
    probs = exps / exps.sum()
    top_k = min(top_k, len(labels))
    idxs = np.argsort(-probs)[:top_k]
    return [{"label": labels[int(i)], "probability": float(probs[i])} for i in idxs]
