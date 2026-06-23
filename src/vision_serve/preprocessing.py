"""Torch-free image preprocessing for ImageNet classification models.

Reproduces torchvision's `ResNet18_Weights.DEFAULT.transforms()` pipeline
using only Pillow and NumPy, so backends that don't need PyTorch (e.g. the
ONNX runtime path) can preprocess without importing it.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

# ImageNet preprocessing constants (match torchvision's ImageClassification).
RESIZE_SIZE = 256  # shorter edge resized to this, aspect ratio kept
CROP_SIZE = 224  # final centre crop, square
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _resize_shorter_side(image: Image.Image, target: int) -> Image.Image:
    """Resize so the shorter edge equals `target`, preserving aspect ratio."""
    width, height = image.size
    if width <= height:
        new_width = target
        new_height = round(target * height / width)
    else:
        new_height = target
        new_width = round(target * width / height)
    return image.resize((new_width, new_height), Image.Resampling.BILINEAR)


def _center_crop(image: Image.Image, size: int) -> Image.Image:
    """Crop a `size` x `size` square from the centre of the image."""
    width, height = image.size
    left = (width - size) // 2
    top = (height - size) // 2
    return image.crop((left, top, left + size, top + size))


def preprocess(image_bytes: bytes) -> np.ndarray:
    """Turn raw image bytes into a model-ready (1, 3, 224, 224) float32 array.

    Steps: decode -> RGB -> resize shorter side to 256 -> centre-crop 224 ->
    scale to [0, 1] -> normalise per channel -> HWC->CHW -> add batch dim.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = _resize_shorter_side(image, RESIZE_SIZE)
    image = _center_crop(image, CROP_SIZE)

    # (H, W, C) array, values 0..255 -> 0.0..1.0
    array = np.asarray(image, dtype=np.float32) / 255.0

    # Normalise each channel: (x - mean) / std
    array = (array - MEAN) / STD

    # HWC -> CHW, then add batch dim -> (1, 3, 224, 224)
    array = np.transpose(array, (2, 0, 1))
    array = np.expand_dims(array, axis=0)

    return np.ascontiguousarray(array, dtype=np.float32)
