"""Preprocessing is pure maths. No model, no network, no artifact."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from vision_serve.core.image import preprocess, softmax_top_k
from vision_serve.core.interface import InputSpec


def _png(width: int, height: int, colour: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buffer, format="PNG")
    return buffer.getvalue()


def test_rgb_output_shape() -> None:
    spec = InputSpec(resize_size=256, crop_size=224, mean=(0.0,) * 3, std=(1.0,) * 3)
    array = preprocess(_png(400, 300, (128, 128, 128)), spec)
    assert array.shape == (1, 3, 224, 224)
    assert array.dtype == np.float32


def test_grayscale_output_shape() -> None:
    """Casting will be grayscale. The generic path must handle one channel."""
    spec = InputSpec(crop_size=64, mean=(0.0,), std=(1.0,), mode="L")
    array = preprocess(_png(300, 300, (255, 255, 255)), spec)
    assert array.shape == (1, 1, 64, 64)


def test_normalization_applies_the_values_it_is_given() -> None:
    """Mid-grey scales to 0.5; normalised by mean 0.5 and std 0.5 it becomes 0."""
    spec = InputSpec(crop_size=8, mean=(0.5,) * 3, std=(0.5,) * 3)
    grey = round(0.5 * 255)
    array = preprocess(_png(8, 8, (grey, grey, grey)), spec)
    assert np.allclose(array, 0.0, atol=2e-2)


def test_softmax_top_k_orders_by_probability() -> None:
    logits = np.array([[1.0, 5.0, 3.0]])
    results = softmax_top_k(logits, ["low", "high", "mid"], top_k=2)
    assert [r["label"] for r in results] == ["high", "mid"]
    assert all(0.0 <= r["probability"] <= 1.0 for r in results)
