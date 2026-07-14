"""The contract every application must satisfy.

Core depends on this file. Core never depends on any app.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class InputSpec:
    """Everything core needs to turn image bytes into a model-ready array.

    The VALUES are app-specific. The CODE that applies them lives in core.
    """

    crop_size: int
    mean: tuple[float, ...]
    std: tuple[float, ...]
    mode: str = "RGB"  # "RGB" (3 channels) or "L" (grayscale, 1 channel)
    resize_size: int | None = None  # shorter-side resize before crop; None = resize direct


@runtime_checkable
class App(Protocol):
    """What an application must expose. Core talks to apps only through this."""

    name: str
    labels: list[str]
    input_spec: InputSpec
    model_path: Path

    def postprocess(self, logits: np.ndarray, top_k: int) -> list[dict[str, float | str]]: ...
