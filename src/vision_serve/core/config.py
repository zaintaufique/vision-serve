"""Environment-driven settings."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    app_name: str
    backend: str

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            app_name=os.getenv("VISION_SERVE_APP", "imagenet_resnet18"),
            backend=os.getenv("VISION_SERVE_BACKEND", "pytorch").lower(),
        )
