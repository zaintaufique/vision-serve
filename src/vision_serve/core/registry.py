"""The ONLY file in core that names applications.

Adding an app = adding one line here. Nothing else in core changes.
"""

from __future__ import annotations

import importlib
from types import ModuleType

from vision_serve.core.interface import App

APP_NAMES: tuple[str, ...] = (
    "imagenet_resnet18",
    "casting",
)


def load_module(name: str) -> ModuleType:
    """Import an app's module lazily, so only the shipped app is ever loaded."""
    if name not in APP_NAMES:
        raise ValueError(f"Unknown app {name!r}. Known apps: {', '.join(APP_NAMES)}")
    return importlib.import_module(f"vision_serve.apps.{name}.app")


def get_app(name: str) -> App:
    """Return the App object an application module exposes as `APP`."""
    module = load_module(name)
    return module.APP
