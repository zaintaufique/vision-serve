"""Core server tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from vision_serve.core.registry import get_app
from vision_serve.core.server import create_app

APP = get_app("imagenet_resnet18")


def test_registry_resolves_the_app_without_loading_a_model() -> None:
    """The contract is readable without touching the model artifact."""
    assert APP.name == "imagenet_resnet18"
    assert len(APP.labels) == 1000
    assert APP.input_spec.crop_size == 224
    assert APP.input_spec.mean == (0.485, 0.456, 0.406)


@pytest.mark.skipif(
    not APP.model_path.exists(),
    reason=f"No ONNX artifact at {APP.model_path}. Run the export first.",
)
def test_healthz_reports_app_and_backend() -> None:
    """/healthz over the ONNX backend. Requires the exported model."""
    api = create_app(APP, "onnx")
    with TestClient(api) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "app": "imagenet_resnet18",
        "backend": "onnx",
    }
