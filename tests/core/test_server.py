"""Core server tests, run against the imagenet app."""

from __future__ import annotations

from fastapi.testclient import TestClient

from vision_serve.core.registry import get_app
from vision_serve.core.server import create_app


def test_healthz_returns_ok_with_app_and_backend() -> None:
    api = create_app(get_app("imagenet_resnet18"), "onnx")
    with TestClient(api) as client:
        response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["app"] == "imagenet_resnet18"
    assert body["backend"] == "onnx"
