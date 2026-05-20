"""Tests for the FastAPI app."""

from __future__ import annotations

from fastapi.testclient import TestClient

from vision_serve.api import app


def test_healthz_returns_ok_with_backend() -> None:
    """GET /healthz returns 200 OK with status and active backend name."""
    with TestClient(app) as client:
        response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["backend"] in {"pytorch", "onnx"}
