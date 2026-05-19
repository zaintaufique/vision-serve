from fastapi.testclient import TestClient

from vision_serve.api import app

client = TestClient(app)


def test_healthz_returns_ok() -> None:
    """GET /healthz should return 200 OK with {'status': 'ok'}."""
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
