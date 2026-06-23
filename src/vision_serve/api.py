"""FastAPI app: routes, lifespan management, and backend selection."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Protocol

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

# from vision_serve.model import Model
# from vision_serve.onnx_model import OnnxModel


class InferenceBackend(Protocol):
    """Common interface implemented by both `Model` and `OnnxModel`."""

    def predict(self, image_bytes: bytes, top_k: int = 5) -> list[dict[str, float | str]]: ...


# def _select_backend() -> InferenceBackend:
#     """Pick the inference backend based on the VISION_SERVE_BACKEND env var."""
#     name = os.getenv("VISION_SERVE_BACKEND", "pytorch").lower()
#     if name == "pytorch":
#         return Model()
#     if name == "onnx":
#         return OnnxModel()
#     raise ValueError(f"Unknown VISION_SERVE_BACKEND={name!r}. Use 'pytorch' or 'onnx'.")


def _select_backend() -> InferenceBackend:
    """Pick the inference backend based on the VISION_SERVE_BACKEND env var.

    Imports are done lazily inside each branch so a deployment that only
    installs one backend's dependencies (e.g. the ONNX-only production
    image, which has no torch) never imports the other backend's modules.
    """
    name = os.getenv("VISION_SERVE_BACKEND", "pytorch").lower()
    if name == "pytorch":
        from vision_serve.model import Model

        return Model()
    if name == "onnx":
        from vision_serve.onnx_model import OnnxModel

        return OnnxModel()
    raise ValueError(f"Unknown VISION_SERVE_BACKEND={name!r}. Use 'pytorch' or 'onnx'.")


# Populated at startup, cleared at shutdown.
_state: dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the inference backend once when the app starts."""
    backend_name = os.getenv("VISION_SERVE_BACKEND", "pytorch").lower()
    _state["backend"] = _select_backend()
    _state["backend_name"] = backend_name
    yield
    _state.clear()


app = FastAPI(
    title="vision-serve",
    description="Production-grade inference service for deep neural networks.",
    version="0.1.0",
    lifespan=lifespan,
)


class Prediction(BaseModel):
    """One predicted class with its probability."""

    label: str = Field(..., description="Human-readable class name.")
    probability: float = Field(..., ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    """Top-K predictions returned by /predict."""

    predictions: list[Prediction]
    backend: str = Field(..., description="Which inference backend produced this result.")


class HealthResponse(BaseModel):
    status: str
    backend: str


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    """Liveness probe. Reports which backend is active."""
    return HealthResponse(status="ok", backend=str(_state.get("backend_name", "unknown")))


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: Annotated[UploadFile, File(...)],
    top_k: int = 5,
) -> PredictResponse:
    """Classify an uploaded image. Returns the top-K predictions."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    image_bytes = await file.read()
    backend: InferenceBackend = _state["backend"]  # type: ignore[assignment]

    try:
        results = backend.predict(image_bytes, top_k=top_k)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not process image: {exc}") from exc

    return PredictResponse(
        predictions=[Prediction(**r) for r in results],
        backend=str(_state["backend_name"]),
    )
