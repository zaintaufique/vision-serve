"""FastAPI factory. Serves whichever App it is handed."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from vision_serve.core.config import Settings
from vision_serve.core.interface import App
from vision_serve.core.registry import get_app
from vision_serve.core.runtime import select_backend


class Prediction(BaseModel):
    label: str = Field(..., description="Human-readable class name.")
    probability: float = Field(..., ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    app: str
    backend: str


class HealthResponse(BaseModel):
    status: str
    app: str
    backend: str


def create_app(application: App, backend_name: str) -> FastAPI:
    """Build a FastAPI instance that serves one application."""
    state: dict[str, object] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        state["backend"] = select_backend(application, backend_name)
        yield
        state.clear()

    api = FastAPI(
        title=f"vision-serve: {application.name}",
        description="Production-grade inference service for deep neural networks.",
        version="0.2.0",
        lifespan=lifespan,
    )

    @api.get("/healthz", response_model=HealthResponse)
    def healthz() -> HealthResponse:
        return HealthResponse(status="ok", app=application.name, backend=backend_name)

    @api.post("/predict", response_model=PredictResponse)
    async def predict(
        file: Annotated[UploadFile, File(...)],
        top_k: int = 5,
    ) -> PredictResponse:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image.")

        image_bytes = await file.read()
        backend = state["backend"]
        try:
            results = backend.predict(image_bytes, top_k=top_k)  # type: ignore[attr-defined]
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not process image: {exc}") from exc

        return PredictResponse(
            predictions=[Prediction(**r) for r in results],
            app=application.name,
            backend=backend_name,
        )

    return api


# Entrypoint for uvicorn: vision_serve.core.server:app
_settings = Settings.from_env()
app = create_app(get_app(_settings.app_name), _settings.backend)
