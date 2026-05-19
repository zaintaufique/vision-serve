"""FastAPI app: routes and lifespan management."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from vision_serve.model import Model

# Holds the loaded model between requests. Populated at startup.
_state: dict[str, Model] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model once when the app starts; clean up when it stops."""
    _state["model"] = Model()
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
    probability: float = Field(..., ge=0.0, le=1.0, description="Predicted probability (0..1).")


class PredictResponse(BaseModel):
    """The top-K predictions returned by /predict."""

    predictions: list[Prediction]


class HealthResponse(BaseModel):
    status: str


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    """Liveness probe. Returns 200 OK when the service is running."""
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...), top_k: int = 5) -> PredictResponse:
    """Classify an uploaded image. Returns the top-K predictions."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    image_bytes = await file.read()
    try:
        results = _state["model"].predict(image_bytes, top_k=top_k)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not process image: {exc}") from exc

    return PredictResponse(predictions=[Prediction(**r) for r in results])
