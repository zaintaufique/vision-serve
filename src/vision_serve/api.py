from fastapi import FastAPI

app = FastAPI(
    title="vision-serve",
    description="Production-grade inference service for deep neural networks.",
    version="0.1.0",
)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Liveness probe. Returns 200 OK when the service is running."""
    return {"status": "ok"}
