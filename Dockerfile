# syntax=docker/dockerfile:1

ARG BACKEND=onnx
ARG BASE=python:3.12-slim-bookworm

# ---- Stage 1: exporter — runs export_onnx.py, produces resnet18.onnx ----
FROM ${BASE} AS exporter

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH=/app/src

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --extra export --frozen --no-install-project

COPY src ./src
RUN python -m vision_serve.export_onnx

# ---- Stage 2: builder — installs the clean, torch-free runtime venv ----
FROM ${BASE} AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ARG BACKEND
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --extra "${BACKEND}" --frozen --no-install-project

# ---- Stage 3: runtime — the only stage that ships ----
FROM ${BASE} AS runtime
ARG BACKEND
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 appuser

COPY --from=builder  --chown=appuser:appuser /app/.venv /app/.venv
COPY                 --chown=appuser:appuser src /app/src
COPY --from=exporter --chown=appuser:appuser /app/models/resnet18.onnx /app/models/resnet18.onnx

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    VISION_SERVE_BACKEND=${BACKEND}

USER appuser
EXPOSE 8000
CMD ["uvicorn", "vision_serve.api:app", "--host", "0.0.0.0", "--port", "8000"]