# syntax=docker/dockerfile:1

ARG BACKEND=onnx
ARG APP=imagenet_resnet18
ARG BASE=python:3.12-slim-bookworm

# ---- Stage 1: exporter ----
FROM ${BASE} AS exporter
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
ARG APP

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:${PATH}" PYTHONPATH=/app/src

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --extra export --frozen --no-install-project

COPY src ./src
RUN python -m vision_serve.apps.${APP}.export

# ---- Stage 2: builder ----
FROM ${BASE} AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
ARG BACKEND
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --extra "${BACKEND}" --frozen --no-install-project

# ---- Stage 3: runtime — the only stage that ships ----
FROM ${BASE} AS runtime
ARG BACKEND
ARG APP
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 10001 appuser

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# ONLY core and ONE app are shipped.
COPY --chown=appuser:appuser src/vision_serve/__init__.py    /app/src/vision_serve/__init__.py
COPY --chown=appuser:appuser src/vision_serve/core           /app/src/vision_serve/core
COPY --chown=appuser:appuser src/vision_serve/apps/__init__.py /app/src/vision_serve/apps/__init__.py
COPY --chown=appuser:appuser src/vision_serve/apps/${APP}    /app/src/vision_serve/apps/${APP}

COPY --from=exporter --chown=appuser:appuser /app/artifacts /app/artifacts

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    VISION_SERVE_BACKEND=${BACKEND} \
    VISION_SERVE_APP=${APP}

USER appuser
EXPOSE 8000
CMD ["uvicorn", "vision_serve.core.server:app", "--host", "0.0.0.0", "--port", "8000"]