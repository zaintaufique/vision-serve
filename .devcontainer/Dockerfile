FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Better defaults for uv inside a dev container
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/workspaces/.venv

# Make sure git is installed (the slim image doesn't include it)
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

WORKDIR /workspaces