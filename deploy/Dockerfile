# One image for every role: the SuperLink (server) and each hospital's SuperNode.
# CPU-only PyTorch, dependencies pinned by uv.lock, non-root user, no data inside.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    PYTHONWARNINGS=ignore \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    MEDMNIST_ROOT=/data/medmnist \
    FLWR_HOME=/home/hospital/.flwr \
    PATH=/app/.venv/bin:$PATH

# Same UID as the default WSL / Linux user so mounted certificates and results are readable.
RUN useradd --create-home --uid 1000 hospital
WORKDIR /app

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
# The uv download cache lives in a build cache mount, not in the image (~1.4 GB saved).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev && chown -R hospital /app

USER hospital
