# syntax=docker/dockerfile:1
# Multi-stage build for FraudML.
# Goal: a slim runtime image (< 2 GB) that can run training, the batch
# scoring CLI, and the FastAPI serving layer.

# ---------------------------------------------------------------------------
# builder: install build deps + project requirements into /root/.local
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# Build essentials for compiling numpy/lightgbm/pyarrow wheels when no
# prebuilt wheel is available for the target platform.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt ./
# Install into /root/.local so the runtime stage can copy just the deps.
RUN pip install --user -r requirements.txt

# ---------------------------------------------------------------------------
# runtime: slim image with only the installed deps + source
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:${PATH}" \
    PYTHONPATH="/app"

# curl is needed for the Docker HEALTHCHECK; ca-certificates for HTTPS
# calls to MLflow / object stores; tini for proper signal handling.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed Python packages from the builder.
COPY --from=builder /root/.local /root/.local

# Copy only the source and config the runtime needs. Data / artifacts /
# mlruns are mounted as volumes from docker-compose, not baked in.
COPY src/ ./src/
COPY configs/ ./configs/
COPY pyproject.toml ./

# Default entrypoint: training. Override via `command:` in compose or
# `docker run` (e.g. `uvicorn src.serving.main:app`).
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "src.train"]

# Liveness probe — cheap import check that the package is intact.
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import src" || exit 1
