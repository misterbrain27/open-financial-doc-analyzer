# syntax=docker/dockerfile:1

# ---------- builder : installe les dépendances dans un venv isolé ----------
FROM python:3.12-slim AS builder
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# INSTALL_DEV=true installe aussi les outils de dev (ruff, pytest, debugpy...).
ARG INSTALL_DEV=false
COPY pyproject.toml ./
COPY app ./app
RUN pip install --upgrade pip && \
    if [ "$INSTALL_DEV" = "true" ]; then pip install ".[dev]"; else pip install .; fi

# ---------- runtime : image finale légère ----------
FROM python:3.12-slim AS runtime
WORKDIR /app
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY --from=builder /opt/venv /opt/venv
COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
