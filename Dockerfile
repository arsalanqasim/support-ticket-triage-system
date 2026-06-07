# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specs first for better layer caching
COPY pyproject.toml ./
COPY src/ ./src/

# Install the package and its runtime dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e "."


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="Arsalan Qasim"
LABEL org.opencontainers.image.title="TriageIQ API"
LABEL org.opencontainers.image.description="AI-powered support ticket classification REST API"
LABEL org.opencontainers.image.version="1.0.0"

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY src/ ./src/
COPY scripts/ ./scripts/

# Models directory — mount a volume here in production
# e.g., docker run -v $(pwd)/models:/app/models ...
RUN mkdir -p /app/models

# Non-root user for security
RUN useradd --create-home --shell /bin/bash triageiq
RUN chown -R triageiq:triageiq /app
USER triageiq

# Expose the API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Default command — start the API server
CMD ["python", "-m", "uvicorn", "triage.api.app:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
