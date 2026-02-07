FROM python:3.11-slim AS base

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY pyproject.toml setup.py requirements.txt ./
COPY igqk/__init__.py igqk/__init__.py
RUN pip install --no-cache-dir -e ".[all]" 2>/dev/null || \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -e ".[all]"

# Copy source
COPY . .

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

EXPOSE 8000 7860

# Default: run API server
CMD ["python", "-m", "igqk.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
