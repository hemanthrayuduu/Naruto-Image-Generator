# Use Python 3.10 slim as base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model files
COPY --chown=app:app . .

# Set default environment variables for production
ENV DEVICE=cpu \
    TORCH_DTYPE=float32 \
    MODEL_DIR=/app/model \
    DEFAULT_STEPS=10 \
    DEFAULT_GUIDANCE_SCALE=7.5 \
    DEFAULT_WIDTH=256 \
    DEFAULT_HEIGHT=256 \
    ALLOWED_ORIGINS=*

# Create model directory if it doesn't exist
RUN mkdir -p /app/model && chown -R app:app /app

# Switch to non-root user
USER app

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI app with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"] 