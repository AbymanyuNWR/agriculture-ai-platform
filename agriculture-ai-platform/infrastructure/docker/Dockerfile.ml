FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Production stage
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages
COPY --from=builder /usr/local/lib/python3.9/dist-packages /usr/local/lib/python3.9/dist-packages
COPY --from=builder /usr/bin/python3 /usr/bin/python3

# Copy application code
COPY src/ src/
COPY data/ data/

# Create directories
RUN mkdir -p uploads temp logs models

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run ML server
CMD ["python3", "-m", "uvicorn", "src.ml.serving.model_server:app", "--host", "0.0.0.0", "--port", "8001"]
