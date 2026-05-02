# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile  —  Containerised Churn Prediction API
# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2 KEY TOOL: Docker containerises the inference service.
#
# Build:  docker build -t telco-churn-api .
# Run:    docker run -p 8000:8000 telco-churn-api
# Test:   curl http://localhost:8000/health
# Docs:   http://localhost:8000/docs
#
# The model version is baked into the image at build time via the MLflow
# artefact in models/. To deploy a new model version, rebuild the image.
# ─────────────────────────────────────────────────────────────────────────────
# py -3.12 -m pip install -r requirements.txt
FROM python:3.12-slim

WORKDIR /app

# Needed by the health check command.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and artefacts
COPY src/ src/
COPY serve.py .
COPY models/ models/
COPY mlruns/ mlruns/
COPY data/ data/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
