FROM node:22-bookworm-slim AS frontend-build

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build


FROM python:3.13-bookworm AS app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    RESUME_MODEL_CACHE_DIR=/app/data/model_cache \
    RESUME_OFFLINE_MODE=true \
    HF_ENDPOINT=https://hf-mirror.com \
    HF_HOME=/app/data/model_cache/huggingface \
    HF_HUB_CACHE=/app/data/model_cache/huggingface/hub \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    SENTENCE_TRANSFORMERS_HOME=/app/data/model_cache/sentence_transformers \
    TORCH_HOME=/app/data/model_cache/torch \
    EMBEDDING_MODEL_PATH=/app/data/models/bge-small-zh-v1.5 \
    RERANKER_MODEL_PATH=/app/data/models/bge-reranker-base

WORKDIR /app

RUN set -eux; \
    for attempt in 1 2 3; do \
        apt-get -o Acquire::Retries=5 update; \
        if DEBIAN_FRONTEND=noninteractive apt-get -o Acquire::Retries=5 install -y --no-install-recommends \
            antiword \
            build-essential \
            curl; then \
            break; \
        fi; \
        if [ "$attempt" = "3" ]; then exit 1; fi; \
        apt-get -f install -y || true; \
        apt-get clean; \
        rm -rf /var/lib/apt/lists/*; \
        sleep 10; \
    done; \
    rm -rf /var/lib/apt/lists/*

# CPU-only build: install CPU torch wheel first, then strip the torch line
# from the main lockfile so pip does not re-resolve a CUDA-bundled torch.
COPY requirements.txt requirements-cpu.txt ./
RUN set -eux; \
    python -m pip install --upgrade pip; \
    python -m pip install -r requirements-cpu.txt; \
    grep -v '^torch==' requirements.txt > /tmp/requirements-no-torch.txt; \
    python -m pip install -r /tmp/requirements-no-torch.txt; \
    apt-get purge -y --auto-remove build-essential gcc g++ make

ARG RESUME_BUILD_ID=dev
ENV RESUME_BUILD_ID=${RESUME_BUILD_ID}
LABEL org.opencontainers.image.title="Resume RAG Agent" \
      org.opencontainers.image.version=${RESUME_BUILD_ID}

COPY backend/ ./backend/
COPY scripts/ ./scripts/
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

RUN mkdir -p /app/data/documents/originals /app/data/qdrant /app/data/model_cache

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
