# syntax=docker/dockerfile:1
#
# Multi-stage build for the RAG Tutor app (core dependencies only).
#
# The image runs the full Streamlit UI (chat, quiz, flashcards, PDF reader,
# eval harness). The optional [hybrid] extras (BGE-M3 embeddings + BGE
# reranker, which pull in torch) are intentionally excluded to keep the image
# small; see the README "Run with Docker" section for enabling them.
#
# Build:  docker build -t rag-tutor .
# Run:    docker run -p 8501:8501 -e GEMINI_API_KEY=... rag-tutor

ARG PYTHON_VERSION=3.12

# --------------------------------------------------------------------------- #
# Builder stage: install dependencies into a clean prefix.                     #
# --------------------------------------------------------------------------- #
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# Build tools needed to compile any C extensions in the dep tree.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (cached layer) using an isolated prefix so we can
# copy just the installed packages to the runtime image. Use a NON-editable
# install so the package source is copied into site-packages (editable installs
# only reference the build-dir source, which doesn't exist in the runtime stage).
COPY pyproject.toml README.md ./
COPY rag_tutor ./rag_tutor

RUN pip install --prefix=/install .


# --------------------------------------------------------------------------- #
# Runtime stage: minimal image with only what's needed to run.                 #
# --------------------------------------------------------------------------- #
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Default config: data lives under /data and is volume-mounted.
    CHROMA_DB_PATH=/data/chroma_db \
    STUDY_MATERIALS_DIR=/data/study_materials

# Copy the installed site-packages from the builder.
COPY --from=builder /install /usr/local

# Create a non-root user and a data directory owned by it.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /data/chroma_db /data/study_materials \
    && chown -R appuser:appuser /data

USER appuser
WORKDIR /home/appuser

# Streamlit default port.
EXPOSE 8501

# Persist data across container recreations when a volume is mounted at /data.
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health').status == 200 else 1)"

# Headless, no browser auto-open, bind all interfaces.
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

CMD ["python", "-m", "rag_tutor", "run"]
