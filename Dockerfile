# --------------------------------------------------------------------------- #
# Stage 1 -- builder: install dependencies and train the model bundle.
# Training happens at build time so the runtime image is stateless: any pod can
# serve immediately without a shared volume or a warm-up job.
# --------------------------------------------------------------------------- #
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY tests ./tests
COPY pyproject.toml .

# Fail the build if the pipeline is broken -- the image is only worth shipping
# if the tests pass.
ENV PYTHONPATH=/build/src
RUN python -m pytest tests -q

# Train on the bundled simulator. Swap for `--source yelp` with the dataset
# mounted at /build/data to produce a production bundle.
ARG TRAIN_ARGS="--source synthetic --n-businesses 120 --months 36"
RUN python -m reputation.pipeline.train ${TRAIN_ARGS} --artifact-dir /build/artifacts

# --------------------------------------------------------------------------- #
# Stage 2 -- runtime: only what is needed to serve.
# --------------------------------------------------------------------------- #
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    ARTIFACT_DIR=/app/artifacts

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY --from=builder /build/artifacts ./artifacts

# Run as a non-root user: required by most hardened Kubernetes admission policies.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "reputation.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
