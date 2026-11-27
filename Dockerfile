# S7c — the model's web API in one image (specs/s7c-serve/spec.md; D-045, D-046).
#
# Build (from the repository folder, with the private model folder in ./serving):
#   docker build -t anthro-api .
# Run:
#   docker run -p 8080:8080 -e ANTHRO_API_KEY=choose-a-long-key anthro-api
#
# ./serving holds model.pt, eval_report.json and MANIFEST.json from `anthro prepare-serving`.
# It is never committed (only its README is), so this image must stay PRIVATE (D-014; BodyM CC BY-NC 4.0).
# Works on Intel/AMD (x86-64) and Arm (e.g. Oracle Ampere) machines.
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    TORCH_HOME=/models/torch ANTHRO_MODEL_DIR=/models ANTHRO_TOOL=torchvision PORT=8080

# PyTorch CPU build: about a third of the size of the GPU build; a server needs no GPU (S7c).
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[serve]"

COPY serving/ /models/
# Bake the segmentation tool's weights into the image, so the first request does not wait for a download.
# BAKE_WEIGHTS=0 skips it (CI checks that the image builds and starts, without the private model).
ARG BAKE_WEIGHTS=1
RUN if [ "$BAKE_WEIGHTS" = "1" ]; then \
      python -c "from anthro.segment import TorchvisionSegmenter; TorchvisionSegmenter('cpu')"; fi

# Run as an ordinary user, not root.
RUN useradd --create-home app && chown -R app /models
USER app
EXPOSE 8080
CMD ["sh", "-c", "uvicorn anthro.serve:create_app_from_env --factory --host 0.0.0.0 --port ${PORT} --workers 1"]
