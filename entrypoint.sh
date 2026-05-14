#!/bin/sh
set -e

MODEL="${CLIP_MODEL:-openai/clip-vit-base-patch32}"
MODELS_DIR="${MODELS_DIR:-/app/models}"

echo "[entrypoint] Checking CLIP model: $MODEL"

python - <<'PY'
import os, sys, logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("model-preload")

model_id  = os.environ.get("CLIP_MODEL", "openai/clip-vit-base-patch32")
cache_dir = os.environ.get("MODELS_DIR", "/app/models")

from transformers import CLIPModel, CLIPProcessor

marker = os.path.join(cache_dir, ".model_ready")
if os.path.exists(marker):
    log.info("Model already cached, skipping download.")
    sys.exit(0)

log.info("Downloading processor for %s ...", model_id)
CLIPProcessor.from_pretrained(model_id, cache_dir=cache_dir)
log.info("Downloading model weights for %s ...", model_id)
CLIPModel.from_pretrained(model_id, cache_dir=cache_dir)
log.info("Model download complete.")

open(marker, "w").close()
PY

echo "[entrypoint] Starting application..."
exec python -m app.main
