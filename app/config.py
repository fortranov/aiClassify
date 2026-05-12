import os

PHOTO_ROOT   = os.getenv("PHOTO_ROOT",   "/photos")
CACHE_DIR    = os.getenv("CACHE_DIR",    "/app/cache")
MODELS_DIR   = os.getenv("MODELS_DIR",   "/app/models")
TAGS_FILE    = os.getenv("TAGS_FILE",    "/app/app/tags.json")

CLIP_MODEL       = os.getenv("CLIP_MODEL",       "openai/clip-vit-base-patch32")
SCORE_THRESHOLD  = float(os.getenv("SCORE_THRESHOLD",  "0.24"))
MAX_TAGS         = int(os.getenv("MAX_TAGS",           "10"))
USE_POLLING      = os.getenv("USE_POLLING", "true").lower() == "true"
