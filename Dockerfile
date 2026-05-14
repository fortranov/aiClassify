FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_CACHE=/app/models \
    HF_HOME=/app/models

RUN apt-get update && apt-get install -y --no-install-recommends \
        libimage-exiftool-perl \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install torch CPU-only first (much smaller than the default CUDA build)
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app /app/app
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

RUN mkdir -p /app/cache /app/models /photos

# Pre-download the CLIP model into the image so it is available without
# internet access at runtime (the model cache is kept in /app/models which
# can also be overridden by a bind-mount; the entrypoint skips the download
# when a .model_ready marker already exists).
RUN python -c "\
from transformers import CLIPModel, CLIPProcessor; \
import os; \
m = os.environ.get('CLIP_MODEL', 'openai/clip-vit-base-patch32'); \
d = '/app/models'; \
CLIPProcessor.from_pretrained(m, cache_dir=d); \
CLIPModel.from_pretrained(m, cache_dir=d); \
open(d + '/.model_ready', 'w').close(); \
print('Model pre-downloaded.')"

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

ENTRYPOINT ["/app/entrypoint.sh"]
