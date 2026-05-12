"""CLIP-based zero-shot image classifier.

Tags are loaded from tags.json. Text embeddings are computed once at startup
and reused for every image. When tags.json changes on disk the embeddings
are recomputed automatically.
"""
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from app.config import CLIP_MODEL, MODELS_DIR, SCORE_THRESHOLD, MAX_TAGS, TAGS_FILE

log = logging.getLogger(__name__)


class Classifier:
    def __init__(self):
        self._load_model()
        self._tags: list[str] = []
        self._tag_embeddings: Optional[torch.Tensor] = None
        self._tags_mtime: float = 0.0
        self._score_threshold: float = SCORE_THRESHOLD
        self._max_tags: int = MAX_TAGS
        self._load_tags()

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        log.info("Loading CLIP model %s …", CLIP_MODEL)
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL, cache_dir=MODELS_DIR)
        self.model = CLIPModel.from_pretrained(CLIP_MODEL, cache_dir=MODELS_DIR)
        self.model.eval()
        log.info("CLIP model loaded")

    # ------------------------------------------------------------------
    # Tags
    # ------------------------------------------------------------------

    def _load_tags(self) -> None:
        path = Path(TAGS_FILE)
        if not path.exists():
            log.warning("tags.json not found at %s, using empty list", TAGS_FILE)
            self._tags = []
            self._tag_embeddings = None
            return

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            tags = [str(t).strip() for t in data if str(t).strip()]
        elif isinstance(data, dict):
            tags = []
            for category_tags in data.values():
                tags.extend(str(t).strip() for t in category_tags if str(t).strip())
        else:
            raise ValueError("tags.json must be a JSON array or object")

        self._tags = tags
        self._tags_mtime = path.stat().st_mtime
        self._compute_tag_embeddings()
        log.info("Loaded %d tags from %s", len(tags), TAGS_FILE)

    def _compute_tag_embeddings(self) -> None:
        if not self._tags:
            self._tag_embeddings = None
            return
        with torch.no_grad():
            inputs = self.processor(text=self._tags, return_tensors="pt", padding=True)
            feats = self.model.get_text_features(**inputs)
            self._tag_embeddings = feats / feats.norm(dim=-1, keepdim=True)

    def reload_tags_if_changed(self) -> bool:
        """Return True and reload if tags.json was modified."""
        try:
            mtime = Path(TAGS_FILE).stat().st_mtime
        except FileNotFoundError:
            return False
        if mtime != self._tags_mtime:
            self._load_tags()
            return True
        return False

    def get_tags(self) -> list[str]:
        return list(self._tags)

    def update_settings(self, score_threshold: float = None, max_tags: int = None) -> None:
        if score_threshold is not None:
            self._score_threshold = score_threshold
        if max_tags is not None:
            self._max_tags = max_tags

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def classify_image(self, image_path: str) -> list[str]:
        """Return list of matching tags above SCORE_THRESHOLD."""
        if self._tag_embeddings is None or not self._tags:
            return []
        try:
            image = Image.open(image_path).convert("RGB")
            image.thumbnail((512, 512), Image.LANCZOS)

            with torch.no_grad():
                inputs = self.processor(images=image, return_tensors="pt")
                img_feat = self.model.get_image_features(**inputs)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                # cosine similarity: [n_tags]
                scores = (img_feat @ self._tag_embeddings.T).squeeze(0).numpy()

            matched = [
                (self._tags[i], float(scores[i]))
                for i in range(len(self._tags))
                if scores[i] >= self._score_threshold
            ]
            matched.sort(key=lambda x: x[1], reverse=True)
            return [tag for tag, _ in matched[:self._max_tags]]

        except Exception as exc:
            log.error("classify_image failed for %s: %s", image_path, exc)
            return []
