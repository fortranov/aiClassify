"""CLIP-based zero-shot image classifier.

Supports both standard OpenAI CLIP models (via transformers) and multilingual
M-CLIP models (via the multilingual-clip package). M-CLIP replaces the text
encoder with XLM-Roberta while keeping the standard CLIP vision encoder.

Tags are loaded from tags.json. Text embeddings are computed once at startup
and reused for every image. When tags.json changes on disk the embeddings
are recomputed automatically.
"""
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from app.config import CLIP_MODEL, MODELS_DIR, SCORE_THRESHOLD, MAX_TAGS, TAGS_FILE

log = logging.getLogger(__name__)

# Maps M-CLIP model name fragments to the corresponding CLIP vision encoder.
_MCLIP_VISION_MAP = {
    "Vit-B-32": "openai/clip-vit-base-patch32",
    "Vit-B-16": "openai/clip-vit-base-patch16",
    "Vit-L-14": "openai/clip-vit-large-patch14",
}


def _is_mclip(model_name: str) -> bool:
    return model_name.startswith("M-CLIP/") or "multilingual-clip" in model_name.lower()


def _mclip_vision_model(model_name: str) -> str:
    for key, val in _MCLIP_VISION_MAP.items():
        if key in model_name:
            return val
    return "openai/clip-vit-base-patch32"


class Classifier:
    def __init__(self):
        self._tags: list[str] = []
        self._tag_embeddings: Optional[torch.Tensor] = None
        self._tags_mtime: float = 0.0
        self._score_threshold: float = SCORE_THRESHOLD
        self._max_tags: int = MAX_TAGS
        self._use_mclip: bool = _is_mclip(CLIP_MODEL)
        self._load_model()
        self._load_tags()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self._use_mclip:
            self._load_mclip_model()
        else:
            self._load_standard_model()

    def _load_standard_model(self) -> None:
        log.info("Loading CLIP model %s …", CLIP_MODEL)
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL, cache_dir=MODELS_DIR)
        self._model = CLIPModel.from_pretrained(CLIP_MODEL, cache_dir=MODELS_DIR)
        self._model.eval()
        log.info("CLIP model loaded")

    def _load_mclip_model(self) -> None:
        log.info("Loading M-CLIP text encoder %s …", CLIP_MODEL)
        from multilingual_clip import pt_multilingual_clip
        import transformers as hf
        self._mclip_text = pt_multilingual_clip.MultilingualCLIP.from_pretrained(
            CLIP_MODEL, cache_dir=MODELS_DIR)
        self._mclip_text.eval()
        self._mclip_tokenizer = hf.AutoTokenizer.from_pretrained(
            CLIP_MODEL, cache_dir=MODELS_DIR)

        vision_name = _mclip_vision_model(CLIP_MODEL)
        log.info("Loading M-CLIP vision encoder %s …", vision_name)
        self.processor = CLIPProcessor.from_pretrained(vision_name, cache_dir=MODELS_DIR)
        self._vision_model = CLIPModel.from_pretrained(vision_name, cache_dir=MODELS_DIR)
        self._vision_model.eval()
        log.info("M-CLIP model loaded")

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
        if self._use_mclip:
            with torch.no_grad():
                feats = self._mclip_text.forward(self._tags, self._mclip_tokenizer)
            self._tag_embeddings = feats / feats.norm(dim=-1, keepdim=True)
        else:
            with torch.no_grad():
                inputs = self.processor(text=self._tags, return_tensors="pt", padding=True)
                feats = self._model.get_text_features(**inputs)
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
        """Return list of matching tags above score threshold."""
        if self._tag_embeddings is None or not self._tags:
            return []
        try:
            image = Image.open(image_path).convert("RGB")
            image.thumbnail((512, 512), Image.LANCZOS)

            with torch.no_grad():
                inputs = self.processor(images=image, return_tensors="pt")
                if self._use_mclip:
                    img_feat = self._vision_model.get_image_features(**inputs)
                else:
                    img_feat = self._model.get_image_features(**inputs)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
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
