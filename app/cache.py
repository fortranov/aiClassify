import hashlib
import logging
from pathlib import Path

from app.config import CACHE_DIR

log = logging.getLogger(__name__)

_EMBEDDINGS = Path(CACHE_DIR) / "embeddings"
_DONE       = Path(CACHE_DIR) / "done"


def init():
    _EMBEDDINGS.mkdir(parents=True, exist_ok=True)
    _DONE.mkdir(parents=True, exist_ok=True)


def file_hash(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def is_done(fhash: str) -> bool:
    return (_DONE / fhash).exists()


def mark_done(fhash: str) -> None:
    (_DONE / fhash).touch()


def clear_done() -> int:
    removed = 0
    for p in _DONE.iterdir():
        p.unlink(missing_ok=True)
        removed += 1
    log.info("Cleared %d done markers (tags changed)", removed)
    return removed
