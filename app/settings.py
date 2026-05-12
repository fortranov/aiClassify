"""Runtime-configurable settings stored in CACHE_DIR/settings.json.

Overrides env-var defaults from config.py and can be updated via API.
"""
import json
import logging
import threading
from pathlib import Path

log = logging.getLogger(__name__)
_lock = threading.Lock()
_current: dict = {}


def _file() -> Path:
    from app.config import CACHE_DIR
    return Path(CACHE_DIR) / "settings.json"


def load() -> None:
    global _current
    from app.config import PHOTO_ROOT, SCORE_THRESHOLD, MAX_TAGS
    defaults = {
        "photo_root": PHOTO_ROOT,
        "score_threshold": SCORE_THRESHOLD,
        "max_tags": MAX_TAGS,
    }
    f = _file()
    if f.exists():
        try:
            with open(f, encoding="utf-8") as fp:
                saved = json.load(fp)
            with _lock:
                _current = {**defaults, **saved}
            log.info("Settings loaded from %s", f)
            return
        except Exception as exc:
            log.warning("Failed to load settings.json: %s", exc)
    with _lock:
        _current = defaults.copy()


def get() -> dict:
    with _lock:
        if not _current:
            load()
        return dict(_current)


def save(updates: dict) -> dict:
    with _lock:
        if not _current:
            load()
        _current.update(updates)
        result = dict(_current)
    f = _file()
    f.parent.mkdir(parents=True, exist_ok=True)
    with open(f, "w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2, ensure_ascii=False)
    log.info("Settings saved: %s", updates)
    return result
