import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import settings
from app.classifier import Classifier
from app.config import TAGS_FILE
from app.exif import ExifToolDaemon
from app.scanner import Scanner, IMAGE_EXTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

_classifier: Classifier = None
_exiftool:   ExifToolDaemon = None
_scanner:    Scanner = None

_STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _classifier, _exiftool, _scanner

    settings.load()

    _exiftool = ExifToolDaemon()
    _exiftool.start()

    _classifier = Classifier()

    # Apply persisted settings to classifier
    s = settings.get()
    _classifier.update_settings(
        score_threshold=s.get("score_threshold"),
        max_tags=s.get("max_tags"),
    )

    _scanner = Scanner(_classifier, _exiftool)
    threading.Thread(target=_scanner.run, daemon=True, name="scanner").start()

    yield

    _scanner.stop()
    _exiftool.stop()


app = FastAPI(title="photo-tagger", version="1.0.0", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Original routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats")
def stats():
    return _scanner.get_stats()


@app.get("/tags")
def get_tags():
    return _classifier.get_tags()


class TagsBody(BaseModel):
    tags: List[str]


@app.put("/tags")
def put_tags(body: TagsBody):
    tags = [t.strip() for t in body.tags if t.strip()]
    with open(TAGS_FILE, "w", encoding="utf-8") as f:
        json.dump(tags, f, ensure_ascii=False, indent=2)
    _classifier.reload_tags_if_changed()
    return {"tags": tags}


@app.post("/scan")
def trigger_scan():
    _scanner.trigger_full_scan()
    return {"status": "scan queued"}


@app.get("/photo")
def photo_info(path: str):
    p = Path(path)
    if not p.exists():
        raise HTTPException(404, "File not found")
    tags = _exiftool.read_tags(str(p))
    return {"path": str(p), "tags": tags}


# ---------------------------------------------------------------------------
# Admin UI
# ---------------------------------------------------------------------------

@app.get("/admin", include_in_schema=False)
def admin_page():
    return FileResponse(str(_STATIC_DIR / "admin.html"))


# ---------------------------------------------------------------------------
# Admin API
# ---------------------------------------------------------------------------

@app.get("/api/admin/stats")
def admin_stats():
    from app import cache
    scanner_stats = _scanner.get_stats()

    current_settings = settings.get()
    photo_root = current_settings.get("photo_root", "")

    total_photos = 0
    if os.path.isdir(photo_root):
        for root, _, files in os.walk(photo_root):
            for fname in files:
                if Path(fname).suffix.lower() in IMAGE_EXTS:
                    total_photos += 1

    done_count = 0
    if cache._DONE.exists():
        try:
            done_count = sum(1 for _ in cache._DONE.iterdir())
        except Exception:
            pass

    return {
        **scanner_stats,
        "total_photos": total_photos,
        "total_processed_cache": done_count,
        "photo_root": photo_root,
    }


@app.get("/api/admin/config")
def get_config():
    return settings.get()


class ConfigBody(BaseModel):
    photo_root: str = None
    score_threshold: float = None
    max_tags: int = None


@app.post("/api/admin/config")
def update_config(body: ConfigBody):
    updates = {}
    if body.photo_root is not None:
        updates["photo_root"] = body.photo_root.strip()
    if body.score_threshold is not None:
        if not (0.0 < body.score_threshold < 1.0):
            raise HTTPException(400, "score_threshold must be between 0 and 1")
        updates["score_threshold"] = body.score_threshold
    if body.max_tags is not None:
        if body.max_tags < 1:
            raise HTTPException(400, "max_tags must be >= 1")
        updates["max_tags"] = body.max_tags

    saved = settings.save(updates)

    # Apply classifier settings immediately
    _classifier.update_settings(
        score_threshold=saved.get("score_threshold"),
        max_tags=saved.get("max_tags"),
    )

    photo_root_changed = "photo_root" in updates
    return {**saved, "restart_required": photo_root_changed}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8080, log_level="info")
