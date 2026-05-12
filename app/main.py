import json
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.classifier import Classifier
from app.config import TAGS_FILE
from app.exif import ExifToolDaemon
from app.scanner import Scanner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

_classifier: Classifier = None
_exiftool:   ExifToolDaemon = None
_scanner:    Scanner = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _classifier, _exiftool, _scanner

    _exiftool = ExifToolDaemon()
    _exiftool.start()

    _classifier = Classifier()

    _scanner = Scanner(_classifier, _exiftool)
    threading.Thread(target=_scanner.run, daemon=True, name="scanner").start()

    yield

    _scanner.stop()
    _exiftool.stop()


app = FastAPI(title="photo-tagger", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Routes
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8080, log_level="info")
