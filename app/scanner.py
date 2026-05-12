"""Folder scanner + watchdog watcher.

Uses polling observer by default (USE_POLLING=true) because inotify is
unreliable on SMB/NFS mounts. Polling interval defaults to 30 s.
"""
import logging
import os
import queue
import threading
import time
from pathlib import Path

from app import cache
from app.classifier import Classifier
from app.config import PHOTO_ROOT, USE_POLLING
from app.exif import ExifToolDaemon

log = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp", ".bmp", ".heic"}


# ---------------------------------------------------------------------------
# Watchdog observer (imported lazily to allow easy swap)
# ---------------------------------------------------------------------------

def _make_observer(handler, path: str):
    if USE_POLLING:
        from watchdog.observers.polling import PollingObserver
        obs = PollingObserver(timeout=30)
    else:
        from watchdog.observers import Observer
        obs = Observer()
    obs.schedule(handler, path, recursive=True)
    return obs


class _Handler:
    """Minimal duck-type compatible with watchdog FileSystemEventHandler."""

    def __init__(self, work_queue: queue.Queue):
        self._q = work_queue

    # watchdog calls dispatch() → on_*
    def dispatch(self, event):
        if event.is_directory:
            return
        for method in ("on_created", "on_moved", "on_modified"):
            if event.event_type == method.split("_", 1)[1]:
                getattr(self, method)(event)

    def on_created(self, event):
        self._enqueue(event.src_path)

    def on_moved(self, event):
        self._enqueue(event.dest_path)

    def on_modified(self, event):
        # ignore — we track "done" by hash; re-scan handles modified files
        pass

    def _enqueue(self, path: str):
        if Path(path).suffix.lower() in IMAGE_EXTS:
            self._q.put(path)


class Scanner:
    def __init__(self, classifier: Classifier, exiftool: ExifToolDaemon):
        self.classifier = classifier
        self.exiftool = exiftool
        self._queue: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._stats = {"processed": 0, "skipped": 0, "errors": 0}
        self._tag_counts: dict[str, int] = {}
        self._tag_counts_lock = threading.Lock()
        cache.init()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        with self._tag_counts_lock:
            tag_counts = dict(sorted(self._tag_counts.items(), key=lambda x: x[1], reverse=True))
        return {**self._stats, "queue_size": self._queue.qsize(), "tag_counts": tag_counts}

    def trigger_full_scan(self) -> None:
        threading.Thread(target=self._initial_scan, daemon=True, name="full-scan").start()

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------
    # Main loop (runs in its own thread)
    # ------------------------------------------------------------------

    def run(self) -> None:
        if not os.path.isdir(PHOTO_ROOT):
            log.warning("PHOTO_ROOT %s does not exist yet, waiting …", PHOTO_ROOT)
            while not os.path.isdir(PHOTO_ROOT) and not self._stop.is_set():
                time.sleep(5)

        handler = _Handler(self._queue)
        observer = _make_observer(handler, PHOTO_ROOT)
        observer.start()
        log.info("Watchdog observer started on %s (polling=%s)", PHOTO_ROOT, USE_POLLING)

        threading.Thread(target=self._initial_scan, daemon=True, name="initial-scan").start()

        while not self._stop.is_set():
            if self.classifier.reload_tags_if_changed():
                cache.clear_done()

            try:
                filepath = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            self._process(filepath)

        observer.stop()
        observer.join()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _initial_scan(self) -> None:
        count = 0
        log.info("Initial scan starting: %s", PHOTO_ROOT)
        for root, _dirs, files in os.walk(PHOTO_ROOT):
            for name in files:
                if Path(name).suffix.lower() in IMAGE_EXTS:
                    self._queue.put(os.path.join(root, name))
                    count += 1
        log.info("Initial scan queued %d files", count)

    def _process(self, filepath: str) -> None:
        try:
            fhash = cache.file_hash(filepath)
        except (FileNotFoundError, PermissionError):
            return  # file disappeared or inaccessible

        if cache.is_done(fhash):
            self._stats["skipped"] += 1
            return

        try:
            tags = self.classifier.classify_image(filepath)
            if tags:
                ok = self.exiftool.write_tags(filepath, tags)
                if ok:
                    log.info("Tagged %s → %s", filepath, tags)
                    with self._tag_counts_lock:
                        for tag in tags:
                            self._tag_counts[tag] = self._tag_counts.get(tag, 0) + 1
                else:
                    log.warning("ExifTool write failed for %s", filepath)
            else:
                log.debug("No tags above threshold for %s", filepath)

            cache.mark_done(fhash)
            self._stats["processed"] += 1

        except Exception as exc:
            log.error("Processing failed for %s: %s", filepath, exc)
            self._stats["errors"] += 1
