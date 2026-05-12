"""ExifTool stay_open daemon wrapper — one process, reused for all writes."""
import json
import logging
import os
import subprocess
import threading
from typing import Optional

log = logging.getLogger(__name__)


class ExifToolDaemon:
    def __init__(self):
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        self._proc = subprocess.Popen(
            ["exiftool", "-stay_open", "True", "-@", "-", "-charset", "UTF8"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        log.info("ExifTool daemon started (pid=%d)", self._proc.pid)

    def _execute(self, args: list[str]) -> str:
        """Send args list to daemon, return stdout up to {ready}."""
        if self._proc is None:
            raise RuntimeError("ExifTool daemon not started")
        cmd = "\n".join(args) + "\n-execute\n"
        with self._lock:
            self._proc.stdin.write(cmd.encode("utf-8"))
            self._proc.stdin.flush()
            lines: list[str] = []
            while True:
                raw = self._proc.stdout.readline()
                line = raw.decode("utf-8", errors="replace").rstrip("\n")
                if line == "{ready}":
                    break
                lines.append(line)
        return "\n".join(lines)

    def write_tags(self, filepath: str, tags: list[str]) -> bool:
        """Append tags to XMP-dc:Subject without overwriting existing ones."""
        if not tags:
            return True
        try:
            args = ["-overwrite_original", "-P"]
            for tag in tags:
                clean = tag.replace("\n", " ").strip()
                if clean:
                    args.append(f"-XMP-dc:Subject+={clean}")
            args.append(os.path.abspath(filepath))
            result = self._execute(args)
            ok = "1 image files updated" in result
            if not ok:
                log.debug("exiftool write result for %s: %s", filepath, result.strip())
            return ok
        except Exception as exc:
            log.error("ExifTool write failed for %s: %s", filepath, exc)
            return False

    def read_tags(self, filepath: str) -> list[str]:
        """Return current XMP-dc:Subject tags."""
        try:
            result = self._execute(["-json", "-XMP-dc:Subject", os.path.abspath(filepath)])
            data = json.loads(result)
            subjects = data[0].get("Subject", []) if data else []
            if isinstance(subjects, str):
                return [subjects]
            return list(subjects)
        except Exception:
            return []

    def stop(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.stdin.write(b"-stay_open\nFalse\n")
            self._proc.stdin.flush()
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.terminate()
        log.info("ExifTool daemon stopped")
