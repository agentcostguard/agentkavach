"""Local disk buffer with automatic fallback chain.

Fallback order:
    1. ``~/.agentkavach/buffer.jsonl``  (best — survives restarts)
    2. ``/tmp/agentkavach/buffer.jsonl`` (Lambda, containers)
    3. In-memory ring buffer          (read-only filesystems)

The buffer stores cost events as newline-delimited JSON (JSONL).
Events are appended on every ``post_flight`` and flushed when the
OTel exporter successfully delivers a batch.

Usage (internal):
    buf = Buffer()                # auto-detects best mode
    buf.write({"agent": "a", "cost": 0.05, ...})
    events = buf.read_all()       # returns unflushed events
    buf.purge(up_to=10)           # remove delivered events
"""

from __future__ import annotations

import collections
import json
import logging
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Maximum events kept in the in-memory ring buffer fallback.
_RING_BUFFER_MAX = 10_000

# Directory names (not full paths — resolved at runtime).
_HOME_DIR_NAME = ".agentkavach"
_TMP_DIR_NAME = "agentkavach"
_BUFFER_FILE = "buffer.jsonl"


class Buffer:
    """Append-only event buffer with automatic storage fallback.

    Thread-safe: all mutations go through ``_lock``.
    """

    def __init__(self, path: Optional[str] = None, *, fsync: bool = False) -> None:
        self._lock = threading.Lock()
        self._ring: collections.deque[Dict[str, Any]] = collections.deque(maxlen=_RING_BUFFER_MAX)
        self._path: Optional[Path] = None
        self._mode: str = "memory"
        self._fsync: bool = fsync

        if path is not None:
            # Explicit path — use it or fail.
            self._try_path(Path(path))
        else:
            self._auto_detect()

        logger.info("AgentKavach: using %s buffer%s", self._mode, self._path_info())

    # -- public interface ---------------------------------------------------

    @property
    def mode(self) -> str:
        """Return the active buffer mode: ``"disk"``, ``"tmp"``, or ``"memory"``."""
        return self._mode

    @property
    def path(self) -> Optional[Path]:
        """Return the buffer file path (``None`` for in-memory mode)."""
        return self._path

    def write(self, event: Dict[str, Any]) -> None:
        """Append *event* to the buffer."""
        with self._lock:
            if self._path is not None:
                self._append_to_file(event)
            else:
                self._ring.append(event)

    def read_all(self) -> List[Dict[str, Any]]:
        """Return all buffered events (oldest first)."""
        with self._lock:
            if self._path is not None:
                return self._read_file()
            return list(self._ring)

    def purge(self, up_to: int) -> None:
        """Remove the first *up_to* events from the buffer.

        For file-backed buffers this rewrites the file without the
        purged lines.  For in-memory buffers it pops from the left.
        """
        with self._lock:
            if self._path is not None:
                self._purge_file(up_to)
            else:
                for _ in range(min(up_to, len(self._ring))):
                    self._ring.popleft()

    def count(self) -> int:
        """Return the number of buffered events."""
        with self._lock:
            if self._path is not None:
                return self._count_file()
            return len(self._ring)

    # -- auto-detection -----------------------------------------------------

    def _auto_detect(self) -> None:
        """Try disk → /tmp → memory."""
        home = Path.home() / _HOME_DIR_NAME
        if self._try_path(home / _BUFFER_FILE):
            self._mode = "disk"
            return

        tmp = Path(tempfile.gettempdir()) / _TMP_DIR_NAME
        if self._try_path(tmp / _BUFFER_FILE):
            self._mode = "tmp"
            return

        self._mode = "memory"

    def _try_path(self, path: Path) -> bool:
        """Attempt to create and write to *path*.  Returns success."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Verify we can actually write by touching the file.
            path.touch(exist_ok=True)
            self._path = path
            return True
        except OSError:
            return False

    def _path_info(self) -> str:
        if self._path is not None:
            return f" at {self._path}"
        return " (in-memory ring, max {0} events)".format(_RING_BUFFER_MAX)

    # -- file I/O (called under lock) --------------------------------------

    def _append_to_file(self, event: Dict[str, Any]) -> None:
        try:
            with open(self._path, "a") as f:  # type: ignore[arg-type]
                f.write(json.dumps(event, default=str) + "\n")
                if self._fsync:
                    f.flush()
                    import os as _os

                    _os.fsync(f.fileno())
        except OSError:
            logger.warning("Buffer write failed — falling back to memory")
            self._ring.append(event)

    def _read_file(self) -> List[Dict[str, Any]]:
        if self._path is None or not self._path.exists():
            return []
        events: List[Dict[str, Any]] = []
        try:
            with open(self._path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            logger.warning("Skipping malformed buffer line")
        except OSError:
            logger.warning("Buffer read failed")
        return events

    def _purge_file(self, up_to: int) -> None:
        if self._path is None:
            return
        events = self._read_file()
        remaining = events[up_to:]
        try:
            with open(self._path, "w") as f:
                for event in remaining:
                    f.write(json.dumps(event, default=str) + "\n")
        except OSError:
            logger.warning("Buffer purge failed")

    def _count_file(self) -> int:
        if self._path is None or not self._path.exists():
            return 0
        count = 0
        try:
            with open(self._path) as f:
                for line in f:
                    if line.strip():
                        count += 1
        except OSError:
            pass
        return count
