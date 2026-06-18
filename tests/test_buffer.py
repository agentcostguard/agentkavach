"""Buffer durability tests for agentkavach.buffer."""

from __future__ import annotations

import json
import threading

import pytest

from agentkavach.buffer import _RING_BUFFER_MAX, Buffer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def disk_buffer(tmp_path):
    """Buffer backed by a temporary file."""
    path = str(tmp_path / "buffer.jsonl")
    return Buffer(path=path)


@pytest.fixture()
def memory_buffer():
    """In-memory ring buffer (force by giving an unwritable path)."""
    buf = Buffer.__new__(Buffer)
    buf._lock = threading.Lock()
    buf._ring = __import__("collections").deque(maxlen=_RING_BUFFER_MAX)
    buf._path = None
    buf._mode = "memory"
    return buf


# ---------------------------------------------------------------------------
# Disk buffer
# ---------------------------------------------------------------------------


class TestDiskBuffer:
    def test_write_and_read(self, disk_buffer: Buffer):
        event = {"agent": "a", "cost": 0.05}
        disk_buffer.write(event)
        events = disk_buffer.read_all()
        assert len(events) == 1
        assert events[0]["agent"] == "a"

    def test_multiple_writes(self, disk_buffer: Buffer):
        for i in range(5):
            disk_buffer.write({"i": i})
        assert disk_buffer.count() == 5
        events = disk_buffer.read_all()
        assert [e["i"] for e in events] == [0, 1, 2, 3, 4]

    def test_purge(self, disk_buffer: Buffer):
        for i in range(5):
            disk_buffer.write({"i": i})
        disk_buffer.purge(up_to=3)
        events = disk_buffer.read_all()
        assert len(events) == 2
        assert events[0]["i"] == 3

    def test_purge_all(self, disk_buffer: Buffer):
        for i in range(3):
            disk_buffer.write({"i": i})
        disk_buffer.purge(up_to=10)
        assert disk_buffer.count() == 0

    def test_survives_restart(self, tmp_path):
        """Buffer contents persist across object lifetimes."""
        path = str(tmp_path / "buffer.jsonl")
        buf1 = Buffer(path=path)
        buf1.write({"cost": 1.0})
        buf1.write({"cost": 2.0})

        # Create a new buffer pointing to the same file.
        buf2 = Buffer(path=path)
        events = buf2.read_all()
        assert len(events) == 2
        assert events[0]["cost"] == 1.0

    def test_mode_is_disk_or_tmp(self, disk_buffer: Buffer):
        # Explicit path won't set mode via auto-detect, but path should exist.
        assert disk_buffer.path is not None

    def test_jsonl_format(self, disk_buffer: Buffer):
        disk_buffer.write({"key": "value"})
        with open(disk_buffer.path) as f:  # type: ignore
            content = f.read()
        assert content.strip() == '{"key": "value"}'
        # Verify it's valid JSON.
        parsed = json.loads(content.strip())
        assert parsed == {"key": "value"}

    def test_handles_special_types(self, disk_buffer: Buffer):
        """Non-serializable values are converted via default=str."""
        from datetime import datetime

        disk_buffer.write({"ts": datetime(2026, 3, 13)})
        events = disk_buffer.read_all()
        assert len(events) == 1
        assert "2026" in events[0]["ts"]


# ---------------------------------------------------------------------------
# In-memory ring buffer
# ---------------------------------------------------------------------------


class TestMemoryBuffer:
    def test_write_and_read(self, memory_buffer: Buffer):
        memory_buffer.write({"cost": 0.10})
        events = memory_buffer.read_all()
        assert len(events) == 1

    def test_purge(self, memory_buffer: Buffer):
        for i in range(5):
            memory_buffer.write({"i": i})
        memory_buffer.purge(up_to=2)
        events = memory_buffer.read_all()
        assert len(events) == 3
        assert events[0]["i"] == 2

    def test_ring_buffer_evicts_oldest(self, memory_buffer: Buffer):
        for i in range(_RING_BUFFER_MAX + 100):
            memory_buffer.write({"i": i})
        assert memory_buffer.count() == _RING_BUFFER_MAX
        events = memory_buffer.read_all()
        # Oldest events should have been evicted.
        assert events[0]["i"] == 100

    def test_mode(self, memory_buffer: Buffer):
        assert memory_buffer.mode == "memory"
        assert memory_buffer.path is None

    def test_count(self, memory_buffer: Buffer):
        assert memory_buffer.count() == 0
        memory_buffer.write({"x": 1})
        assert memory_buffer.count() == 1


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------


class TestAutoDetect:
    def test_auto_detect_finds_writable_path(self):
        buf = Buffer()
        assert buf.mode in ("disk", "tmp", "memory")

    def test_auto_detect_prefers_home_dir(self):
        buf = Buffer()
        # On most systems, home dir is writable.
        if buf.mode == "disk":
            assert ".agentkavach" in str(buf.path)


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestBufferThreadSafety:
    def test_concurrent_writes(self, disk_buffer: Buffer):
        errors: list[Exception] = []

        def writer():
            try:
                for i in range(50):
                    disk_buffer.write({"i": i})
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert disk_buffer.count() == 250

    def test_concurrent_writes_memory(self, memory_buffer: Buffer):
        errors: list[Exception] = []

        def writer():
            try:
                for i in range(50):
                    memory_buffer.write({"i": i})
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert memory_buffer.count() == 250


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestBufferEdgeCases:
    def test_read_empty_buffer(self, disk_buffer: Buffer):
        assert disk_buffer.read_all() == []

    def test_purge_empty_buffer(self, disk_buffer: Buffer):
        disk_buffer.purge(up_to=5)
        assert disk_buffer.count() == 0

    def test_malformed_line_skipped(self, tmp_path):
        path = tmp_path / "buffer.jsonl"
        path.write_text('{"valid": true}\nnot json\n{"also": "valid"}\n')
        buf = Buffer(path=str(path))
        events = buf.read_all()
        assert len(events) == 2
        assert events[0]["valid"] is True
        assert events[1]["also"] == "valid"

    def test_try_path_failure_returns_false(self, tmp_path):
        """_try_path returns False when path is not writable."""
        buf = Buffer.__new__(Buffer)
        buf._lock = threading.Lock()
        buf._ring = __import__("collections").deque(maxlen=_RING_BUFFER_MAX)
        buf._path = None
        buf._mode = "memory"
        # Use a path under /proc (or similar unwritable location).
        result = buf._try_path(__import__("pathlib").Path("/dev/null/impossible/buffer.jsonl"))
        assert result is False

    def test_path_info_memory(self):
        """_path_info returns ring buffer description when in memory mode."""
        buf = Buffer.__new__(Buffer)
        buf._lock = threading.Lock()
        buf._ring = __import__("collections").deque(maxlen=_RING_BUFFER_MAX)
        buf._path = None
        buf._mode = "memory"
        info = buf._path_info()
        assert "in-memory ring" in info
        assert str(_RING_BUFFER_MAX) in info

    def test_path_info_disk(self, disk_buffer: Buffer):
        """_path_info returns file path when in disk mode."""
        info = disk_buffer._path_info()
        assert "buffer.jsonl" in info

    def test_write_oserror_falls_back_to_memory(self, disk_buffer: Buffer):
        """When file write fails, event falls back to in-memory ring."""

        original_path = disk_buffer._path
        # Point to an unwritable path to trigger OSError.
        disk_buffer._path = __import__("pathlib").Path("/dev/null/impossible/buffer.jsonl")
        disk_buffer.write({"fallback": True})
        # Event should be in the ring buffer.
        assert len(disk_buffer._ring) == 1
        assert disk_buffer._ring[0]["fallback"] is True
        # Restore path.
        disk_buffer._path = original_path

    def test_read_file_oserror(self, disk_buffer: Buffer):
        """_read_file returns empty list when file read fails."""
        original_path = disk_buffer._path
        disk_buffer._path = __import__("pathlib").Path("/dev/null/impossible/buffer.jsonl")
        # _read_file should handle OSError gracefully.
        events = disk_buffer._read_file()
        assert events == []
        disk_buffer._path = original_path

    def test_read_file_nonexistent_path(self, tmp_path):
        """_read_file returns empty list when path does not exist."""
        path = tmp_path / "nonexistent.jsonl"
        buf = Buffer.__new__(Buffer)
        buf._lock = threading.Lock()
        buf._ring = __import__("collections").deque(maxlen=_RING_BUFFER_MAX)
        buf._path = path
        buf._mode = "disk"
        events = buf._read_file()
        assert events == []

    def test_purge_file_oserror(self, disk_buffer: Buffer, tmp_path):
        """_purge_file handles OSError gracefully."""
        disk_buffer.write({"i": 0})
        disk_buffer.write({"i": 1})
        original_path = disk_buffer._path
        # Point to unwritable location for the write phase of purge.
        # First we need _read_file to succeed, then write to fail.
        # Simplest: make the file read-only after writing.
        import os

        os.chmod(original_path, 0o444)
        # purge should not raise.
        disk_buffer._purge_file(1)
        # Restore permissions for cleanup.
        os.chmod(original_path, 0o644)

    def test_purge_file_none_path(self):
        """_purge_file returns early when path is None."""
        buf = Buffer.__new__(Buffer)
        buf._lock = threading.Lock()
        buf._ring = __import__("collections").deque(maxlen=_RING_BUFFER_MAX)
        buf._path = None
        buf._mode = "memory"
        # Should not raise.
        buf._purge_file(5)

    def test_count_file_nonexistent(self, tmp_path):
        """_count_file returns 0 when file does not exist."""
        buf = Buffer.__new__(Buffer)
        buf._lock = threading.Lock()
        buf._ring = __import__("collections").deque(maxlen=_RING_BUFFER_MAX)
        buf._path = tmp_path / "nonexistent.jsonl"
        buf._mode = "disk"
        assert buf._count_file() == 0

    def test_count_file_none_path(self):
        """_count_file returns 0 when path is None."""
        buf = Buffer.__new__(Buffer)
        buf._lock = threading.Lock()
        buf._ring = __import__("collections").deque(maxlen=_RING_BUFFER_MAX)
        buf._path = None
        buf._mode = "memory"
        assert buf._count_file() == 0

    def test_count_file_oserror(self, disk_buffer: Buffer):
        """_count_file returns 0 on OSError."""
        disk_buffer.write({"i": 0})
        original_path = disk_buffer._path
        disk_buffer._path = __import__("pathlib").Path("/dev/null/impossible/buffer.jsonl")
        assert disk_buffer._count_file() == 0
        disk_buffer._path = original_path


# ---------------------------------------------------------------------------
# Auto-detection fallback chain
# ---------------------------------------------------------------------------


class TestAutoDetectFallback:
    def test_falls_back_to_tmp_when_home_unwritable(self, monkeypatch):
        """When home dir is unwritable, auto-detect falls to tmp."""
        from pathlib import Path

        def fake_home():
            return Path("/dev/null/impossible")

        monkeypatch.setattr(Path, "home", staticmethod(fake_home))
        buf = Buffer()
        # Should fall back to tmp or memory, not crash.
        assert buf.mode in ("tmp", "memory")

    def test_falls_back_to_memory_when_all_unwritable(self, monkeypatch):
        """When both home and tmp are unwritable, falls to memory."""
        import tempfile
        from pathlib import Path

        monkeypatch.setattr(Path, "home", staticmethod(lambda: Path("/dev/null/impossible")))
        monkeypatch.setattr(tempfile, "gettempdir", lambda: "/dev/null/impossible")
        buf = Buffer()
        assert buf.mode == "memory"
        assert buf.path is None
