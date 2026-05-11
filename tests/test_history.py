"""Tests for history.py — add_entry, get_recent, MAX_ENTRIES, and thread safety."""
import json
import threading
from pathlib import Path

import pytest

import history


@pytest.fixture(autouse=True)
def _patch_history_file(monkeypatch, tmp_path):
    """Redirect HISTORY_FILE to a temp directory so tests never touch real data."""
    history_file = tmp_path / "outputs" / "history.json"
    monkeypatch.setattr(history, "HISTORY_FILE", history_file)


# --------------- add_entry ---------------

class TestAddEntry:
    def test_add_entry_creates_file(self, tmp_path):
        assert not history.HISTORY_FILE.exists()
        history.add_entry("tool_a", "/input/file.txt", "did something")
        assert history.HISTORY_FILE.exists()

    def test_add_entry_contains_expected_fields(self, tmp_path):
        history.add_entry("ocr", "/img.png", "extracted text", source="local")
        entries = json.loads(history.HISTORY_FILE.read_text(encoding="utf-8"))
        assert len(entries) == 1
        entry = entries[0]
        assert entry["tool"] == "ocr"
        assert entry["input"] == "/img.png"
        assert entry["summary"] == "extracted text"
        assert entry["source"] == "local"
        assert "time" in entry

    def test_add_entry_default_source_is_web(self, tmp_path):
        history.add_entry("t", "/p", "s")
        entries = json.loads(history.HISTORY_FILE.read_text(encoding="utf-8"))
        assert entries[0]["source"] == "web"

    def test_add_entry_prepends_newest_first(self, tmp_path):
        history.add_entry("a", "/1", "first")
        history.add_entry("b", "/2", "second")
        entries = json.loads(history.HISTORY_FILE.read_text(encoding="utf-8"))
        assert entries[0]["summary"] == "second"
        assert entries[1]["summary"] == "first"

    def test_add_entry_creates_parent_dirs(self, tmp_path):
        # HISTORY_FILE already points to tmp_path / "outputs" / "history.json"
        # which doesn't exist yet — add_entry should mkdir -p
        history.add_entry("t", "/p", "s")
        assert history.HISTORY_FILE.exists()


# --------------- get_recent ---------------

class TestGetRecent:
    def test_get_recent_empty_when_no_file(self, tmp_path):
        assert history.get_recent() == []

    def test_get_recent_returns_specified_count(self, tmp_path):
        for i in range(5):
            history.add_entry("t", f"/f{i}", f"entry {i}")
        result = history.get_recent(count=3)
        assert len(result) == 3
        # Most recent first
        assert result[0]["summary"] == "entry 4"

    def test_get_recent_default_count(self, tmp_path):
        for i in range(25):
            history.add_entry("t", f"/f{i}", f"entry {i}")
        result = history.get_recent()
        assert len(result) == 20  # default count

    def test_get_recent_count_larger_than_entries(self, tmp_path):
        history.add_entry("t", "/f", "only one")
        result = history.get_recent(count=100)
        assert len(result) == 1

    def test_get_recent_returns_list_of_dicts(self, tmp_path):
        history.add_entry("t", "/f", "s")
        result = history.get_recent()
        assert isinstance(result, list)
        assert isinstance(result[0], dict)


# --------------- MAX_ENTRIES limit ---------------

class TestMaxEntries:
    def test_max_entries_is_100(self, tmp_path):
        assert history.MAX_ENTRIES == 100

    def test_entries_capped_at_max(self, tmp_path):
        for i in range(105):
            history.add_entry("t", f"/f{i}", f"entry {i}")
        entries = json.loads(history.HISTORY_FILE.read_text(encoding="utf-8"))
        assert len(entries) == history.MAX_ENTRIES

    def test_oldest_entries_dropped_when_exceeding_max(self, tmp_path):
        for i in range(103):
            history.add_entry("t", f"/f{i}", f"entry {i}")
        entries = json.loads(history.HISTORY_FILE.read_text(encoding="utf-8"))
        # The most recent (entry 102) should be first
        assert entries[0]["summary"] == "entry 102"
        # The oldest kept should be entry 3 (103 entries, keep top 100)
        assert entries[-1]["summary"] == "entry 3"


# --------------- concurrent access ---------------

class TestConcurrentAccess:
    def test_concurrent_add_entry_no_corruption(self, tmp_path):
        """Multiple threads adding entries simultaneously should not corrupt the file."""
        errors = []
        num_threads = 8
        entries_per_thread = 10

        def worker(thread_id):
            try:
                for i in range(entries_per_thread):
                    history.add_entry(
                        f"tool_{thread_id}",
                        f"/file_{thread_id}_{i}",
                        f"summary {thread_id}-{i}",
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(tid,)) for tid in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Errors during concurrent writes: {errors}"

        # File should be valid JSON and respect MAX_ENTRIES
        entries = json.loads(history.HISTORY_FILE.read_text(encoding="utf-8"))
        assert isinstance(entries, list)
        assert len(entries) <= history.MAX_ENTRIES
        # All entries should have required keys
        for entry in entries:
            assert "tool" in entry
            assert "input" in entry
            assert "summary" in entry
            assert "time" in entry
            assert "source" in entry

    def test_concurrent_read_and_write(self, tmp_path):
        """get_recent should not raise while entries are being added."""
        errors = []
        # Pre-populate some entries
        for i in range(10):
            history.add_entry("t", f"/f{i}", f"entry {i}")

        def writer():
            try:
                for i in range(20):
                    history.add_entry("writer", f"/w{i}", f"write {i}")
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(20):
                    result = history.get_recent(count=5)
                    assert isinstance(result, list)
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(3):
            threads.append(threading.Thread(target=writer))
            threads.append(threading.Thread(target=reader))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Errors during concurrent read/write: {errors}"
