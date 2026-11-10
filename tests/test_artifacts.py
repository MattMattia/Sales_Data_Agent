"""Tests for artifacts.read_chart_bytes: only charts written during the current run are attached."""

import os
import time

from artifacts import CHART_MTIME_TOLERANCE_SECONDS, LEGACY_CHART_FILE, read_chart_bytes

PNG = b"\x89PNG\r\n\x1a\n-fake"


def _write(path, age_seconds=0.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(PNG)
    stamp = time.time() - age_seconds
    os.utime(path, (stamp, stamp))


def test_returns_chart_from_state_written_during_this_run(isolated_cwd):
    _write(isolated_cwd / "charts" / "deals.png")

    assert read_chart_bytes({"chart_path": "charts/deals.png"}, time.time() - 5) == PNG


def test_ignores_chart_from_state_left_by_an_earlier_run(isolated_cwd):
    _write(isolated_cwd / "charts" / "deals.png", age_seconds=600)

    assert read_chart_bytes({"chart_path": "charts/deals.png"}, time.time()) is None


def test_ignores_stale_legacy_chart_and_accepts_fresh_one(isolated_cwd):
    start = time.time()
    _write(isolated_cwd / LEGACY_CHART_FILE, age_seconds=600)
    assert read_chart_bytes({}, start) is None

    _write(isolated_cwd / LEGACY_CHART_FILE, age_seconds=CHART_MTIME_TOLERANCE_SECONDS / 2)
    assert read_chart_bytes({}, start) == PNG


def test_returns_none_when_no_chart_exists(isolated_cwd):
    assert read_chart_bytes({"chart_path": "missing.png"}, time.time()) is None
