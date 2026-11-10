"""
artifacts.py - Run artifacts surfaced in the UI (chart images).

Kept free of Streamlit imports so the freshness rules can be unit-tested.
"""

import os
from typing import Any, Dict, Optional

LEGACY_CHART_FILE = "top_deals_chart.png"

# A chart only belongs to the current answer if it was written after the run
# started. The tolerance absorbs coarse filesystem timestamps (FAT/exFAT and
# some network shares store mtimes with 2 s granularity) without letting a
# chart from an earlier run be attached to a new message.
CHART_MTIME_TOLERANCE_SECONDS = 2.0


def _written_during_run(path: str, start_timestamp: float) -> bool:
    return os.path.getmtime(path) >= start_timestamp - CHART_MTIME_TOLERANCE_SECONDS


def read_chart_bytes(final_state: Dict[str, Any], start_timestamp: float) -> Optional[bytes]:
    """
    Returns the chart image produced by this run as bytes, so each chat message
    keeps its own immutable chart even if the file on disk is later overwritten.
    Prefers the validated `chart_path` from the graph state, then the legacy
    `top_deals_chart.png`. Files not written during this run are ignored: a
    failed or blocked chart step can still report a CHART_PATH that points at
    an earlier run's image.
    """
    for path in (final_state.get("chart_path"), LEGACY_CHART_FILE):
        if not path or not os.path.isfile(path) or not _written_during_run(path, start_timestamp):
            continue
        with open(path, "rb") as chart_file:
            return chart_file.read()
    return None
