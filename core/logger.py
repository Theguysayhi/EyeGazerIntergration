"""
logger.py
Lightweight in-memory rolling log.  Entries are printed to stdout and stored
in a fixed-size deque so old entries are automatically discarded once used.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

import eyeware.beam_eye_tracker as bet
from PIL import Image

# How many log entries to keep in memory at once before discarding old ones
MAX_LOG_ENTRIES = 100


@dataclass
class LogEntry:
    timestamp: float                    # wall-clock seconds since epoch
    gaze_x: int
    gaze_y: int
    confidence: str
    region_size: tuple[int, int]        # (width, height) of the captured image
    segment_name: str | None = None     # e.g. "segment_1_2", or None if unknown
    image: Image.Image | None = field(repr=False, default=None)

    def summary(self) -> str:
        ts = time.strftime("%H:%M:%S", time.localtime(self.timestamp))
        ms = int((self.timestamp % 1) * 1000)
        w, h = self.region_size
        seg = f" seg={self.segment_name}" if self.segment_name else ""
        return (
            f"[{ts}.{ms:03d}] "
            f"gaze=({self.gaze_x:4d}, {self.gaze_y:4d}) "
            f"conf={self.confidence:<12s} "
            f"capture={w}x{h}px"
            f"{seg}"
        )


class GazeLogger:
    """
    Rolling in-memory log.  Entries older than MAX_LOG_ENTRIES are silently
    dropped – no files are written to disk.
    """

    def __init__(self, max_entries: int = MAX_LOG_ENTRIES):
        self._log: Deque[LogEntry] = deque(maxlen=max_entries)

    # ------------------------------------------------------------------
    def log(
        self,
        gaze_x: int,
        gaze_y: int,
        confidence: bet.TrackingConfidence,
        image: Image.Image | None,
        segment_name: str | None = None,
    ) -> LogEntry:
        """Create a log entry, print it, and store it (dropping the oldest if full)."""
        entry = LogEntry(
            timestamp=time.time(),
            gaze_x=gaze_x,
            gaze_y=gaze_y,
            confidence=confidence.name,
            region_size=image.size if image is not None else (0, 0),
            segment_name=segment_name,
            image=image,
        )
        self._log.append(entry)
        print(entry.summary())
        return entry

    # ------------------------------------------------------------------
    def consume_all(self) -> list[LogEntry]:
        """
        Return all current entries and clear the buffer.
        Use this when you want to 'use up' the log (e.g., pass to AI later).
        """
        entries = list(self._log)
        self._log.clear()
        return entries

    def __len__(self) -> int:
        return len(self._log)
