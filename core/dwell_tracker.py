"""
dwell_tracker.py
Tracks gaze dwell time per screen segment and signals dwell events.
"""

from __future__ import annotations

import time

from .segment_map import Segment


class DwellTracker:
    """
    Tracks how long the gaze has been inside the same segment and signals
    when the dwell threshold is reached.

    After firing, the segment enters a cooldown period during which it cannot
    fire again.
    """

    def __init__(self, dwell_ms: int = 400, cooldown_ms: int = 1500) -> None:
        self.dwell_ms    = dwell_ms
        self.cooldown_ms = cooldown_ms

        self._current_seg:    Segment | None = None
        self._dwell_start:    float          = 0.0      # time.monotonic()
        self._last_fired:     dict[tuple, float] = {}   # (row,col) → time.monotonic()
        self._just_fired_seg: Segment | None = None     # for preview flash

    # ------------------------------------------------------------------
    def update(self, segment: Segment) -> bool:
        """
        Notify the tracker of the current gaze segment.

        Returns True if a command was fired this call.
        """
        now   = time.monotonic()
        fired = False

        # Segment changed → reset dwell timer
        if self._current_seg != segment:
            self._current_seg = segment
            self._dwell_start = now
            return False

        # Still in the same segment – check dwell threshold
        elapsed_ms = (now - self._dwell_start) * 1000
        if elapsed_ms < self.dwell_ms:
            return False

        # Dwell threshold reached – check cooldown
        key  = (segment.row, segment.col)
        last = self._last_fired.get(key, 0.0)
        if (now - last) * 1000 < self.cooldown_ms:
            return False

        # Fire!
        self._last_fired[key] = now
        self._just_fired_seg  = segment
        fired = True

        return fired

    # ------------------------------------------------------------------
    def dwell_progress(self) -> float:
        """
        Return 0.0–1.0 representing how far through the dwell window the
        current gaze is.  1.0 means the command has just fired / is ready.
        """
        if self._current_seg is None:
            return 0.0
        elapsed_ms = (time.monotonic() - self._dwell_start) * 1000
        return min(elapsed_ms / self.dwell_ms, 1.0)

    def is_on_cooldown(self, segment: Segment) -> bool:
        key  = (segment.row, segment.col)
        last = self._last_fired.get(key, 0.0)
        return (time.monotonic() - last) * 1000 < self.cooldown_ms

    def consume_fired(self) -> Segment | None:
        """Return and clear the most-recently-fired segment (for preview flash)."""
        seg = self._just_fired_seg
        self._just_fired_seg = None
        return seg
