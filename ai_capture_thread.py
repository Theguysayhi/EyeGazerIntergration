"""
ai_capture_thread.py
Background thread that periodically captures the currently focused gaze
segment and queues it for NSFW analysis.

Cache-aware capture logic
--------------------------
Before placing a frame on the AI queue this thread applies two fast-path
checks (both require an NSFWConsumer reference to be passed in):

  1. Mono-colour fast path
     If ≥ mono_colour_threshold (default 70 %) of the segment's pixels share
     the same colour the segment is trivially a blank/desktop area.
     NSFWConsumer.cache_sfw() is called directly and the frame is NOT queued
     – the model is never invoked.

  2. Cache-valid check
     If a previous scan result already exists for this segment AND the new
     capture differs from the cached image by ≤ pixel_diff_threshold (default
     60 % of pixels changed significantly), the cache is still considered
     fresh.  The frame is skipped; get_segment_result() will return the
     existing result on next query.

     IMPORTANT: while a re-scan is in flight the cache entry is NOT cleared,
     so the previous SFW/NSFW verdict remains visible until the new result
     overwrites it.

  3. Full scan
     If neither fast-path applies the image is queued for the NSFWConsumer
     model pipeline.  After the scan completes NSFWConsumer updates the cache
     atomically.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import TYPE_CHECKING

from segment_map import Segment
from screen_capture import capture_segment

if TYPE_CHECKING:
    from nsfw_consumer import NSFWConsumer


class AICaptureThread(threading.Thread):
    """
    Captures the currently focused segment every `interval_s` seconds and
    places ``(segment_image, active_seg)`` on ``ai_queue`` for the
    NSFWConsumer.

    Call :meth:`set_active_segment` from the main loop whenever the gaze
    moves to a new segment.

    Parameters
    ----------
    interval_s    : float           – seconds between captures
    ai_queue      : queue.Queue     – destination queue (maxsize=1 recommended)
    nsfw_consumer : NSFWConsumer | None
        Optional reference to the consumer.  When supplied, the mono-colour
        and cache-validity fast-paths are active.  When None the thread
        behaves exactly as before (always queues every frame).
    """

    def __init__(
        self,
        interval_s: float,
        ai_queue: queue.Queue,
        nsfw_consumer: "NSFWConsumer | None" = None,
    ) -> None:
        super().__init__(daemon=True, name="AICaptureThread")
        self.interval_s     = interval_s
        self._queue         = ai_queue
        self._nsfw_consumer = nsfw_consumer
        self._stop_event    = threading.Event()
        self._seg_lock      = threading.Lock()
        self._active_seg: Segment | None = None

    # ------------------------------------------------------------------
    def set_active_segment(self, seg: Segment) -> None:
        """Thread-safe update of the segment to scan next."""
        with self._seg_lock:
            self._active_seg = seg

    # ------------------------------------------------------------------
    def stop(self) -> None:
        self._stop_event.set()

    # ------------------------------------------------------------------
    def run(self) -> None:
        print(f"[ai] Segment capture started (every {self.interval_s}s)")
        nc = self._nsfw_consumer  # local alias for speed

        while not self._stop_event.is_set():
            time.sleep(self.interval_s)

            with self._seg_lock:
                seg = self._active_seg

            if seg is None:
                continue

            try:
                img = capture_segment(seg)
            except Exception as exc:
                print(f"[ai] Capture error: {exc}")
                continue

            # ----------------------------------------------------------
            # Fast-path 1: mono-colour → auto-SFW, no queue
            # ----------------------------------------------------------
            if nc is not None and nc.is_mono_colour(img, nc._mono_colour_threshold):
                nc.cache_sfw(seg, img)
                continue

            # ----------------------------------------------------------
            # Fast-path 2: cache still valid → skip re-scan
            # The existing cached result (SFW or NSFW) remains in place
            # and will be returned by get_segment_result() as-is.
            # ----------------------------------------------------------
            if nc is not None and nc.is_cache_valid(seg, img):
                # Nothing to do – cached result is current.
                continue

            # ----------------------------------------------------------
            # Full scan: drop any stale queued frame then enqueue this one.
            # The cache entry is NOT cleared here; the previous verdict
            # remains visible until NSFWConsumer writes the new result.
            # ----------------------------------------------------------
            try:
                self._queue.get_nowait()   # drop stale frame
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait((img, seg))
            except queue.Full:
                pass   # consumer hasn't processed the previous frame yet
