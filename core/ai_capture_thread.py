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

from .segment_map import Segment, SegmentMap
from .screen_capture import capture_segment, capture_segments_expanded

if TYPE_CHECKING:
    from .nsfw_consumer import NSFWConsumer


class AICaptureThread(threading.Thread):
    """
    Captures the currently focused segment (plus its 3×3 neighbourhood) every
    `interval_s` seconds and places ``(expanded_image, active_seg)`` on
    ``ai_queue`` for the NSFWConsumer.

    The captured image covers the active segment AND all immediately surrounding
    segments (up to a 3×3 block) so the model sees richer context.  The cache
    key is still keyed to the **centre segment** only.

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
    segment_map   : SegmentMap | None
        The active segment map used to look up neighbours.  When supplied
        the capture is expanded to the 3×3 neighbourhood around the active
        segment.  When None only the active segment itself is captured
        (legacy behaviour).
    """

    def __init__(
        self,
        interval_s: float,
        ai_queue: queue.Queue,
        nsfw_consumer: "NSFWConsumer | None" = None,
        segment_map: "SegmentMap | None" = None,
    ) -> None:
        super().__init__(daemon=True, name="AICaptureThread")
        self.interval_s     = interval_s
        self._queue         = ai_queue
        self._nsfw_consumer = nsfw_consumer
        self._segment_map   = segment_map
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
        nc  = self._nsfw_consumer  # local alias for speed
        sm  = self._segment_map    # local alias for speed

        while not self._stop_event.is_set():
            time.sleep(self.interval_s)

            with self._seg_lock:
                seg = self._active_seg

            if seg is None:
                continue

            try:
                # 1. Capture the single active segment first (smaller, faster).
                seg_img = capture_segment(seg)

                # 2. Capture the active segment plus its surrounding 3×3
                #    neighbourhood as a single expanded screenshot.
                #    Falls back to the centre segment alone if no map is set.
                if sm is not None:
                    neighbours = sm.get_neighbours(seg, radius=1)
                    grid_img = capture_segments_expanded(neighbours)
                else:
                    grid_img = capture_segments_expanded([seg])
            except Exception as exc:
                print(f"[ai] Capture error: {exc}")
                continue

            # ----------------------------------------------------------
            # Fast-path 1: mono-colour → auto-SFW, no queue
            # (checked against the grid image for broader coverage)
            # ----------------------------------------------------------
            if nc is not None and nc.is_mono_colour(grid_img, nc._mono_colour_threshold):
                nc.cache_sfw(seg, grid_img)
                continue

            # ----------------------------------------------------------
            # Fast-path 2: cache still valid → skip re-scan
            # The existing cached result (SFW or NSFW) remains in place
            # and will be returned by get_segment_result() as-is.
            # ----------------------------------------------------------
            if nc is not None and nc.is_cache_valid(seg, grid_img):
                # Nothing to do – cached result is current.
                continue

            # ----------------------------------------------------------
            # Full scan: drop any stale queued frame then enqueue this one.
            # Both the single-segment image and the 3×3 grid image are sent
            # so the consumer can classify each independently.
            # The cache entry is NOT cleared here; the previous verdict
            # remains visible until NSFWConsumer writes the new result.
            # ----------------------------------------------------------
            try:
                self._queue.get_nowait()   # drop stale frame
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait((grid_img, seg_img, seg))
            except queue.Full:
                pass   # consumer hasn't processed the previous frame yet
