"""
nsfw_consumer.py
NSFW safety consumer using the AdamCodd/vit-base-nsfw-detector ViT model.

Classifies the currently focused segment image that is placed on the queue
by AICaptureThread.  A shell command is fired when the segment's score
exceeds the confidence threshold and is not on cooldown.

Per-segment result caching
--------------------------
Every classification result (SFW or NSFW) is stored in a per-segment cache
keyed by (row, col).  Before a segment is re-queued for scanning,
AICaptureThread compares the freshly captured image against the cached image:

  * If >= 70 % of the pixels share the same colour the frame is trivially
    background / desktop and is auto-flagged SFW without touching the model.
  * If <= PIXEL_DIFF_THRESHOLD (default 60 %) of pixels have changed
    significantly the cache is still considered valid - the segment is NOT
    re-queued and the stored result is used.
  * Otherwise the segment is re-queued, the model runs, and the cache is
    updated.

When gaze returns to a segment that already has a valid cache entry
get_segment_result() returns it immediately with no model invocation.

The model (~350 MB) is downloaded from HuggingFace on first use and cached
locally.  All subsequent runs are fully offline.
"""

from __future__ import annotations

import queue
import subprocess
import threading
from typing import Any, Callable

import numpy as np
from PIL import Image

from .segment_map import Segment

# Thumbnail size used for pixel comparisons (speed/accuracy trade-off).
_THUMB = (64, 64)

# A pixel is "different" if its mean-absolute per-channel deviation exceeds
# this value (0-255 scale).
_PER_PIXEL_DIFF_SENSITIVITY = 15


class NSFWConsumer:
    """
    Pulls (PIL segment image, Segment) pairs from `ai_queue`, runs NSFW
    classification on the segment image, and fires a shell command when
    the score exceeds the threshold and the segment is not on cooldown.

    Per-segment cache
    -----------------
    _cache : dict[(row, col) -> (is_nsfw, label, score, np.ndarray)]
        np.ndarray is the 64x64 RGB thumbnail of the image that was used for
        the last successful classification of that segment.

    Parameters
    ----------
    ai_queue   : queue.Queue  - source of (image, segment) tuples
    model      : str          - HuggingFace model identifier
    labels     : list[str]    - label names that count as NSFW
    threshold  : float        - minimum confidence to trigger (0.0-1.0)
    command    : str          - shell command to run on detection (blank = log only)
    mono_colour_threshold : float
        Fraction of pixels that must share the same colour for the auto-SFW
        fast path to kick in (default 0.70).
    pixel_diff_threshold : float
        Fraction of pixels that must have changed before the cache is
        considered stale and a re-scan is triggered (default 0.60).
    """

    def __init__(
        self,
        ai_queue:   queue.Queue,
        model:      str,
        labels:     list[str],
        threshold:  float,
        command:    str,
        mono_colour_threshold: float = 0.70,
        pixel_diff_threshold:  float = 0.60,
    ) -> None:
        self._queue      = ai_queue
        self._model      = model
        self._labels     = labels
        self._threshold  = threshold
        self._command    = command

        self._mono_colour_threshold = mono_colour_threshold
        self._pixel_diff_threshold  = pixel_diff_threshold

        self._pipeline: Any = None                # lazy-loaded on first use

        self._result_lock = threading.Lock()
        # Per-segment cache: (row, col) -> (is_nsfw, label, score, thumb_ndarray)
        self._cache: dict[tuple[int, int], tuple[bool, str, float, np.ndarray]] = {}
        # Most-recent classification across ALL segments (backwards compat)
        self._last_result: tuple[bool, str, float] | None = None

        # State-change callbacks for external integrations (e.g. buttplug)
        self._on_nsfw_cb: Callable[[], None] | None = None
        self._on_sfw_cb:  Callable[[], None] | None = None
        self._nsfw_active: bool = False   # True while content is classified NSFW

    # ------------------------------------------------------------------
    # Callback API
    # ------------------------------------------------------------------

    def set_callbacks(
        self,
        on_nsfw: Callable[[], None] | None = None,
        on_sfw:  Callable[[], None] | None = None,
    ) -> None:
        """
        Register callables that are invoked on NSFW/SFW state transitions.

        on_nsfw() is called once when content flips from SFW -> NSFW.
        on_sfw()  is called once when content flips from NSFW -> SFW.

        Both are invoked on a short-lived daemon thread so they must be
        thread-safe but are free to block (e.g. to start script playback).
        """
        self._on_nsfw_cb = on_nsfw
        self._on_sfw_cb  = on_sfw

    def _fire_callback(self, cb: Callable[[], None]) -> None:
        """Run a callback on a daemon thread so it never blocks the consumer."""
        threading.Thread(target=cb, daemon=True, name="NSFWCallback").start()

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_thumb(img: Image.Image) -> np.ndarray:
        """Return a 64x64 RGB uint8 numpy array."""
        return np.array(img.resize(_THUMB).convert("RGB"), dtype=np.uint8)

    @staticmethod
    def is_mono_colour(img: Image.Image, threshold: float = 0.70) -> bool:
        """
        Return True if *threshold* fraction (default 70 %) of the image's
        pixels share the same colour.

        Uses a 64x64 thumbnail for speed.
        """
        arr = NSFWConsumer._to_thumb(img)
        flat = arr.reshape(-1, 3)
        # Pack RGB into a single uint32 for fast unique-counting.
        packed = (flat[:, 0].astype(np.uint32) << 16
                  | flat[:, 1].astype(np.uint32) << 8
                  | flat[:, 2].astype(np.uint32))
        _, counts = np.unique(packed, return_counts=True)
        return counts.max() / len(packed) >= threshold

    # ------------------------------------------------------------------
    # Cache API (called from AICaptureThread and from run())
    # ------------------------------------------------------------------

    def is_cache_valid(self, seg: Segment, new_img: Image.Image) -> bool:
        """
        Return True if the cached result for *seg* is still valid, i.e.
        the new image has NOT changed by more than pixel_diff_threshold.

        Returns False if there is no cache entry for this segment.
        """
        with self._result_lock:
            cached = self._cache.get((seg.row, seg.col))
        if cached is None:
            return False
        _, _, _, cached_arr = cached
        new_arr = self._to_thumb(new_img)
        # Fraction of pixels whose mean-absolute per-channel deviation
        # exceeds the sensitivity level.
        diff_frac = float(
            np.mean(
                np.mean(np.abs(new_arr.astype(np.int32) - cached_arr.astype(np.int32)), axis=2)
                > _PER_PIXEL_DIFF_SENSITIVITY
            )
        )
        return diff_frac <= self._pixel_diff_threshold

    def cache_sfw(self, seg: Segment, img: Image.Image) -> None:
        """
        Store a fast-path SFW result for *seg* without running the model.
        Called by AICaptureThread when the mono-colour heuristic fires.
        """
        arr = self._to_thumb(img)
        with self._result_lock:
            self._cache[(seg.row, seg.col)] = (False, "sfw", 1.0, arr)
            self._last_result = (False, "sfw", 1.0)
        print(f"[nsfw] {seg.name} - auto-SFW (>={self._mono_colour_threshold:.0%} mono-colour)")
        # State transition: NSFW -> SFW
        # NOTE: on_sfw_cb is NOT fired here — the main loop applies a dwell
        # delay before pausing playback, so rapid SFW flickers don't interrupt.
        if self._nsfw_active:
            self._nsfw_active = False

    def get_segment_result(self, seg: Segment) -> tuple[bool, str, float] | None:
        """
        Thread-safe read of the cached classification for *seg*.

        Returns the stored (is_nsfw, label, score) or None if this segment
        has never been classified.

        Note: while a re-scan is in progress the *previous* cached result is
        returned unchanged.  The cache is only updated once the model has
        finished - so the displayed badge never flickers to "pending" during
        a background re-scan.
        """
        with self._result_lock:
            cached = self._cache.get((seg.row, seg.col))
        if cached is None:
            return None
        is_nsfw, label, score, _ = cached
        return (is_nsfw, label, score)

    def get_last_result(self) -> tuple[bool, str, float] | None:
        """
        Thread-safe read of the most recent classification result
        (any segment).  Kept for backwards compatibility.
        """
        with self._result_lock:
            return self._last_result

    @property
    def nsfw_active(self) -> bool:
        """
        Thread-safe read of the current NSFW/SFW state.

        True  → content is currently classified NSFW.
        False → content is SFW (or not yet classified).

        Used by the main loop to apply a dwell delay before pausing playback
        on SFW transitions, instead of reacting immediately.
        """
        return self._nsfw_active

    # ------------------------------------------------------------------
    def _load_pipeline(self) -> None:
        """Load the HuggingFace pipeline (no-op after first call)."""
        if self._pipeline is not None:
            return
        print(f"[nsfw] Loading model {self._model!r} (first run may download ~350 MB) ...")
        from transformers import pipeline as hf_pipeline  # deferred: keeps startup fast
        self._pipeline = hf_pipeline(
            "image-classification",
            model=self._model,
            device=-1,   # CPU; set to 0 for the first CUDA GPU
        )
        print(
            f"[nsfw] Model ready - "
            f"labels={self._labels}  threshold={self._threshold}"
        )

    # ------------------------------------------------------------------
    def _classify(self, img: Image.Image) -> tuple[bool, str, float]:
        """
        Run the pipeline on a PIL image.

        Returns
        -------
        (is_nsfw, top_nsfw_label, top_nsfw_score)
        """
        results = self._pipeline(img)  # list of {"label": str, "score": float}
        for item in sorted(results, key=lambda x: x["score"], reverse=True):
            if item["label"] in self._labels and item["score"] >= self._threshold:
                return True, item["label"], item["score"]
        return False, "", 0.0

    # ------------------------------------------------------------------
    def run(self) -> None:
        """
        Blocking loop - run this inside a daemon thread.

        Stale-result guarantee
        ~~~~~~~~~~~~~~~~~~~~~~
        The cache entry for a segment is only written *after* classification
        completes.  While the model is running the previous cached result
        (SFW or NSFW) remains visible to get_segment_result().  This means
        the debug badge never flickers back to "pending" during a re-scan.
        """
        while True:
            try:
                grid_img, seg_img, seg = self._queue.get(timeout=2)
            except queue.Empty:
                continue
            except Exception as exc:
                print(f"[nsfw] Queue error: {exc}")
                continue

            try:
                self._load_pipeline()

                # Classify the single segment first, then the 3×3 grid.
                # Either one returning NSFW is sufficient to flag the segment.
                is_nsfw_seg,  label_seg,  score_seg  = self._classify(seg_img)
                is_nsfw_grid, label_grid, score_grid = self._classify(grid_img)

                is_nsfw = is_nsfw_seg or is_nsfw_grid

                # Pick the label/score that triggered NSFW; if neither did,
                # use the grid result as the authoritative SFW reading.
                if is_nsfw_seg and (not is_nsfw_grid or score_seg >= score_grid):
                    label, score, source = label_seg,  score_seg,  "segment"
                elif is_nsfw_grid:
                    label, score, source = label_grid, score_grid, "grid"
                else:
                    # Both SFW — report whichever had the higher NSFW score
                    # (useful for debug visibility).
                    if score_seg >= score_grid:
                        label, score, source = label_seg,  score_seg,  "segment"
                    else:
                        label, score, source = label_grid, score_grid, "grid"

                # Build thumbnail from the grid image (used for cache-valid
                # comparison by AICaptureThread) and write the new result.
                # The previous cached value is replaced only at this point, so
                # the displayed result was stable throughout the scan.
                arr = self._to_thumb(grid_img)
                with self._result_lock:
                    self._cache[(seg.row, seg.col)] = (is_nsfw, label, score, arr)
                    self._last_result = (is_nsfw, label, score)

                if not is_nsfw:
                    # print(f"[nsfw] {seg.name} - clear (cached)")
                    # State transition: NSFW -> SFW
                    # NOTE: on_sfw_cb is NOT fired here — the main loop applies
                    # a dwell delay before pausing playback, so rapid SFW
                    # flickers don't interrupt the session.
                    if self._nsfw_active:
                        self._nsfw_active = False
                    continue

                print(
                    f"[nsfw] WARNING  {seg.name}  source={source!r}  "
                    f"label={label!r}  score={score:.2f}"
                    f"  (seg={score_seg:.2f}  grid={score_grid:.2f})"
                )

                # State transition: SFW -> NSFW
                # Fire the shell command and callbacks exactly once per transition.
                if not self._nsfw_active:
                    self._nsfw_active = True
                    cb = self._on_nsfw_cb
                    if cb is not None:
                        self._fire_callback(cb)

                    if self._command:
                        print(f"[nsfw]    firing: {self._command}")
                        try:
                            subprocess.Popen(self._command, shell=True)
                        except Exception as exc:
                            print(f"[nsfw]    command error: {exc}")
                    else:
                        print("[nsfw]    (no command configured - set [nsfw] command in config.ini)")

            except Exception as exc:
                print(f"[nsfw] Detection error: {exc}")
