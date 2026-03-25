"""
main.py
Entry point for the Beam Eye Tracker gaze capture application.

Run:
    b

Press Ctrl+C to stop.

Behaviour
---------
1.  The screen is divided into a configurable N×M grid (see config.ini).
2.  The gaze point is mapped to whichever segment it falls in.
3.  If the gaze dwells in one segment for `dwell_ms` ms the segment's
    shell command (if any) fires  →  dwell_tracker.py
4.  A background thread captures the focused segment every
    `capture_interval_s` seconds and queues it for NSFW analysis
    →  ai_capture_thread.py  /  nsfw_consumer.py
5.  The debug preview window draws the segment grid with live overlays
    →  debug_preview.py
"""

from __future__ import annotations

import configparser
import os
import queue
import signal
import threading

import eyeware.beam_eye_tracker as bet

from gaze_tracker import GazeTracker, SCREEN_WIDTH, SCREEN_HEIGHT
from screen_capture import capture_fullscreen, capture_segment
from segment_map import SegmentMap
from logger import GazeLogger
from dwell_tracker import DwellTracker
from nsfw_consumer import NSFWConsumer
from ai_capture_thread import AICaptureThread
from debug_preview import DebugPreview

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.ini")
_cfg = configparser.ConfigParser(inline_comment_prefixes=(";",))
_cfg.read(_CONFIG_PATH)

POLL_TIMEOUT_MS:    int   = 42
DEBUG_PREVIEW:      bool  = True

DWELL_MS:           int   = _cfg.getint(  "dwell", "dwell_ms",          fallback=400)
COOLDOWN_MS:        int   = _cfg.getint(  "dwell", "cooldown_ms",        fallback=1500)
CAPTURE_INTERVAL_S: float = _cfg.getfloat("ai",    "capture_interval_s", fallback=1.0)

NSFW_MODEL:               str   = _cfg.get(     "nsfw", "model",                  fallback="AdamCodd/vit-base-nsfw-detector").strip()
NSFW_LABELS:              list  = [l.strip() for l in _cfg.get("nsfw", "nsfw_labels", fallback="nsfw").split(",")]
NSFW_THRESHOLD:           float = _cfg.getfloat("nsfw", "threshold",              fallback=0.65)
NSFW_COOLDOWN_S:          float = _cfg.getfloat("nsfw", "cooldown_s",             fallback=30.0)
NSFW_COMMAND:             str   = _cfg.get(     "nsfw", "command",                fallback="").strip()
NSFW_MONO_COLOUR_THRESH:  float = _cfg.getfloat("nsfw", "mono_colour_threshold",  fallback=0.70)
NSFW_PIXEL_DIFF_THRESH:   float = _cfg.getfloat("nsfw", "pixel_diff_threshold",   fallback=0.60)

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

_running = True


def _handle_stop(sig, frame):  # noqa: ANN001
    global _running
    print("\n[main] Shutting down…")
    _running = False


signal.signal(signal.SIGINT, _handle_stop)
if hasattr(signal, "SIGTERM"):
    signal.signal(signal.SIGTERM, _handle_stop)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    global _running

    logger  = GazeLogger()
    seg_map = SegmentMap(SCREEN_WIDTH, SCREEN_HEIGHT)
    dwell   = DwellTracker(dwell_ms=DWELL_MS, cooldown_ms=COOLDOWN_MS)

    print(f"[main] Screen: {SCREEN_WIDTH}×{SCREEN_HEIGHT}  "
          f"Grid: {seg_map.columns}×{seg_map.rows}  "
          f"Dwell: {DWELL_MS}ms  Cooldown: {COOLDOWN_MS}ms")
    print(f"[main] NSFW model: {NSFW_MODEL}  labels: {NSFW_LABELS}")
    for seg in seg_map.all_segments():
        cmd_str = f"  → {seg.command}" if seg.command else "  (no command)"
        print(f"        {seg.name}  {seg.rect}{cmd_str}")
    print("-" * 60)

    ai_queue: queue.Queue                   = queue.Queue(maxsize=1)
    ai_capture_thread: AICaptureThread | None = None
    nsfw_consumer: NSFWConsumer | None        = None

    if CAPTURE_INTERVAL_S > 0:
        # NSFWConsumer is created first so it can be passed into
        # AICaptureThread for the cache-aware fast-paths.
        nsfw_consumer = NSFWConsumer(
            ai_queue=ai_queue,
            model=NSFW_MODEL,
            labels=NSFW_LABELS,
            threshold=NSFW_THRESHOLD,
            cooldown_s=NSFW_COOLDOWN_S,
            command=NSFW_COMMAND,
            mono_colour_threshold=NSFW_MONO_COLOUR_THRESH,
            pixel_diff_threshold=NSFW_PIXEL_DIFF_THRESH,
        )
        threading.Thread(
            target=nsfw_consumer.run, daemon=True, name="NSFWConsumer"
        ).start()

        ai_capture_thread = AICaptureThread(
            interval_s=CAPTURE_INTERVAL_S,
            ai_queue=ai_queue,
            nsfw_consumer=nsfw_consumer,
        )
        ai_capture_thread.start()

    preview: DebugPreview | None = DebugPreview(seg_map) if DEBUG_PREVIEW else None

    print("[main] Initialising Beam Eye Tracker API…")
    with GazeTracker() as tracker:
        tracker._api.attempt_starting_the_beam_eye_tracker()
        print("[main] Waiting for tracking data  (Ctrl+C to quit)…")
        print("-" * 60)

        last_seg_image     = None
        last_preview_image = None
        last_active_seg    = seg_map.get_segment(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)

        while _running:
            result = tracker.poll(timeout_ms=POLL_TIMEOUT_MS)

            if result is None:
                status = tracker.status()
                if status != bet.TrackingDataReceptionStatus.RECEIVING_TRACKING_DATA:
                    print(f"[main] Status: {status.name}", end="\r")
                if preview is not None and last_preview_image is not None:
                    nsfw_res = nsfw_consumer.get_segment_result(last_active_seg) if nsfw_consumer else None
                    preview.update(last_preview_image, last_active_seg,
                                   dwell.dwell_progress(), dwell.consume_fired(),
                                   nsfw_result=nsfw_res)
                continue

            gaze_x, gaze_y, confidence = result
            active_seg      = seg_map.get_segment(gaze_x, gaze_y)
            last_active_seg = active_seg

            # Keep the AI capture thread focused on the current segment.
            if ai_capture_thread is not None:
                ai_capture_thread.set_active_segment(active_seg)

            dwell.update(active_seg)
            fired_seg = dwell.consume_fired()

            seg_image      = capture_segment(active_seg)
            last_seg_image = seg_image
            logger.log(gaze_x=gaze_x, gaze_y=gaze_y, confidence=confidence,
                       image=seg_image, segment_name=active_seg.name)

            if preview is not None:
                preview_image      = capture_fullscreen()
                last_preview_image = preview_image
                nsfw_res = nsfw_consumer.get_segment_result(active_seg) if nsfw_consumer else None
                preview.update(preview_image, active_seg,
                               dwell.dwell_progress(), fired_seg,
                               nsfw_result=nsfw_res)

    if ai_capture_thread is not None:
        ai_capture_thread.stop()
    print("[main] Done.")


if __name__ == "__main__":
    main()
