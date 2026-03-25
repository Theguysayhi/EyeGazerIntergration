"""
main.py
Entry point for the Beam Eye Tracker gaze capture application.

Run:
    py -3.10 main.py

Press Ctrl+C to stop.

Behaviour
---------
1.  The screen is divided into a configurable N×M grid (see config.ini).
2.  The gaze point is mapped to whichever segment it falls in.
3.  If the gaze dwells in one segment for `dwell_ms` ms a dwell event
    is signalled  →  dwell_tracker.py
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
import time

import eyeware.beam_eye_tracker as bet

from core.gaze_tracker import GazeTracker, SCREEN_WIDTH, SCREEN_HEIGHT
from core.screen_capture import capture_fullscreen, capture_segment
from core.segment_map import SegmentMap
from core.logger import GazeLogger
from core.dwell_tracker import DwellTracker
from core.nsfw_consumer import NSFWConsumer
from core.ai_capture_thread import AICaptureThread
from core.debug_preview import DebugPreview
from core.buttplug.controller import ButtplugController

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
NSFW_COMMAND:             str   = _cfg.get(     "nsfw", "command",                fallback="").strip()
NSFW_MONO_COLOUR_THRESH:  float = _cfg.getfloat("nsfw", "mono_colour_threshold",  fallback=0.70)
NSFW_PIXEL_DIFF_THRESH:   float = _cfg.getfloat("nsfw", "pixel_diff_threshold",   fallback=0.60)

BUTTPLUG_ENABLED:      bool  = _cfg.getboolean("buttplug", "enabled",       fallback=True)
BUTTPLUG_SERVER_URL:   str   = _cfg.get(       "buttplug", "server_url",    fallback="ws://127.0.0.1:12345").strip()
BUTTPLUG_SCRIPTS_DIR:  str   = os.path.join(
    os.path.dirname(_CONFIG_PATH),
    _cfg.get("buttplug", "scripts_dir", fallback="scripts").strip(),
)
BUTTPLUG_STARTER_LOOPS: int  = _cfg.getint(    "buttplug", "starter_loops", fallback=1)

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
            segment_map=seg_map,
        )
        ai_capture_thread.start()

    # -------------------------------------------------------------------
    # Buttplug / Intiface Central controller
    # -------------------------------------------------------------------
    buttplug_ctrl: ButtplugController | None = None
    if BUTTPLUG_ENABLED:
        print(f"[main] Buttplug enabled — server: {BUTTPLUG_SERVER_URL}  "
              f"scripts: {BUTTPLUG_SCRIPTS_DIR}")
        buttplug_ctrl = ButtplugController(
            server_url=BUTTPLUG_SERVER_URL,
            scripts_dir=BUTTPLUG_SCRIPTS_DIR,
            starter_loops=BUTTPLUG_STARTER_LOOPS,
        )
        if nsfw_consumer is not None:
            nsfw_consumer.set_callbacks(
                on_nsfw=buttplug_ctrl.on_nsfw,
                on_sfw=buttplug_ctrl.on_sfw,
            )
        buttplug_ctrl.start()
    else:
        print("[main] Buttplug disabled (set [buttplug] enabled = true to enable)")

    preview: DebugPreview | None = DebugPreview(seg_map) if DEBUG_PREVIEW else None

    print("[main] Initialising Beam Eye Tracker API…")
    with GazeTracker() as tracker:
        tracker._api.attempt_starting_the_beam_eye_tracker()
        print("[main] Waiting for tracking data  (Ctrl+C to quit)…")
        print("-" * 60)

        last_seg_image     = None
        last_preview_image = None
        last_active_seg    = seg_map.get_segment(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)

        # -------------------------------------------------------------------
        # SFW dwell state — used to delay the pause command so that brief
        # SFW flickers (e.g. tab-switching) don't interrupt playback.
        # Dwell_ms controls how long content must stay SFW before we pause.
        # NSFW is handled immediately by the on_nsfw_cb in NSFWConsumer.
        # -------------------------------------------------------------------
        _prev_nsfw_active: bool       = False
        _sfw_since:        float | None = None   # monotonic timestamp
        _sfw_pause_fired:  bool       = False    # fire only once per SFW window

        while _running:
            result = tracker.poll(timeout_ms=POLL_TIMEOUT_MS)

            # -----------------------------------------------------------------
            # SFW dwell check — runs every poll cycle regardless of gaze data.
            # When content transitions NSFW→SFW we start a timer.  Only after
            # DWELL_MS of sustained SFW do we call on_sfw() to pause playback.
            # If content flips back to NSFW the timer resets automatically.
            # -----------------------------------------------------------------
            if nsfw_consumer is not None and buttplug_ctrl is not None:
                currently_nsfw = nsfw_consumer.nsfw_active
                if currently_nsfw != _prev_nsfw_active:
                    if currently_nsfw:
                        # Flipped to NSFW — reset SFW dwell timer.
                        # on_nsfw_cb already fired immediately inside NSFWConsumer.
                        _sfw_since       = None
                        _sfw_pause_fired = False
                    else:
                        # Flipped to SFW — start the dwell timer.
                        _sfw_since       = time.monotonic()
                        _sfw_pause_fired = False
                    _prev_nsfw_active = currently_nsfw

                # Pause once DWELL_MS of sustained SFW has elapsed.
                if (
                    not currently_nsfw
                    and _sfw_since is not None
                    and not _sfw_pause_fired
                    and (time.monotonic() - _sfw_since) * 1000 >= DWELL_MS
                ):
                    _sfw_pause_fired = True
                    print(f"[main] SFW dwell {DWELL_MS} ms reached — pausing playback")
                    buttplug_ctrl.on_sfw()

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

    if buttplug_ctrl is not None:
        buttplug_ctrl.stop()
    if ai_capture_thread is not None:
        ai_capture_thread.stop()
    print("[main] Done.")


if __name__ == "__main__":
    main()
