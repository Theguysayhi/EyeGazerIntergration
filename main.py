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
3.  If the gaze *dwells* in one segment for `dwell_ms` milliseconds the
    segment's configured shell command (if any) is fired.
4.  A background thread captures the current segment every
    `capture_interval_s` seconds and queues it for AI analysis (stub).
5.  The debug preview window draws the segment grid with live overlays.
"""

from __future__ import annotations

import configparser
import os
import queue
import signal
import subprocess
import threading
import time
import tkinter as tk

import eyeware.beam_eye_tracker as bet
from PIL import ImageDraw, ImageFont, ImageTk

from gaze_tracker import GazeTracker, SCREEN_WIDTH, SCREEN_HEIGHT
from screen_capture import capture_fullscreen, capture_segment
from segment_map import SegmentMap, Segment
from logger import GazeLogger

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.ini")
_cfg = configparser.ConfigParser(inline_comment_prefixes=(";",))
_cfg.read(_CONFIG_PATH)

POLL_TIMEOUT_MS: int  = 42          # ~24 fps tracker poll
DEBUG_PREVIEW:   bool = True

DWELL_MS:    int   = _cfg.getint(  "dwell", "dwell_ms",    fallback=400)
COOLDOWN_MS: int   = _cfg.getint(  "dwell", "cooldown_ms", fallback=1500)
CAPTURE_INTERVAL_S: float = _cfg.getfloat("ai", "capture_interval_s", fallback=1.0)

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
# Dwell tracker
# ---------------------------------------------------------------------------

class DwellTracker:
    """
    Tracks how long the gaze has been inside the same segment and fires the
    segment's command once the dwell threshold is reached.

    After firing, the segment enters a cooldown period during which it cannot
    fire again.
    """

    def __init__(self, dwell_ms: int = DWELL_MS, cooldown_ms: int = COOLDOWN_MS) -> None:
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
        now = time.monotonic()
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
        key = (segment.row, segment.col)
        last = self._last_fired.get(key, 0.0)
        if (now - last) * 1000 < self.cooldown_ms:
            return False

        # Fire!
        self._last_fired[key] = now
        self._just_fired_seg  = segment
        fired = True

        if segment.command:
            print(f"[dwell] Firing command for {segment.name}: {segment.command}")
            try:
                subprocess.Popen(segment.command, shell=True)
            except Exception as exc:
                print(f"[dwell] Command error: {exc}")
        else:
            print(f"[dwell] Dwell confirmed on {segment.name} (no command configured)")

        return fired

    # ------------------------------------------------------------------
    def dwell_progress(self) -> float:
        """
        Return a 0.0–1.0 float representing how far through the dwell window
        the current gaze is.  1.0 means the command has just fired / is ready.
        """
        if self._current_seg is None:
            return 0.0
        elapsed_ms = (time.monotonic() - self._dwell_start) * 1000
        return min(elapsed_ms / self.dwell_ms, 1.0)

    def is_on_cooldown(self, segment: Segment) -> bool:
        key = (segment.row, segment.col)
        last = self._last_fired.get(key, 0.0)
        return (time.monotonic() - last) * 1000 < self.cooldown_ms

    def consume_fired(self) -> Segment | None:
        """Return and clear the most-recently-fired segment (for preview flash)."""
        seg = self._just_fired_seg
        self._just_fired_seg = None
        return seg


# ---------------------------------------------------------------------------
# Periodic AI capture thread
# ---------------------------------------------------------------------------

class AICaptureThread(threading.Thread):
    """
    Background thread that captures the current active segment every
    `capture_interval_s` seconds and places it in an `ai_queue` for
    downstream AI processing (currently a stub).
    """

    def __init__(
        self,
        interval_s: float,
        current_segment_ref: list,   # mutable single-element list: [Segment | None]
        ai_queue: queue.Queue,
    ) -> None:
        super().__init__(daemon=True, name="AICaptureThread")
        self.interval_s          = interval_s
        self._segment_ref        = current_segment_ref
        self._queue              = ai_queue
        self._stop_event         = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        print(f"[ai] Periodic capture started (every {self.interval_s}s)")
        while not self._stop_event.is_set():
            time.sleep(self.interval_s)
            seg: Segment | None = self._segment_ref[0]
            if seg is None:
                continue
            try:
                img = capture_segment(seg)
                # Drop oldest frame if the consumer hasn't caught up
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                self._queue.put_nowait((seg, img))
                print(f"[ai] Captured {seg.name} ({img.width}×{img.height}px) → queue")
            except Exception as exc:
                print(f"[ai] Capture error: {exc}")


def _ai_consumer_stub(ai_queue: queue.Queue) -> None:
    """
    Runs in its own daemon thread.  Pulls (Segment, PIL.Image) pairs from
    the queue and passes them to the AI model.

    TODO: Replace the print stub with a real AI vision model call.
    """
    while True:
        try:
            seg, img = ai_queue.get(timeout=2)
            # ----------------------------------------------------------------
            # TODO: call your AI vision model here, e.g.:
            #   description = ai_model.describe(img)
            #   print(f"[ai] {seg.name}: {description}")
            # ----------------------------------------------------------------
            print(f"[ai] TODO – analyse {seg.name} image ({img.width}×{img.height}px)")
        except queue.Empty:
            pass
        except Exception as exc:
            print(f"[ai] Consumer error: {exc}")


# ---------------------------------------------------------------------------
# Debug preview window
# ---------------------------------------------------------------------------

# Scale factor for the debug preview (segment captures can be large)
PREVIEW_SCALE = 0.45   # render at 45 % of the original segment size


class DebugPreview:
    """
    Tkinter window that shows:
    - The latest segment capture (scaled)
    - A grid overlay with segment names
    - The currently-gazed segment highlighted in yellow
    - A dwell progress bar along the bottom of the active segment
    - A green flash when a command fires
    """

    def __init__(self, segment_map: SegmentMap) -> None:
        self._seg_map  = segment_map
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._thread   = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    def _run(self) -> None:
        self._root = tk.Tk()
        self._root.title("Gaze Capture – Segment Preview")
        self._root.resizable(False, False)
        self._label = tk.Label(self._root, bg="black")
        self._label.pack()
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._root.after(16, self._poll)
        self._root.mainloop()

    def _poll(self) -> None:
        try:
            payload = self._queue.get_nowait()
            pil_image, active_seg, dwell_progress, fired_seg = payload
            self._photo = ImageTk.PhotoImage(pil_image)
            self._label.config(image=self._photo)
        except queue.Empty:
            pass
        self._root.after(16, self._poll)

    def _on_close(self) -> None:
        global _running
        _running = False
        self._root.destroy()

    # ------------------------------------------------------------------
    def update(
        self,
        seg_image,          # PIL image of the current segment
        active_seg: Segment,
        dwell_progress: float,  # 0.0–1.0
        fired_seg: Segment | None,
    ) -> None:
        """Build the annotated preview image and push it to the Tk thread."""
        try:
            # Scale down the segment capture for display
            w = int(seg_image.width  * PREVIEW_SCALE)
            h = int(seg_image.height * PREVIEW_SCALE)
            display = seg_image.resize((w, h))

            draw = ImageDraw.Draw(display, "RGBA")

            seg_map = self._seg_map
            cols, rows = seg_map.columns, seg_map.rows

            # -- Grid lines --------------------------------------------------
            for c in range(1, cols):
                x = int(w * c / cols)
                draw.line([(x, 0), (x, h)], fill=(200, 200, 200, 160), width=1)
            for r in range(1, rows):
                y = int(h * r / rows)
                draw.line([(0, y), (w, y)], fill=(200, 200, 200, 160), width=1)

            # -- Segment labels ----------------------------------------------
            cell_w = w // cols
            cell_h = h // rows
            for seg in seg_map.all_segments():
                cx = int((seg.col + 0.5) * cell_w)
                cy = int((seg.row + 0.5) * cell_h)
                label = seg.name.replace("segment_", "")
                draw.text((cx - 10, cy - 6), label, fill=(200, 200, 200, 200))

            # -- Active segment highlight ------------------------------------
            ac = active_seg.col
            ar = active_seg.row
            ax0 = ac * cell_w
            ay0 = ar * cell_h
            ax1 = ax0 + cell_w
            ay1 = ay0 + cell_h

            # Yellow translucent fill
            draw.rectangle([(ax0, ay0), (ax1, ay1)], fill=(255, 220, 0, 55))
            # Yellow border
            draw.rectangle([(ax0, ay0), (ax1, ay1)], outline=(255, 220, 0, 220), width=2)

            # -- Dwell progress bar ------------------------------------------
            if dwell_progress < 1.0:
                bar_h = 5
                bar_w = int(cell_w * dwell_progress)
                draw.rectangle(
                    [(ax0, ay1 - bar_h), (ax0 + bar_w, ay1)],
                    fill=(255, 220, 0, 230),
                )

            # -- Fired flash (green overlay) ---------------------------------
            if fired_seg is not None:
                fc = fired_seg.col
                fr = fired_seg.row
                fx0 = fc * cell_w
                fy0 = fr * cell_h
                fx1 = fx0 + cell_w
                fy1 = fy0 + cell_h
                draw.rectangle([(fx0, fy0), (fx1, fy1)], fill=(0, 255, 80, 100))
                draw.rectangle([(fx0, fy0), (fx1, fy1)], outline=(0, 255, 80, 255), width=3)

            # -- Info label --------------------------------------------------
            info = (
                f"seg={active_seg.name}  "
                f"dwell={int(dwell_progress * 100)}%  "
                f"{SCREEN_WIDTH}×{SCREEN_HEIGHT}"
            )
            draw.rectangle([(0, 0), (len(info) * 7 + 6, 18)], fill=(0, 0, 0, 180))
            draw.text((4, 2), info, fill=(200, 255, 200))

            payload = (display, active_seg, dwell_progress, fired_seg)
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(payload)

        except Exception:
            pass  # silently ignore if window was closed


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
    print(f"[main] AI capture interval: {CAPTURE_INTERVAL_S}s")
    print(f"[main] Segments:")
    for seg in seg_map.all_segments():
        cmd_str = f"  → {seg.command}" if seg.command else "  (no command)"
        print(f"        {seg.name}  {seg.rect}{cmd_str}")
    print("-" * 60)

    # Shared mutable reference so the AI thread always sees the latest segment
    current_segment_ref: list[Segment | None] = [None]

    # AI capture queue and threads
    ai_queue: queue.Queue = queue.Queue(maxsize=1)
    ai_capture_thread: AICaptureThread | None = None
    if CAPTURE_INTERVAL_S > 0:
        ai_capture_thread = AICaptureThread(
            interval_s=CAPTURE_INTERVAL_S,
            current_segment_ref=current_segment_ref,
            ai_queue=ai_queue,
        )
        ai_capture_thread.start()

        consumer_thread = threading.Thread(
            target=_ai_consumer_stub, args=(ai_queue,), daemon=True, name="AIConsumer"
        )
        consumer_thread.start()

    preview: DebugPreview | None = DebugPreview(seg_map) if DEBUG_PREVIEW else None

    print("[main] Initialising Beam Eye Tracker API…")
    with GazeTracker() as tracker:

        tracker._api.attempt_starting_the_beam_eye_tracker()

        print("[main] Waiting for tracking data  (Ctrl+C to quit)…")
        print("-" * 60)

        # Keep the last captured images so preview doesn't blank when
        # the tracker briefly returns None
        last_seg_image     = None   # segment capture (for AI / logger)
        last_preview_image = None   # full-screen capture (for debug preview)
        last_active_seg: Segment = seg_map.get_segment(
            SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
        )

        while _running:
            result = tracker.poll(timeout_ms=POLL_TIMEOUT_MS)

            if result is None:
                status = tracker.status()
                if status != bet.TrackingDataReceptionStatus.RECEIVING_TRACKING_DATA:
                    print(f"[main] Status: {status.name}", end="\r")

                # Still update preview even when tracking is lost
                if preview is not None and last_preview_image is not None:
                    preview.update(
                        seg_image=last_preview_image,
                        active_seg=last_active_seg,
                        dwell_progress=dwell.dwell_progress(),
                        fired_seg=dwell.consume_fired(),
                    )
                continue

            gaze_x, gaze_y, confidence = result

            # ------------------------------------------------------------------
            # Segment lookup
            # ------------------------------------------------------------------
            active_seg = seg_map.get_segment(gaze_x, gaze_y)
            current_segment_ref[0] = active_seg
            last_active_seg = active_seg

            # ------------------------------------------------------------------
            # Dwell check → may fire command
            # ------------------------------------------------------------------
            dwell.update(active_seg)
            fired_seg = dwell.consume_fired()

            # ------------------------------------------------------------------
            # Capture the active segment for logging / AI
            # ------------------------------------------------------------------
            seg_image = capture_segment(active_seg)
            last_seg_image = seg_image

            # Log entry
            logger.log(
                gaze_x=gaze_x,
                gaze_y=gaze_y,
                confidence=confidence,
                image=seg_image,
                segment_name=active_seg.name,
            )

            # ------------------------------------------------------------------
            # Capture the full screen for the debug preview
            # ------------------------------------------------------------------
            if preview is not None:
                preview_image      = capture_fullscreen()
                last_preview_image = preview_image
                preview.update(
                    seg_image=preview_image,
                    active_seg=active_seg,
                    dwell_progress=dwell.dwell_progress(),
                    fired_seg=fired_seg,
                )

    # Shutdown
    if ai_capture_thread is not None:
        ai_capture_thread.stop()

    print("[main] Done.")


if __name__ == "__main__":
    main()
