"""
main.py
Entry point for the Beam Eye Tracker gaze capture application.

Run:
    py -3.10 main.py

Press Ctrl+C to stop.
"""

from __future__ import annotations

import queue
import signal
import threading
import tkinter as tk

import eyeware.beam_eye_tracker as bet
from PIL import ImageDraw, ImageTk

from gaze_tracker import GazeTracker, SCREEN_WIDTH, SCREEN_HEIGHT
from screen_capture import capture_region
from logger import GazeLogger

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Target ~24 fps → ~42 ms between frames.
# wait_for_new_tracking_state_set will block for up to this many ms.
POLL_TIMEOUT_MS = 42

# Set to True to open a live preview window showing each captured region.
# Close the window or press Ctrl+C in the terminal to quit.
DEBUG_PREVIEW = True

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
# Debug preview window (Tkinter, runs in its own daemon thread)
# ---------------------------------------------------------------------------

class DebugPreview:
    """
    Lightweight Tkinter window that displays the latest gaze capture.
    Runs in a background daemon thread so it never blocks the main loop.
    Frames are passed via a queue (only the most-recent frame is kept).
    """

    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        self._root = tk.Tk()
        self._root.title("Gaze Capture – Debug Preview")
        self._root.resizable(False, False)
        self._label = tk.Label(self._root, bg="black")
        self._label.pack()
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._root.after(16, self._poll)   # ~60 Hz UI refresh
        self._root.mainloop()

    def _poll(self) -> None:
        """Pull the latest frame from the queue and update the label."""
        try:
            pil_image = self._queue.get_nowait()
            # PhotoImage must be created on the Tk thread
            self._photo = ImageTk.PhotoImage(pil_image)  # keep reference to prevent GC
            self._label.config(image=self._photo)
        except queue.Empty:
            pass
        self._root.after(16, self._poll)

    def _on_close(self) -> None:
        global _running
        _running = False
        self._root.destroy()

    def update(self, pil_image) -> None:
        """Send a new PIL image to the preview window (non-blocking).
        
        The raw PIL image is queued; ImageTk.PhotoImage conversion happens
        on the Tk thread inside _poll() to ensure thread safety.
        """
        try:
            # Drop any unread frame so we always show the freshest capture
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(pil_image)
        except Exception:
            pass  # silently ignore if the window has been closed


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    global _running
    logger = GazeLogger()

    preview: DebugPreview | None = DebugPreview() if DEBUG_PREVIEW else None

    print(f"[main] Detected screen: {SCREEN_WIDTH}×{SCREEN_HEIGHT} px")
    print("[main] Initialising Beam Eye Tracker API…")
    with GazeTracker() as tracker:

        # Optionally try to auto-start the Beam app if it isn't running
        tracker._api.attempt_starting_the_beam_eye_tracker()

        print("[main] Waiting for tracking data  (Ctrl+C to quit)…")
        print("-" * 60)

        while _running:
            result = tracker.poll(timeout_ms=POLL_TIMEOUT_MS)

            if result is None:
                # Timeout or lost tracking – print status at reduced rate
                status = tracker.status()
                if status != bet.TrackingDataReceptionStatus.RECEIVING_TRACKING_DATA:
                    print(f"[main] Status: {status.name}", end="\r")
                continue

            gaze_x, gaze_y, confidence = result

            # Capture 400×400 region centred on the gaze point
            image = capture_region(gaze_x, gaze_y, SCREEN_WIDTH, SCREEN_HEIGHT)

            # Log the frame (in-memory, entries auto-discarded when buffer is full)
            entry = logger.log(
                gaze_x=gaze_x,
                gaze_y=gaze_y,
                confidence=confidence,
                image=image,
            )

            # ----------------------------------------------------------------
            # TODO (future): pass entry.image to an AI vision model here
            # and enrich entry with a description of what the user is looking at.
            # ----------------------------------------------------------------

            # Live debug preview
            if preview is not None and entry.image is not None:
                # Draw overlay directly onto a copy of the PIL image
                debug_img = entry.image.copy()
                draw = ImageDraw.Draw(debug_img)

                # Crosshair at centre
                cx, cy = debug_img.width // 2, debug_img.height // 2
                draw.line([(cx - 12, cy), (cx + 12, cy)], fill="lime", width=2)
                draw.line([(cx, cy - 12), (cx, cy + 12)], fill="lime", width=2)

                # Gaze info label
                label = f"({gaze_x}, {gaze_y})  {confidence.name}"
                draw.rectangle([(4, 4), (len(label) * 7 + 8, 22)], fill="black")
                draw.text((6, 5), label, fill="lime")

                preview.update(debug_img)

    print("[main] Done.")


if __name__ == "__main__":
    main()
