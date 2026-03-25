"""
debug_preview.py
Tkinter debug preview window showing the live full-screen capture with:
  - Segment grid overlay
  - Active segment highlighted in yellow
  - Dwell progress bar
  - Green flash on command fire
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk

from PIL import ImageDraw, ImageTk

from .gaze_tracker import SCREEN_WIDTH, SCREEN_HEIGHT
from .segment_map import SegmentMap, Segment

# Scale factor: preview is rendered at this fraction of the raw capture size.
PREVIEW_SCALE = 0.45


class DebugPreview:
    """
    Runs its own Tkinter event loop in a daemon thread so it doesn't block
    the main gaze-polling loop.

    Call `update()` from any thread; it is thread-safe.
    """

    def __init__(self, segment_map: SegmentMap) -> None:
        self._seg_map = segment_map
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._thread  = threading.Thread(target=self._run, daemon=True)
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
            pil_image, _, _, _, _ = self._queue.get_nowait()
            self._photo = ImageTk.PhotoImage(pil_image)
            self._label.config(image=self._photo)
        except queue.Empty:
            pass
        self._root.after(16, self._poll)

    def _on_close(self) -> None:
        # Signal the main loop to stop by importing and setting the flag.
        import main as _main
        _main._running = False
        self._root.destroy()

    # ------------------------------------------------------------------
    def update(
        self,
        seg_image,
        active_seg: Segment,
        dwell_progress: float,
        fired_seg: Segment | None,
        nsfw_result: tuple[bool, str, float] | None = None,
    ) -> None:
        """Build the annotated preview image and push it to the Tk thread."""
        try:
            w = int(seg_image.width  * PREVIEW_SCALE)
            h = int(seg_image.height * PREVIEW_SCALE)
            display = seg_image.resize((w, h))
            draw    = ImageDraw.Draw(display, "RGBA")

            seg_map        = self._seg_map
            cols, rows     = seg_map.columns, seg_map.rows
            cell_w, cell_h = w // cols, h // rows

            # Grid lines
            for c in range(1, cols):
                x = int(w * c / cols)
                draw.line([(x, 0), (x, h)], fill=(200, 200, 200, 160), width=1)
            for r in range(1, rows):
                y = int(h * r / rows)
                draw.line([(0, y), (w, y)], fill=(200, 200, 200, 160), width=1)

            # Segment labels
            for seg in seg_map.all_segments():
                cx = int((seg.col + 0.5) * cell_w)
                cy = int((seg.row + 0.5) * cell_h)
                draw.text((cx - 10, cy - 6), seg.name.replace("segment_", ""),
                          fill=(200, 200, 200, 200))

            # AI capture area: union of the 3×3 neighbourhood (blue tint)
            neighbours = self._seg_map.get_neighbours(active_seg, radius=1)
            nb_col_min = min(s.col for s in neighbours)
            nb_col_max = max(s.col for s in neighbours)
            nb_row_min = min(s.row for s in neighbours)
            nb_row_max = max(s.row for s in neighbours)
            nx0 = nb_col_min * cell_w
            ny0 = nb_row_min * cell_h
            nx1 = (nb_col_max + 1) * cell_w
            ny1 = (nb_row_max + 1) * cell_h
            draw.rectangle([(nx0, ny0), (nx1, ny1)], fill=(80, 180, 255, 30))
            draw.rectangle([(nx0, ny0), (nx1, ny1)], outline=(80, 180, 255, 200), width=2)

            # Active segment highlight (yellow, drawn on top of the blue area)
            ax0 = active_seg.col * cell_w
            ay0 = active_seg.row * cell_h
            ax1, ay1 = ax0 + cell_w, ay0 + cell_h
            draw.rectangle([(ax0, ay0), (ax1, ay1)], fill=(255, 220, 0, 55))
            draw.rectangle([(ax0, ay0), (ax1, ay1)], outline=(255, 220, 0, 220), width=2)

            # Dwell progress bar
            if dwell_progress < 1.0:
                bar_w = int(cell_w * dwell_progress)
                draw.rectangle([(ax0, ay1 - 5), (ax0 + bar_w, ay1)], fill=(255, 220, 0, 230))

            # Fired flash (green)
            if fired_seg is not None:
                fx0 = fired_seg.col * cell_w
                fy0 = fired_seg.row * cell_h
                fx1, fy1 = fx0 + cell_w, fy0 + cell_h
                draw.rectangle([(fx0, fy0), (fx1, fy1)], fill=(0, 255, 80, 100))
                draw.rectangle([(fx0, fy0), (fx1, fy1)], outline=(0, 255, 80, 255), width=3)

            # Info label (bottom-left style, top strip)
            info = (f"seg={active_seg.name}  dwell={int(dwell_progress * 100)}%  "
                    f"{SCREEN_WIDTH}×{SCREEN_HEIGHT}")
            draw.rectangle([(0, 0), (len(info) * 7 + 6, 18)], fill=(0, 0, 0, 180))
            draw.text((4, 2), info, fill=(200, 255, 200))

            # NSFW status badge (top-right corner)
            if nsfw_result is None:
                badge_text  = "AI: pending..."
                badge_bg    = (80, 80, 80, 200)
                badge_fg    = (200, 200, 200)
            elif nsfw_result[0]:
                # NSFW flagged
                label_str   = nsfw_result[1] or "nsfw"
                score_str   = f"{int(nsfw_result[2] * 100)}%"
                badge_text  = f"!! NSFW: {label_str} {score_str}"
                badge_bg    = (200, 0, 0, 220)
                badge_fg    = (255, 255, 255)
            else:
                badge_text  = ">> SAFE"
                badge_bg    = (0, 160, 40, 200)
                badge_fg    = (200, 255, 200)

            badge_w = len(badge_text) * 7 + 10
            draw.rectangle([(w - badge_w, 0), (w, 18)], fill=badge_bg)
            draw.text((w - badge_w + 5, 2), badge_text, fill=badge_fg)

            payload = (display, active_seg, dwell_progress, fired_seg, nsfw_result)
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(payload)

        except Exception:
            pass  # silently ignore if window was closed
