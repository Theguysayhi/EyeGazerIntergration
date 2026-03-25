"""
screen_capture.py
Captures screen regions – either a floating crop centred on a gaze point,
or a fixed segment rectangle from the grid defined in segment_map.py.
"""

from __future__ import annotations

import mss
import mss.tools
from PIL import Image

# Size of the capture region in pixels
CAPTURE_SIZE = 400  # 400 x 400


def _get_primary_monitor() -> dict:
    """Return the mss monitor dict for the primary display."""
    with mss.mss() as sct:
        # monitors[0] = full virtual screen (all monitors combined)
        # monitors[1] = primary monitor
        return dict(sct.monitors[1])


def capture_region(
    gaze_x: int,
    gaze_y: int,
    screen_width: int = 0,
    screen_height: int = 0,
) -> Image.Image:
    """
    Capture a CAPTURE_SIZE x CAPTURE_SIZE region centred on (gaze_x, gaze_y).

    The region is clamped so it never exceeds the screen boundaries.

    Parameters
    ----------
    gaze_x, gaze_y : int
        Screen-space gaze coordinates (may be slightly outside screen bounds).
    screen_width, screen_height : int, optional
        Full screen dimensions used for clamping.
        If 0 (default), the primary monitor size is detected automatically via mss.

    Returns
    -------
    PIL.Image.Image
        The captured region as an RGB image.
    """
    # Auto-detect screen dimensions if not provided
    if screen_width <= 0 or screen_height <= 0:
        mon = _get_primary_monitor()
        screen_width  = mon["width"]
        screen_height = mon["height"]

    # Clamp gaze point to valid screen range first
    gaze_x = max(0, min(screen_width  - 1, gaze_x))
    gaze_y = max(0, min(screen_height - 1, gaze_y))

    half = CAPTURE_SIZE // 2

    # Initial region centred on gaze
    left   = gaze_x - half
    top    = gaze_y - half
    right  = left + CAPTURE_SIZE
    bottom = top  + CAPTURE_SIZE

    # Shift the window if it overflows the right / bottom edge
    if right > screen_width:
        left  = screen_width - CAPTURE_SIZE
        right = screen_width
    if bottom > screen_height:
        top    = screen_height - CAPTURE_SIZE
        bottom = screen_height

    # Clamp to left / top edge (handles screens smaller than CAPTURE_SIZE)
    left = max(0, left)
    top  = max(0, top)

    # Recompute right/bottom from the final anchored left/top
    right  = min(screen_width,  left + CAPTURE_SIZE)
    bottom = min(screen_height, top  + CAPTURE_SIZE)

    monitor = {
        "left":   left,
        "top":    top,
        "width":  right - left,
        "height": bottom - top,
    }

    with mss.mss() as sct:
        raw = sct.grab(monitor)
        img = Image.frombytes("RGB", raw.size, bytes(raw.raw), "raw", "BGRX")

    return img


def capture_fullscreen() -> Image.Image:
    """
    Capture the entire primary monitor.

    Returns
    -------
    PIL.Image.Image
        The full screen as an RGB image.
    """
    mon = _get_primary_monitor()
    with mss.mss() as sct:
        raw = sct.grab(mon)
        img = Image.frombytes("RGB", raw.size, bytes(raw.raw), "raw", "BGRX")
    return img


def capture_segment(segment) -> Image.Image:
    """
    Capture the exact bounding rectangle of a Segment object.

    Parameters
    ----------
    segment : Segment
        A Segment instance from segment_map.py.  Only the .left/.top/.width/.height
        attributes are used, so duck-typing is fine.

    Returns
    -------
    PIL.Image.Image
        The captured segment as an RGB image.
    """
    monitor = {
        "left":   segment.left,
        "top":    segment.top,
        "width":  segment.width,
        "height": segment.height,
    }

    with mss.mss() as sct:
        raw = sct.grab(monitor)
        img = Image.frombytes("RGB", raw.size, bytes(raw.raw), "raw", "BGRX")

    return img


def capture_segments_expanded(segments) -> Image.Image:
    """
    Capture the union bounding rectangle of a list of Segment objects in a
    single screenshot call.

    Intended for the 3×3 neighbourhood capture: pass the result of
    ``SegmentMap.get_neighbours(seg)`` and the returned image covers all
    those segments in one contiguous block.

    Parameters
    ----------
    segments : list[Segment]
        One or more Segment instances.  Duck-typing is fine – only
        .left / .top / .right / .bottom are used.

    Returns
    -------
    PIL.Image.Image
        The captured expanded region as an RGB image.

    Raises
    ------
    ValueError
        If *segments* is empty.
    """
    if not segments:
        raise ValueError("capture_segments_expanded requires at least one segment")

    left   = min(s.left   for s in segments)
    top    = min(s.top    for s in segments)
    right  = max(s.right  for s in segments)
    bottom = max(s.bottom for s in segments)

    monitor = {
        "left":   left,
        "top":    top,
        "width":  right  - left,
        "height": bottom - top,
    }

    with mss.mss() as sct:
        raw = sct.grab(monitor)
        img = Image.frombytes("RGB", raw.size, bytes(raw.raw), "raw", "BGRX")

    return img
