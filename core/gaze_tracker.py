"""
gaze_tracker.py
Handles connection to the Beam Eye Tracker API and provides gaze coordinate polling.
"""

import mss
import eyeware.beam_eye_tracker as bet

APP_FRIENDLY_NAME = "GazeCapture"


def _detect_screen_size() -> tuple[int, int]:
    """Return (width, height) of the primary monitor using mss."""
    with mss.mss() as sct:
        mon = sct.monitors[1]  # monitors[1] = primary monitor
        return mon["width"], mon["height"]


# Auto-detect screen resolution at import time.
# These are exported so other modules (e.g. main.py) can read them if needed.
SCREEN_WIDTH, SCREEN_HEIGHT = _detect_screen_size()


def build_viewport_geometry() -> bet.ViewportGeometry:
    """Build a ViewportGeometry that covers the full primary screen."""
    top_left = bet.Point()
    top_left.x = 0
    top_left.y = 0

    bottom_right = bet.Point()
    bottom_right.x = SCREEN_WIDTH - 1
    bottom_right.y = SCREEN_HEIGHT - 1

    return bet.ViewportGeometry(top_left, bottom_right)  # type: ignore[call-arg]


class GazeTracker:
    """
    Wraps the Beam Eye Tracker API.

    Usage:
        tracker = GazeTracker()
        with tracker:
            while True:
                result = tracker.poll()
                if result:
                    x, y, confidence = result
    """

    def __init__(self):
        viewport = build_viewport_geometry()
        self._api = bet.API(APP_FRIENDLY_NAME, viewport)
        self._timestamp = bet.NULL_DATA_TIMESTAMP()

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, *_):
        # The API object cleans up on GC; nothing explicit required.
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def status(self) -> bet.TrackingDataReceptionStatus:
        """Return the current tracking data reception status."""
        return self._api.get_tracking_data_reception_status()

    def poll(self, timeout_ms: int = 42) -> tuple | None:
        """
        Block until new tracking data arrives (or timeout_ms elapses).

        Returns
        -------
        (x: int, y: int, confidence: TrackingConfidence)
            Screen-space gaze coordinates and confidence, or
        None
            If timeout elapsed or confidence is LOST_TRACKING.
        """
        has_new = self._api.wait_for_new_tracking_state_set(
            self._timestamp, timeout_ms  # type: ignore[arg-type]
        )
        if not has_new:
            return None

        state_set = self._api.get_latest_tracking_state_set()
        user_state = state_set.user_state()
        gaze = user_state.unified_screen_gaze

        # Normalise confidence: some SDK versions return a plain int instead of the enum
        conf_raw = gaze.confidence
        conf = bet.TrackingConfidence(conf_raw) if isinstance(conf_raw, int) else conf_raw

        if conf == bet.TrackingConfidence.LOST_TRACKING:
            return None

        x = gaze.point_of_regard.x
        y = gaze.point_of_regard.y
        return x, y, conf
