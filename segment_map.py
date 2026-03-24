"""
segment_map.py
Divides the screen into a configurable N×M grid of named segments.
Each segment can have an optional shell command associated with it.

Configuration is read from config.ini at import time.
"""

from __future__ import annotations

import configparser
import os
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.ini")

_cfg = configparser.ConfigParser(inline_comment_prefixes=(";",))
_cfg.read(_CONFIG_PATH)

GRID_COLUMNS: int = _cfg.getint("grid", "columns", fallback=3)
GRID_ROWS: int    = _cfg.getint("grid", "rows",    fallback=3)


# ---------------------------------------------------------------------------
# Segment dataclass
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    """Represents one cell in the screen grid."""

    row: int
    col: int
    # Screen-space bounding rectangle (pixels, inclusive left/top, exclusive right/bottom)
    left:   int = 0
    top:    int = 0
    right:  int = 0
    bottom: int = 0
    # Shell command to run when this segment is dwelled on (empty = no command)
    command: str = field(default="")

    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        """Unique identifier e.g. 'segment_1_2'."""
        return f"segment_{self.row}_{self.col}"

    @property
    def rect(self) -> tuple[int, int, int, int]:
        """(left, top, right, bottom) bounding rectangle."""
        return (self.left, self.top, self.right, self.bottom)

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def center(self) -> tuple[int, int]:
        return (self.left + self.width // 2, self.top + self.height // 2)

    def contains(self, x: int, y: int) -> bool:
        """Return True if (x, y) falls inside this segment."""
        return self.left <= x < self.right and self.top <= y < self.bottom

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Segment):
            return NotImplemented
        return self.row == other.row and self.col == other.col

    def __hash__(self) -> int:
        return hash((self.row, self.col))

    def __repr__(self) -> str:
        return (
            f"<Segment {self.name} rect=({self.left},{self.top})-"
            f"({self.right},{self.bottom}) cmd={self.command!r}>"
        )


# ---------------------------------------------------------------------------
# SegmentMap
# ---------------------------------------------------------------------------

class SegmentMap:
    """
    Builds and owns the full grid of Segment objects for a given screen size.

    Parameters
    ----------
    screen_width, screen_height : int
        Pixel dimensions of the primary monitor.
    columns, rows : int
        Grid dimensions (default read from config.ini).
    """

    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        columns: int = GRID_COLUMNS,
        rows: int = GRID_ROWS,
    ) -> None:
        self.screen_width  = screen_width
        self.screen_height = screen_height
        self.columns = columns
        self.rows    = rows

        self._segments: list[list[Segment]] = []
        self._build()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        """Compute bounding rects and attach commands for every cell."""
        cols, rows = self.columns, self.rows
        sw, sh     = self.screen_width, self.screen_height

        # Use integer division to avoid sub-pixel gaps.
        # The last column/row absorbs any remainder.
        col_edges = [sw * c // cols for c in range(cols + 1)]
        row_edges = [sh * r // rows for r in range(rows + 1)]

        commands_section = _cfg["commands"] if _cfg.has_section("commands") else {}

        self._segments = []
        for r in range(rows):
            row_list: list[Segment] = []
            for c in range(cols):
                key = f"segment_{r}_{c}"
                cmd = commands_section.get(key, "").strip()
                seg = Segment(
                    row=r,
                    col=c,
                    left=col_edges[c],
                    top=row_edges[r],
                    right=col_edges[c + 1],
                    bottom=row_edges[r + 1],
                    command=cmd,
                )
                row_list.append(seg)
            self._segments.append(row_list)

    # ------------------------------------------------------------------
    def get_segment(self, x: int, y: int) -> Segment:
        """
        Return the Segment that contains screen coordinate (x, y).

        Uses O(1) arithmetic rather than scanning all cells.
        Coordinates are clamped to the screen bounds first.
        """
        # Clamp
        x = max(0, min(self.screen_width  - 1, x))
        y = max(0, min(self.screen_height - 1, y))

        col = min(int(x * self.columns / self.screen_width),  self.columns - 1)
        row = min(int(y * self.rows    / self.screen_height), self.rows    - 1)

        return self._segments[row][col]

    # ------------------------------------------------------------------
    def all_segments(self) -> list[Segment]:
        """Return a flat list of all segments in row-major order."""
        return [seg for row in self._segments for seg in row]

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"<SegmentMap {self.columns}x{self.rows} "
            f"screen={self.screen_width}x{self.screen_height}>"
        )
