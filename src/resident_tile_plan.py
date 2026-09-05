from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class TilePlan:
    """Static row tiling for residual packages.

    Tiles are a transfer unit, not an approximation. The final tile may be
    shorter than ``tile_rows``; callers must use the returned ``(start, stop)``
    range when slicing payloads.
    """

    rows: int
    tile_rows: int
    projections: tuple[str, ...] | Iterable[str]

    def __post_init__(self) -> None:
        if self.rows < 1:
            raise ValueError("rows must be positive")
        if self.tile_rows < 1 or self.tile_rows % 32:
            raise ValueError("tile_rows must be a positive multiple of 32")
        names = tuple(str(name) for name in self.projections)
        if not names:
            raise ValueError("at least one projection is required")
        if len(set(names)) != len(names):
            raise ValueError("projection names must be unique")
        object.__setattr__(self, "projections", names)

    def ranges(self) -> list[tuple[int, int]]:
        return [
            (start, min(start + self.tile_rows, self.rows))
            for start in range(0, self.rows, self.tile_rows)
        ]

    def tile_slices(self) -> list[tuple[int, int]]:
        return self.ranges()

    def tile_bytes(self, *, cols: int, alpha_cols: int, tile_rows: int | None = None) -> int:
        if cols < 1 or cols % 2 or alpha_cols < 1:
            raise ValueError("invalid payload dimensions")
        height = self.tile_rows if tile_rows is None else int(tile_rows)
        if height < 1 or height > self.tile_rows:
            raise ValueError("invalid tile height")
        per_projection = height * (cols // 2) + height * alpha_cols * 4
        return len(self.projections) * per_projection

    def total_bytes(self, *, cols: int, alpha_cols: int) -> int:
        return sum(
            self.tile_bytes(cols=cols, alpha_cols=alpha_cols, tile_rows=stop - start)
            for start, stop in self.ranges()
        )
