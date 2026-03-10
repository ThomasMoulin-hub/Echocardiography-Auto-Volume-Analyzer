from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Optional, Tuple

Point = Tuple[int, int]


@dataclass
class Measurement:
    """A two-point measurement on an image."""

    point1: Optional[Point] = None
    point2: Optional[Point] = None

    def is_complete(self) -> bool:
        return self.point1 is not None and self.point2 is not None

    def distance_pixels(self) -> float:
        if not self.is_complete():
            return 0.0
        assert self.point1 is not None and self.point2 is not None
        return hypot(self.point2[0] - self.point1[0], self.point2[1] - self.point1[1])


@dataclass
class VolumeResult:
    edv: float
    esv: float
    sv: float
    ef: float
    co: Optional[float] = None


@dataclass
class DopplerResult:
    d_ao: float
    vti: float
    hr: float
    area_ao: float
    sv: float
    co: float

