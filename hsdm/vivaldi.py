"""Vivaldi network coordinates (Dabek et al., 2004) with a height term.

Used for one thing only: estimating the round-trip time to a peer we have
heard about but not yet measured, so that between two equally relevant peers
the nearer one is asked first.  Coordinates never encode meaning.
"""

from __future__ import annotations

import math
import random
from typing import List, Sequence

CE = 0.25  # error-weighted step constants from the paper
CC = 0.25


class Vivaldi:
    def __init__(self, seed: int = 0) -> None:
        rng = random.Random(seed)
        self.pos: List[float] = [rng.uniform(-0.5, 0.5) for _ in range(3)]
        self.height = 0.1
        self.error = 1.0

    @staticmethod
    def valid(pos: Sequence[float], height: float, error: float) -> bool:
        vals = list(pos) + [height, error]
        return all(math.isfinite(v) and abs(v) < 60_000 for v in vals) and height >= 0 and error >= 0

    def distance(self, pos: Sequence[float], height: float) -> float:
        """Predicted round-trip time in milliseconds."""
        return math.dist(self.pos, pos) + self.height + height

    def update(self, rtt_ms: float, pos: Sequence[float], height: float, error: float) -> None:
        if rtt_ms <= 0 or not self.valid(pos, height, error):
            return
        predicted = self.distance(pos, height)
        weight = self.error / max(1e-6, self.error + error)
        sample_error = abs(predicted - rtt_ms) / rtt_ms
        self.error = min(2.0, max(0.01, sample_error * CE * weight + self.error * (1 - CE * weight)))
        step = CC * weight * (rtt_ms - predicted)
        flat = math.dist(self.pos, pos)
        if flat > 1e-9:
            unit = [(a - b) / flat for a, b in zip(self.pos, pos)]
        else:
            unit = [random.uniform(-1, 1) for _ in range(3)]
            norm = math.sqrt(sum(u * u for u in unit)) or 1.0
            unit = [u / norm for u in unit]
        self.pos = [p + step * u for p, u in zip(self.pos, unit)]
        self.height = max(0.0, self.height + step * 0.1)
