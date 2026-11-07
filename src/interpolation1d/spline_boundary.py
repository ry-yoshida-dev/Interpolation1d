from __future__ import annotations

from enum import Enum


class SplineBoundary(Enum):
    """
    The condition that closes a cubic spline at the first and last observation.

    Attributes
    ----------
    NOT_A_KNOT
        The first two and the last two pieces are one cubic each, which suits data
        whose behavior at the ends is unknown.
    NATURAL
        Zero second derivative, so the curve straightens towards the ends.
    CLAMPED
        Zero first derivative, so the curve flattens towards the ends.
    """

    NOT_A_KNOT = "not-a-knot"
    NATURAL = "natural"
    CLAMPED = "clamped"
