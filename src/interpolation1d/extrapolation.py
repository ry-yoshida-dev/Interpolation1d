from __future__ import annotations

from enum import Enum


class Extrapolation(Enum):
    """
    How to fill indices before the first or after the last observation.

    Attributes
    ----------
    REJECT
        Raise ValueError, so that a span beyond the observations is never filled
        by accident.
    HOLD
        Repeat the first or last observed value.
    EXTEND
        Continue the interpolation method itself, such as a line or polynomial.
        Step methods have no trend to continue, so they hold the edge values.
    NAN
        Leave NaN.
    """

    REJECT = "reject"
    HOLD = "hold"
    EXTEND = "extend"
    NAN = "nan"
