from __future__ import annotations

from dataclasses import dataclass
from typing import override

from scipy.interpolate import PchipInterpolator as ScipyPchipInterpolator

from ..interpolator import Interpolator
from ..types import FloatArray


@dataclass(frozen=True, kw_only=True)
class PchipInterpolator(Interpolator):
    """
    A piecewise cubic that preserves the shape of the observations (PCHIP).

    Between two observations the curve stays within their values, so it never
    overshoots and keeps monotonic data monotonic. Only the first derivative is
    continuous.
    """

    @property
    @override
    def minimum_observations(self) -> int:
        """
        Get the number of observations needed to estimate a missing index.

        Returns
        -------
        int
            Two.
        """
        return 2

    @override
    def _estimate(
        self, positions: FloatArray, values: FloatArray, queries: FloatArray
    ) -> FloatArray:
        curve: ScipyPchipInterpolator = ScipyPchipInterpolator(
            positions, values, axis=0, extrapolate=True
        )
        return curve(queries)
