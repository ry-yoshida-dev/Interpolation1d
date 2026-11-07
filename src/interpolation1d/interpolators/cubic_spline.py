from __future__ import annotations

from dataclasses import dataclass
from typing import cast, override

from scipy.interpolate import CubicSpline

from ..interpolator import Interpolator
from ..spline_boundary import SplineBoundary
from ..types import FloatArray


@dataclass(frozen=True, kw_only=True)
class CubicSplineInterpolator(Interpolator):
    """
    A cubic spline whose first and second derivatives are continuous.

    The smoothest of the methods, but it can overshoot between observations, such
    as below zero for non-negative data. Extending continues the outermost cubics,
    which diverge quickly.

    Attributes
    ----------
    boundary : SplineBoundary
        The condition at the first and last observation.

    Raises
    ------
    TypeError
        If a field does not have its annotated type.
    """

    boundary: SplineBoundary = SplineBoundary.NOT_A_KNOT

    @override
    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(cast(object, self.boundary), SplineBoundary):
            raise TypeError(
                f"boundary must be a SplineBoundary, got {type(self.boundary).__name__}"
            )

    @property
    @override
    def minimum_observations(self) -> int:
        """
        Get the number of observations needed to estimate a missing index.

        Returns
        -------
        int
            Two, with which every boundary gives a line or a single cubic.
        """
        return 2

    @override
    def _estimate(
        self, positions: FloatArray, values: FloatArray, queries: FloatArray
    ) -> FloatArray:
        spline: CubicSpline = CubicSpline(
            positions, values, axis=0, bc_type=self.boundary.value, extrapolate=True
        )
        return spline(queries)
