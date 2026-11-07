from __future__ import annotations

from dataclasses import dataclass
from typing import cast, override

from scipy.interpolate import Akima1DInterpolator

from ..interpolator import Interpolator
from ..types import FloatArray


@dataclass(frozen=True, kw_only=True)
class AkimaInterpolator(Interpolator):
    """
    A piecewise cubic whose slopes come from nearby observations only (Akima).

    An outlier disturbs only its neighborhood, and oscillation is milder than with
    a cubic spline. Only the first derivative is continuous.

    Attributes
    ----------
    is_modified : bool
        Whether to use the modified Akima method, which also avoids overshoot where
        the data is flat for a while.

    Raises
    ------
    TypeError
        If a field does not have its annotated type.
    """

    is_modified: bool = True

    @override
    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(cast(object, self.is_modified), bool):
            raise TypeError(
                f"is_modified must be a bool, got {type(self.is_modified).__name__}"
            )

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
        curve: Akima1DInterpolator = Akima1DInterpolator(
            positions,
            values,
            axis=0,
            method="makima" if self.is_modified else "akima",
            extrapolate=True,
        )
        return curve(queries)
