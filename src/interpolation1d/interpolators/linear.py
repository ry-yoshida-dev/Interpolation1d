from __future__ import annotations

from dataclasses import dataclass
from typing import override

import numpy as np
from numpy.typing import NDArray

from ..interpolator import Interpolator
from ..types import FloatArray


@dataclass(frozen=True, kw_only=True)
class LinearInterpolator(Interpolator):
    """
    A straight line between the observations on either side.

    Extending continues the line through the first two or the last two
    observations.
    """

    @property
    @override
    def minimum_observations(self) -> int:
        """
        Get the number of observations needed to estimate a missing index.

        Returns
        -------
        int
            Two, the ends of a line.
        """
        return 2

    @override
    def _estimate(
        self, positions: FloatArray, values: FloatArray, queries: FloatArray
    ) -> FloatArray:
        # Clipping the segment makes queries beyond the ends use the outermost segments.
        segment: NDArray[np.intp] = np.clip(
            np.searchsorted(positions, queries, side="right") - 1,
            0,
            positions.size - 2,
        )
        start: FloatArray = positions[segment]
        weight: FloatArray = (queries - start) / (positions[segment + 1] - start)
        broadcast_weight: FloatArray = weight.reshape(-1, *(1,) * (values.ndim - 1))
        return values[segment] + broadcast_weight * (
            values[segment + 1] - values[segment]
        )
