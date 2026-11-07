from __future__ import annotations

from dataclasses import dataclass
from typing import override

import numpy as np
from numpy.typing import NDArray

from ..interpolator import Interpolator
from ..types import FloatArray


@dataclass(frozen=True, kw_only=True)
class NearestInterpolator(Interpolator):
    """
    The value of the closest observation, the earlier one when two are equally close.
    """

    @property
    @override
    def minimum_observations(self) -> int:
        """
        Get the number of observations needed to estimate a missing index.

        Returns
        -------
        int
            One.
        """
        return 1

    @override
    def _estimate(
        self, positions: FloatArray, values: FloatArray, queries: FloatArray
    ) -> FloatArray:
        following: NDArray[np.intp] = np.searchsorted(positions, queries)
        later: NDArray[np.intp] = np.minimum(following, positions.size - 1)
        earlier: NDArray[np.intp] = np.maximum(following - 1, 0)
        is_later_closer: NDArray[np.bool_] = np.abs(
            positions[later] - queries
        ) < np.abs(queries - positions[earlier])
        return values[np.where(is_later_closer, later, earlier)]
