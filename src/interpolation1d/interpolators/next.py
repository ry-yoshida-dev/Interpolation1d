from __future__ import annotations

from dataclasses import dataclass
from typing import override

import numpy as np
from numpy.typing import NDArray

from ..interpolator import Interpolator
from ..types import FloatArray


@dataclass(frozen=True, kw_only=True)
class NextInterpolator(Interpolator):
    """
    The value of the earliest later observation.

    Extending after the last observation holds its value.
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
        following: NDArray[np.intp] = np.minimum(
            np.searchsorted(positions, queries), positions.size - 1
        )
        return values[following]
