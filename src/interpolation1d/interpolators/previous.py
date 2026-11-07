from __future__ import annotations

from dataclasses import dataclass
from typing import override

import numpy as np
from numpy.typing import NDArray

from ..interpolator import Interpolator
from ..types import FloatArray


@dataclass(frozen=True, kw_only=True)
class PreviousInterpolator(Interpolator):
    """
    The value of the latest earlier observation, held until the next one.

    Only the past is used, as a live system would see it. Extending before the
    first observation holds its value.
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
        previous: NDArray[np.intp] = np.maximum(
            np.searchsorted(positions, queries, side="right") - 1, 0
        )
        return values[previous]
