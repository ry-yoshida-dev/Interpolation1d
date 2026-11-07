from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .types import FloatArray


@dataclass(frozen=True)
class FilledSeries:
    """
    A series with a value at every requested index.

    Attributes
    ----------
    indices : NDArray[np.int64]
        The requested indices, strictly increasing.
    values : FloatArray
        The value at each index, with shape ``(len(indices), *value_shape)``. The
        dtype is that of the observed values when they are floating, and float64
        when they are integers. Estimates are computed in float64 whatever the dtype.
    is_observed : NDArray[np.bool_]
        Whether each index was observed, so that its value is the observation itself
        rather than an estimate.
    """

    indices: NDArray[np.int64]
    values: FloatArray
    is_observed: NDArray[np.bool_]
