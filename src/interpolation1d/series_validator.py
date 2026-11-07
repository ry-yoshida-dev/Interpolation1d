from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

from .types import IndexArray, ValueArray

# Every integer up to 2**53 in magnitude is exact in float64, which the methods compute in
# and into which integer values are converted.
LARGEST_EXACT_INTEGER: int = 2**53


class SeriesValidator:
    """
    Runtime checks of observed series and requested spans.
    """

    @staticmethod
    def require_indices(indices: IndexArray) -> None:
        """
        Check that observation indices are strictly increasing integers.

        Parameters
        ----------
        indices : IndexArray
            The index of each observation.

        Raises
        ------
        TypeError
            If the indices are not a plain NumPy array of integers.
        ValueError
            If the indices are not 1-D, are empty, exceed 2**53 in magnitude, or are
            not strictly increasing.
        """
        SeriesValidator._require_plain_array("indices", indices)
        if not np.issubdtype(indices.dtype, np.integer):
            raise TypeError(f"indices must be integers, got dtype {indices.dtype}")
        if indices.ndim != 1:
            raise ValueError(f"indices must be 1-D, got {indices.ndim} dimensions")
        if indices.size == 0:
            raise ValueError("indices must not be empty")
        smallest: int = int(indices.min())
        largest: int = int(indices.max())
        if smallest < -LARGEST_EXACT_INTEGER or largest > LARGEST_EXACT_INTEGER:
            raise ValueError(
                f"indices must be within ±2**53, got the range [{smallest}, {largest}]"
            )
        # Differences of int64 cannot overflow within ±2**53, unlike those of unsigned types.
        is_disordered: NDArray[np.bool_] = np.diff(indices.astype(np.int64)) <= 0
        if is_disordered.any():
            position: int = int(np.argmax(is_disordered))
            raise ValueError(
                f"indices must be strictly increasing without duplicates, got {indices[position : position + 2]} at positions {position} and {position + 1}; sort it with np.argsort"
            )

    @staticmethod
    def require_values(values: ValueArray, *, n_observations: int) -> None:
        """
        Check that observed values are finite numbers, one entry per observation.

        Parameters
        ----------
        values : ValueArray
            The observed values with shape ``(n_observations, *value_shape)``.
        n_observations : int
            The number of observation indices.

        Raises
        ------
        TypeError
            If the values are not a plain NumPy array of integers or floating-point
            numbers.
        ValueError
            If the first axis does not have one entry per observation, a value is
            NaN or infinite, or an integer value exceeds 2**53 in magnitude.
        OverflowError
            If consecutive values differ by more than float64 can represent.
        """
        SeriesValidator._require_plain_array("values", values)
        is_numeric: bool = np.issubdtype(values.dtype, np.integer) or np.issubdtype(
            values.dtype, np.floating
        )
        if not is_numeric:
            raise TypeError(f"values must be numeric, got dtype {values.dtype}")
        if values.ndim == 0 or values.shape[0] != n_observations:
            raise ValueError(
                f"values must have {n_observations} entries along the first axis to match indices, got shape {values.shape}"
            )
        if np.issubdtype(values.dtype, np.integer) and values.size > 0:
            smallest: int = int(values.min())
            largest: int = int(values.max())
            if smallest < -LARGEST_EXACT_INTEGER or largest > LARGEST_EXACT_INTEGER:
                raise ValueError(
                    f"integer values must be within ±2**53 to be kept exactly in float64, got the range [{smallest}, {largest}]; convert them to float64 first to accept rounding"
                )
        if not np.isfinite(values).all():
            raise ValueError(
                "values must be finite; leave missing observations out of indices and values instead of marking them with NaN"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            steps: NDArray[np.float64] = np.diff(values.astype(np.float64), axis=0)
        if not np.isfinite(steps).all():
            raise OverflowError(
                "values must not differ by more than the float64 maximum between consecutive observations; scale them down"
            )

    @staticmethod
    def require_span(span: range) -> None:
        """
        Check that a span of requested indices is non-empty and increasing.

        Parameters
        ----------
        span : range
            The requested indices.

        Raises
        ------
        TypeError
            If the span is not a range.
        ValueError
            If the span is empty, has a negative step, or exceeds 2**53 in magnitude.
        """
        if not isinstance(cast(object, span), range):
            raise TypeError(
                f"span must be a range, got {type(span).__name__}; resample with range(start, stop, step)"
            )
        if span.step < 0:
            raise ValueError(f"span must have a positive step, got {span}")
        if len(span) == 0:
            raise ValueError(f"span must not be empty, got {span}")
        if span[0] < -LARGEST_EXACT_INTEGER or span[-1] > LARGEST_EXACT_INTEGER:
            raise ValueError(f"span must be within ±2**53, got {span}")

    @staticmethod
    def _require_plain_array(name: str, array: IndexArray | ValueArray) -> None:
        if not isinstance(cast(object, array), np.ndarray):
            raise TypeError(
                f"{name} must be a NumPy array, got {type(array).__name__}; convert it with np.asarray"
            )
        if isinstance(array, np.ma.MaskedArray | np.matrix):
            raise TypeError(
                f"{name} must be a plain NumPy array, got {type(array).__name__}, whose semantics would be lost"
            )
