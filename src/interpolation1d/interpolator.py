from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from .extrapolation import Extrapolation
from .filled_series import FilledSeries
from .series_validator import SeriesValidator
from .types import FloatArray, IndexArray, ValueArray


@dataclass(frozen=True, kw_only=True)
class Interpolator(ABC):
    """
    Base class of the methods that fill gaps between observations.

    Observed indices keep their observed values exactly, and only the others are
    estimated.

    Attributes
    ----------
    extrapolation : Extrapolation
        How to fill requested indices before the first or after the last
        observation.

    Raises
    ------
    TypeError
        If a field does not have its annotated type.
    """

    extrapolation: Extrapolation = Extrapolation.REJECT

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.extrapolation), Extrapolation):
            raise TypeError(
                f"extrapolation must be an Extrapolation, got {type(self.extrapolation).__name__}"
            )

    @property
    @abstractmethod
    def minimum_observations(self) -> int:
        """
        Get the number of observations needed to estimate a missing index.

        Returns
        -------
        int
            The number of observations.
        """

    def fill(
        self, indices: IndexArray, values: ValueArray, *, span: range | None = None
    ) -> FilledSeries:
        """
        Give every index in a span a value, estimating those not observed.

        Parameters
        ----------
        indices : IndexArray
            The index of each observation, such as a frame number, strictly
            increasing.
        values : ValueArray
            The observed values with shape ``(len(indices), *value_shape)``.
        span : range | None
            The indices to fill, increasing. None means every index from the first
            to the last observation.

        Returns
        -------
        FilledSeries
            The value at every index in the span.

        Raises
        ------
        TypeError
            If the indices are not integers, the values are not numeric, or the span
            is not a range.
        ValueError
            If the inputs are malformed, the span reaches beyond the observations
            with ``Extrapolation.REJECT``, or a missing index must be estimated from
            fewer observations than ``minimum_observations``.
        OverflowError
            If consecutive values differ by more than float64 can represent, or an
            estimate exceeds the range of the output dtype, as cubic overshoot or
            extension can with float16.
        """
        SeriesValidator.require_indices(indices)
        SeriesValidator.require_values(values, n_observations=len(indices))
        if span is not None:
            SeriesValidator.require_span(span)

        first: int = int(indices.min())
        last: int = int(indices.max())
        requested: range = range(first, last + 1) if span is None else span

        observed: NDArray[np.int64] = indices.astype(np.int64, copy=False)
        queries: NDArray[np.int64] = np.arange(
            requested.start, requested.stop, requested.step, dtype=np.int64
        )
        is_outside: NDArray[np.bool_] = (queries < first) | (queries > last)
        positions: NDArray[np.int64] = queries
        is_blank: NDArray[np.bool_] = np.zeros_like(is_outside)
        match self.extrapolation:
            case Extrapolation.REJECT:
                if is_outside.any():
                    raise ValueError(
                        f"span {requested} reaches beyond the observed indices [{first}, {last}]; choose an extrapolation other than REJECT"
                    )
            case Extrapolation.HOLD:
                positions = np.clip(queries, first, last)
            case Extrapolation.EXTEND:
                pass
            case Extrapolation.NAN:
                is_blank = is_outside

        located: NDArray[np.intp] = np.minimum(
            np.searchsorted(observed, positions), observed.size - 1
        )
        is_exact: NDArray[np.bool_] = np.equal(observed[located], positions)
        is_estimated: NDArray[np.bool_] = ~is_exact & ~is_blank

        output_dtype: np.dtype[np.floating] = (
            # The branch narrows the dtype, which the type checkers cannot follow.
            cast("np.dtype[np.floating]", values.dtype)
            if np.issubdtype(values.dtype, np.floating)
            else np.dtype(np.float64)
        )
        filled: FloatArray = np.empty(
            (queries.size, *values.shape[1:]), dtype=output_dtype
        )
        filled[is_exact] = values[located[is_exact]]
        filled[is_blank] = np.nan
        if is_estimated.any():
            if observed.size < self.minimum_observations:
                raise ValueError(
                    f"{type(self).__name__} needs at least {self.minimum_observations} observations to estimate a missing index, got {observed.size}"
                )
            # SciPy solves only in float32 and float64, so longdouble is narrowed too. The copy is
            # required: where longdouble is 64-bit, astype(copy=False) would keep its dtype.
            # Intermediate overflow is judged by whether the final estimates are finite.
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                estimates: FloatArray = self._estimate(
                    observed.astype(np.float64),
                    values.astype(np.float64),
                    positions[is_estimated].astype(np.float64),
                ).astype(output_dtype, copy=False)
            if not np.isfinite(estimates).all():
                raise OverflowError(
                    f"{type(self).__name__} produced estimates beyond the range of {output_dtype}; scale the values, or pass them in a wider dtype"
                )
            filled[is_estimated] = estimates
        return FilledSeries(
            indices=queries, values=filled, is_observed=is_exact & ~is_outside
        )

    @abstractmethod
    def _estimate(
        self, positions: FloatArray, values: FloatArray, queries: FloatArray
    ) -> FloatArray:
        """
        Estimate values at unobserved positions.

        Parameters
        ----------
        positions : FloatArray
            The observed positions, strictly increasing, with at least
            ``minimum_observations`` entries.
        values : FloatArray
            The observed values with shape ``(len(positions), *value_shape)``.
        queries : FloatArray
            The positions to estimate, none of them observed. They lie outside the
            observed range only with ``Extrapolation.EXTEND``.

        Returns
        -------
        FloatArray
            The estimates with shape ``(len(queries), *value_shape)``.
        """
