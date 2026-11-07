from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hypothesis_numpy
from numpy.typing import NDArray

from interpolation1d import (
    AkimaInterpolator,
    CubicSplineInterpolator,
    Extrapolation,
    FilledSeries,
    FloatArray,
    Interpolator,
    LinearInterpolator,
    NearestInterpolator,
    NextInterpolator,
    PchipInterpolator,
    PreviousInterpolator,
    SplineBoundary,
)

ALL_METHODS: tuple[Interpolator, ...] = (
    LinearInterpolator(),
    NearestInterpolator(),
    PreviousInterpolator(),
    NextInterpolator(),
    CubicSplineInterpolator(),
    CubicSplineInterpolator(boundary=SplineBoundary.NATURAL),
    CubicSplineInterpolator(boundary=SplineBoundary.CLAMPED),
    PchipInterpolator(),
    AkimaInterpolator(),
    AkimaInterpolator(is_modified=False),
)
STEP_METHODS: tuple[Interpolator, ...] = ALL_METHODS[1:4]
# A clamped spline forces a zero slope at the ends, so it alone cannot reproduce a line.
LINE_REPRODUCING_METHODS: tuple[Interpolator, ...] = tuple(
    method
    for method in ALL_METHODS
    if method not in STEP_METHODS
    and method != CubicSplineInterpolator(boundary=SplineBoundary.CLAMPED)
)


def observed_span(indices: NDArray[np.int64], margin: int = 0) -> range:
    return range(int(indices.min()) - margin, int(indices.max()) + 1 + margin)


REALISTIC_FLOATS: st.SearchStrategy[float] = st.floats(-1e6, 1e6)
PROFILE = settings(max_examples=150, deadline=None, derandomize=True, database=None)


@st.composite
def series(
    draw: st.DrawFn, *, minimum_size: int = 2
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """
    Generate irregular observation indices with 2-D values.

    Returns
    -------
    tuple[NDArray[np.int64], NDArray[np.float64]]
        The indices and values.
    """
    gaps: list[int] = draw(
        st.lists(
            st.integers(min_value=1, max_value=20),
            min_size=minimum_size - 1,
            max_size=15,
        )
    )
    first: int = draw(st.integers(min_value=-1000, max_value=1000))
    indices: NDArray[np.int64] = first + np.concatenate(
        [[0], np.cumsum(gaps, dtype=np.int64)]
    )
    values: NDArray[np.float64] = draw(
        hypothesis_numpy.arrays(
            np.float64,
            (indices.size, 2),
            elements=REALISTIC_FLOATS,
        )
    )
    return indices, values


class TestProperties(unittest.TestCase):
    @PROFILE
    @given(series(minimum_size=1))
    def test_observations_are_kept_exactly(
        self, observed: tuple[NDArray[np.int64], NDArray[np.float64]]
    ) -> None:
        indices, values = observed
        for method in ALL_METHODS:
            filled: FilledSeries = replace(
                method, extrapolation=Extrapolation.HOLD
            ).fill(indices, values)
            np.testing.assert_array_equal(filled.indices, list(observed_span(indices)))
            np.testing.assert_array_equal(
                filled.is_observed, np.isin(filled.indices, indices)
            )
            np.testing.assert_array_equal(filled.values[filled.is_observed], values)
            self.assertTrue(np.isfinite(filled.values).all())

    @PROFILE
    @given(
        series(),
        st.floats(-100.0, 100.0).filter(
            lambda slope: slope == 0.0 or abs(slope) > 1e-6
        ),
        st.floats(-1e4, 1e4),
    )
    def test_lines_are_reproduced(
        self,
        observed: tuple[NDArray[np.int64], NDArray[np.float64]],
        slope: float,
        intercept: float,
    ) -> None:
        indices, _ = observed
        line: FloatArray = slope * indices + intercept
        span: range = observed_span(indices, margin=5)
        expected: FloatArray = slope * np.arange(span.start, span.stop) + intercept
        for method in LINE_REPRODUCING_METHODS:
            filled: FilledSeries = replace(
                method, extrapolation=Extrapolation.EXTEND
            ).fill(indices, line, span=span)
            np.testing.assert_allclose(filled.values, expected, rtol=1e-7, atol=1e-6)

    @PROFILE
    @given(series())
    def test_step_methods_only_copy_observations(
        self, observed: tuple[NDArray[np.int64], NDArray[np.float64]]
    ) -> None:
        indices, values = observed
        span: range = observed_span(indices, margin=3)
        queries: NDArray[np.int64] = np.arange(span.start, span.stop)
        previous: NDArray[np.intp] = np.clip(
            np.searchsorted(indices, queries, side="right") - 1, 0, indices.size - 1
        )
        following: NDArray[np.intp] = np.clip(
            np.searchsorted(indices, queries), 0, indices.size - 1
        )
        np.testing.assert_array_equal(
            PreviousInterpolator(extrapolation=Extrapolation.EXTEND)
            .fill(indices, values, span=span)
            .values,
            values[previous],
        )
        np.testing.assert_array_equal(
            NextInterpolator(extrapolation=Extrapolation.EXTEND)
            .fill(indices, values, span=span)
            .values,
            values[following],
        )
        nearest: FloatArray = (
            NearestInterpolator(extrapolation=Extrapolation.EXTEND)
            .fill(indices, values, span=span)
            .values
        )
        distance: NDArray[np.int64] = np.abs(
            queries[:, np.newaxis] - indices[np.newaxis, :]
        )
        np.testing.assert_array_equal(nearest, values[np.argmin(distance, axis=1)])

    @PROFILE
    @given(series())
    def test_linear_and_pchip_stay_between_neighbors(
        self, observed: tuple[NDArray[np.int64], NDArray[np.float64]]
    ) -> None:
        indices, values = observed
        span: range = observed_span(indices)
        queries: NDArray[np.int64] = np.arange(span.start, span.stop)
        segment: NDArray[np.intp] = np.clip(
            np.searchsorted(indices, queries, side="right") - 1, 0, indices.size - 2
        )
        lower: NDArray[np.float64] = np.minimum(values[segment], values[segment + 1])
        upper: NDArray[np.float64] = np.maximum(values[segment], values[segment + 1])
        tolerance: FloatArray = 1e-9 * (np.maximum(np.abs(lower), np.abs(upper)) + 1.0)
        for method in (LinearInterpolator(), PchipInterpolator()):
            filled: FilledSeries = method.fill(indices, values)
            self.assertTrue((filled.values >= lower - tolerance).all())
            self.assertTrue((filled.values <= upper + tolerance).all())
