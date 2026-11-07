from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from interpolation1d import (
    AkimaInterpolator,
    CubicSplineInterpolator,
    Extrapolation,
    FilledSeries,
    Interpolator,
    LinearInterpolator,
    NearestInterpolator,
    NextInterpolator,
    PchipInterpolator,
    PreviousInterpolator,
    SplineBoundary,
)

INDICES: NDArray[np.int64] = np.array([0, 2, 5])
VALUES: NDArray[np.float64] = np.array([0.0, 2.0, 8.0])
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


class TestMethods(unittest.TestCase):
    def test_linear(self) -> None:
        filled: FilledSeries = LinearInterpolator().fill(INDICES, VALUES)
        np.testing.assert_allclose(filled.values, [0.0, 1.0, 2.0, 4.0, 6.0, 8.0])

    def test_nearest_prefers_the_earlier_observation_on_ties(self) -> None:
        filled: FilledSeries = NearestInterpolator().fill(INDICES, VALUES)
        np.testing.assert_array_equal(filled.values, [0.0, 0.0, 2.0, 2.0, 8.0, 8.0])

    def test_previous(self) -> None:
        filled: FilledSeries = PreviousInterpolator().fill(INDICES, VALUES)
        np.testing.assert_array_equal(filled.values, [0.0, 0.0, 2.0, 2.0, 2.0, 8.0])

    def test_next(self) -> None:
        filled: FilledSeries = NextInterpolator().fill(INDICES, VALUES)
        np.testing.assert_array_equal(filled.values, [0.0, 2.0, 2.0, 8.0, 8.0, 8.0])

    def test_cubic_spline_reproduces_a_cubic(self) -> None:
        indices: NDArray[np.int64] = np.array([0, 1, 4, 6, 9])
        cubic: NDArray[np.float64] = indices.astype(np.float64) ** 3
        filled: FilledSeries = CubicSplineInterpolator().fill(indices, cubic)
        np.testing.assert_allclose(filled.values, np.arange(10.0) ** 3, atol=1e-9)

    def test_spline_boundaries_differ(self) -> None:
        indices: NDArray[np.int64] = np.array([0, 3, 4, 8])
        values: NDArray[np.float64] = np.array([0.0, 3.0, 1.0, 5.0])
        estimates: list[NDArray[np.floating]] = [
            CubicSplineInterpolator(boundary=boundary).fill(indices, values).values
            for boundary in SplineBoundary
        ]
        for first, second in [(0, 1), (1, 2), (0, 2)]:
            self.assertFalse(np.allclose(estimates[first], estimates[second]))

    def test_pchip_does_not_overshoot_a_step(self) -> None:
        indices: NDArray[np.int64] = np.array([0, 3, 6, 9])
        step: NDArray[np.float64] = np.array([0.0, 0.0, 1.0, 1.0])
        pchip: FilledSeries = PchipInterpolator().fill(indices, step)
        spline: FilledSeries = CubicSplineInterpolator().fill(indices, step)
        self.assertTrue(((pchip.values >= 0.0) & (pchip.values <= 1.0)).all())
        self.assertTrue(np.all(np.diff(pchip.values) >= 0.0))
        self.assertLess(float(spline.values.min()), 0.0)

    def test_modified_akima_stays_flat_on_flat_data(self) -> None:
        indices: NDArray[np.int64] = np.arange(0, 16, 2)
        values: NDArray[np.float64] = np.array([1.0, 2.0, 3.0, 3.0, 3.0, 4.0, 5.0, 8.0])
        modified: FilledSeries = AkimaInterpolator().fill(indices, values)
        original: FilledSeries = AkimaInterpolator(is_modified=False).fill(
            indices, values
        )
        np.testing.assert_allclose(modified.values[4:9], 3.0)
        self.assertFalse(np.allclose(original.values[4:9], 3.0))

    def test_is_observed_marks_observations(self) -> None:
        for method in ALL_METHODS:
            with self.subTest(method=method):
                filled: FilledSeries = method.fill(INDICES, VALUES)
                np.testing.assert_array_equal(filled.indices, np.arange(6))
                np.testing.assert_array_equal(
                    filled.is_observed, [True, False, True, False, False, True]
                )
                np.testing.assert_array_equal(filled.values[filled.is_observed], VALUES)


class TestValueShapes(unittest.TestCase):
    def test_each_element_of_multidimensional_values_is_interpolated(self) -> None:
        values: NDArray[np.float64] = np.random.default_rng(0).random((3, 4, 2))
        for method in ALL_METHODS:
            with self.subTest(method=method):
                filled: FilledSeries = method.fill(INDICES, values)
                self.assertEqual(filled.values.shape, (6, 4, 2))
                for row in range(4):
                    for column in range(2):
                        np.testing.assert_allclose(
                            filled.values[:, row, column],
                            method.fill(INDICES, values[:, row, column]).values,
                        )

    def test_keeps_floating_dtypes(self) -> None:
        for dtype in (np.float16, np.float32, np.float64):
            for method in ALL_METHODS:
                with self.subTest(dtype=dtype, method=method):
                    filled: FilledSeries = method.fill(INDICES, VALUES.astype(dtype))
                    self.assertEqual(filled.values.dtype, np.dtype(dtype))

    def test_keeps_longdouble_but_estimates_in_float64(self) -> None:
        for method in ALL_METHODS:
            with self.subTest(method=method):
                filled: FilledSeries = method.fill(
                    INDICES, VALUES.astype(np.longdouble)
                )
                self.assertEqual(filled.values.dtype, np.dtype(np.longdouble))
                np.testing.assert_allclose(
                    filled.values.astype(np.float64),
                    method.fill(INDICES, VALUES).values,
                )

    def test_integers_become_float64(self) -> None:
        filled: FilledSeries = LinearInterpolator().fill(
            np.array([0, 2], dtype=np.uint8), np.array([0, 3], dtype=np.int8)
        )
        self.assertEqual(filled.values.dtype, np.float64)
        np.testing.assert_array_equal(filled.values, [0.0, 1.5, 3.0])

    def test_never_modifies_inputs(self) -> None:
        indices: NDArray[np.int64] = INDICES.copy()
        values: NDArray[np.float64] = VALUES.copy()
        for method in ALL_METHODS:
            for extrapolation in Extrapolation:
                if extrapolation is not Extrapolation.REJECT:
                    _ = replace(method, extrapolation=extrapolation).fill(
                        indices, values, span=range(-1, 7)
                    )
        np.testing.assert_array_equal(indices, INDICES)
        np.testing.assert_array_equal(values, VALUES)
        filled: FilledSeries = LinearInterpolator().fill(indices, values)
        filled.values[0] = 100.0
        np.testing.assert_array_equal(values, VALUES)


class TestSpan(unittest.TestCase):
    def test_a_step_resamples(self) -> None:
        filled: FilledSeries = LinearInterpolator().fill(
            INDICES, VALUES, span=range(1, 6, 2)
        )
        np.testing.assert_array_equal(filled.indices, [1, 3, 5])
        np.testing.assert_allclose(filled.values, [1.0, 4.0, 8.0])
        np.testing.assert_array_equal(filled.is_observed, [False, False, True])

    def test_a_span_inside_the_observations_needs_no_extrapolation(self) -> None:
        filled: FilledSeries = LinearInterpolator().fill(
            INDICES, VALUES, span=range(3, 5)
        )
        np.testing.assert_allclose(filled.values, [4.0, 6.0])

    def test_negative_indices(self) -> None:
        filled: FilledSeries = LinearInterpolator().fill(
            np.array([-4, -2]), np.array([1.0, 3.0])
        )
        np.testing.assert_array_equal(filled.indices, [-4, -3, -2])
        np.testing.assert_allclose(filled.values, [1.0, 2.0, 3.0])

    def test_a_span_entirely_outside_the_observations(self) -> None:
        for extrapolation in (Extrapolation.HOLD, Extrapolation.EXTEND):
            for span in (range(-5, -2), range(8, 11)):
                with self.subTest(extrapolation=extrapolation, span=span):
                    filled: FilledSeries = LinearInterpolator(
                        extrapolation=extrapolation
                    ).fill(
                        np.array([0, 5], dtype=np.uint8),
                        np.array([0.0, 5.0]),
                        span=span,
                    )
                    np.testing.assert_array_equal(filled.indices, list(span))
                    self.assertFalse(filled.is_observed.any())
                    expected: list[float] = (
                        [float(min(max(index, 0), 5)) for index in span]
                        if extrapolation is Extrapolation.HOLD
                        else [float(index) for index in span]
                    )
                    np.testing.assert_allclose(filled.values, expected)


class TestArrayLayouts(unittest.TestCase):
    def test_zero_size_value_shapes(self) -> None:
        for shape in ((3, 0), (3, 4, 0, 2)):
            for method in ALL_METHODS:
                with self.subTest(shape=shape, method=method):
                    filled: FilledSeries = method.fill(INDICES, np.zeros(shape))
                    self.assertEqual(filled.values.shape, (6, *shape[1:]))

    def test_read_only_strided_and_fortran_inputs(self) -> None:
        base: NDArray[np.float64] = np.random.default_rng(1).random((6, 2, 3))
        layouts: dict[str, NDArray[np.float64]] = {
            "strided": base[::2],
            "reversed": base[::-2][::-1],
            "fortran": np.asfortranarray(base[::2]),
        }
        read_only: NDArray[np.float64] = base[::2].copy()
        read_only.flags.writeable = False
        layouts["read-only"] = read_only
        strided_indices: NDArray[np.int64] = np.array([0, 9, 2, 9, 5])[::2]
        for name, values in layouts.items():
            for method in ALL_METHODS:
                with self.subTest(layout=name, method=method):
                    np.testing.assert_allclose(
                        method.fill(strided_indices, values).values,
                        method.fill(INDICES, np.ascontiguousarray(values)).values,
                    )


class TestExtrapolation(unittest.TestCase):
    SPAN: range = range(-2, 8)

    def test_reject_is_the_default(self) -> None:
        for method in ALL_METHODS:
            with self.subTest(method=method):
                self.assertIs(method.extrapolation, Extrapolation.REJECT)
                with self.assertRaisesRegex(ValueError, "beyond the observed"):
                    _ = method.fill(INDICES, VALUES, span=self.SPAN)

    def test_hold_repeats_the_edge_values(self) -> None:
        for method in ALL_METHODS:
            with self.subTest(method=method):
                filled: FilledSeries = replace(
                    method, extrapolation=Extrapolation.HOLD
                ).fill(INDICES, VALUES, span=self.SPAN)
                np.testing.assert_array_equal(filled.values[:2], [0.0, 0.0])
                np.testing.assert_array_equal(filled.values[-2:], [8.0, 8.0])
                self.assertFalse(filled.is_observed[:2].any())
                self.assertFalse(filled.is_observed[-2:].any())

    def test_nan_leaves_the_outside_blank(self) -> None:
        for method in ALL_METHODS:
            with self.subTest(method=method):
                filled: FilledSeries = replace(
                    method, extrapolation=Extrapolation.NAN
                ).fill(INDICES, VALUES, span=self.SPAN)
                is_nan: NDArray[np.bool_] = np.isnan(filled.values)
                np.testing.assert_array_equal(
                    is_nan, (filled.indices < 0) | (filled.indices > 5)
                )

    def test_extend_continues_a_line(self) -> None:
        filled: FilledSeries = LinearInterpolator(
            extrapolation=Extrapolation.EXTEND
        ).fill(INDICES, VALUES, span=self.SPAN)
        np.testing.assert_allclose(filled.values[:2], [-2.0, -1.0])
        np.testing.assert_allclose(filled.values[-2:], [10.0, 12.0])

    def test_extend_holds_for_step_methods(self) -> None:
        for method in (NearestInterpolator, PreviousInterpolator, NextInterpolator):
            with self.subTest(method=method):
                filled: FilledSeries = method(extrapolation=Extrapolation.EXTEND).fill(
                    INDICES, VALUES, span=self.SPAN
                )
                np.testing.assert_array_equal(filled.values[:2], [0.0, 0.0])
                np.testing.assert_array_equal(filled.values[-2:], [8.0, 8.0])

    def test_extend_continues_the_outermost_cubic(self) -> None:
        indices: NDArray[np.int64] = np.array([0, 1, 4, 6])
        cubic: NDArray[np.float64] = indices.astype(np.float64) ** 3
        filled: FilledSeries = CubicSplineInterpolator(
            extrapolation=Extrapolation.EXTEND
        ).fill(indices, cubic, span=range(-3, 10))
        np.testing.assert_allclose(filled.values, np.arange(-3.0, 10.0) ** 3)

    def test_extend_never_marks_the_outside_as_observed(self) -> None:
        for method in ALL_METHODS:
            with self.subTest(method=method):
                filled: FilledSeries = replace(
                    method, extrapolation=Extrapolation.EXTEND
                ).fill(INDICES, VALUES, span=self.SPAN)
                np.testing.assert_array_equal(
                    filled.is_observed, np.isin(filled.indices, INDICES)
                )

    def test_two_observations_give_a_line_for_every_smooth_method(self) -> None:
        for method in (
            LinearInterpolator(),
            CubicSplineInterpolator(),
            CubicSplineInterpolator(boundary=SplineBoundary.NATURAL),
            PchipInterpolator(),
            AkimaInterpolator(),
            AkimaInterpolator(is_modified=False),
        ):
            with self.subTest(method=method):
                filled: FilledSeries = replace(
                    method, extrapolation=Extrapolation.EXTEND
                ).fill(np.array([0, 4]), np.array([0.0, 4.0]), span=range(-2, 7))
                np.testing.assert_allclose(filled.values, np.arange(-2.0, 7.0))

    def test_nan_with_a_single_observation(self) -> None:
        for method in ALL_METHODS:
            with self.subTest(method=method):
                filled: FilledSeries = replace(
                    method, extrapolation=Extrapolation.NAN
                ).fill(np.array([3]), np.array([7.0]), span=range(5))
                np.testing.assert_array_equal(
                    filled.values, [np.nan, np.nan, np.nan, 7.0, np.nan]
                )

    def test_a_single_observation_can_be_held(self) -> None:
        for method in ALL_METHODS:
            with self.subTest(method=method):
                filled: FilledSeries = replace(
                    method, extrapolation=Extrapolation.HOLD
                ).fill(np.array([3]), np.array([7.0]), span=range(5))
                np.testing.assert_array_equal(filled.values, np.full(5, 7.0))
                np.testing.assert_array_equal(
                    filled.is_observed, [False, False, False, True, False]
                )
