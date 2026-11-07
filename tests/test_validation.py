from __future__ import annotations

import unittest
from typing import cast

import numpy as np
from numpy.typing import NDArray

from interpolation1d import (
    AkimaInterpolator,
    CubicSplineInterpolator,
    Extrapolation,
    FilledSeries,
    IndexArray,
    LinearInterpolator,
    NearestInterpolator,
    PchipInterpolator,
    SplineBoundary,
    ValueArray,
)

INDICES: NDArray[np.int64] = np.array([0, 2, 5])
VALUES: NDArray[np.float64] = np.array([0.0, 2.0, 8.0])


class TestIndices(unittest.TestCase):
    def test_rejects_what_is_not_an_integer_array(self) -> None:
        for indices in (
            cast(IndexArray, cast(object, [0, 1, 2])),
            cast(IndexArray, np.array([0.0, 1.0, 2.0])),
            cast(IndexArray, np.array([False, True, True])),
            np.ma.masked_array(INDICES),
        ):
            with self.subTest(indices=indices), self.assertRaises(TypeError):
                _ = LinearInterpolator().fill(indices, VALUES)

    def test_rejects_malformed_indices(self) -> None:
        cases: dict[str, IndexArray] = {
            "1-D": INDICES.reshape(1, 3),
            "empty": np.array([], dtype=np.int64),
            "strictly increasing": np.array([0, 5, 2]),
        }
        for message, indices in cases.items():
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                _ = LinearInterpolator().fill(indices, VALUES[: indices.size])

    def test_names_the_first_disorder(self) -> None:
        with self.assertRaisesRegex(ValueError, r"\[3 3\] at positions 1 and 2"):
            _ = LinearInterpolator().fill(np.array([0, 3, 3]), VALUES)

    def test_rejects_indices_that_float64_cannot_represent(self) -> None:
        for indices in (
            np.array([0, 2**53 + 1], dtype=np.int64),
            np.array([0, 2**63], dtype=np.uint64),
        ):
            with (
                self.subTest(indices=indices),
                self.assertRaisesRegex(ValueError, "2\\*\\*53"),
            ):
                _ = LinearInterpolator().fill(indices, VALUES[:2])

    def test_large_unsigned_indices_are_not_wrapped(self) -> None:
        indices: NDArray[np.uint64] = np.array([2**53 - 2, 2**53], dtype=np.uint64)
        filled: FilledSeries = LinearInterpolator().fill(indices, np.array([0.0, 2.0]))
        np.testing.assert_array_equal(filled.indices, [2**53 - 2, 2**53 - 1, 2**53])
        np.testing.assert_allclose(filled.values, [0.0, 1.0, 2.0])


class TestValues(unittest.TestCase):
    def test_rejects_what_is_not_a_numeric_array(self) -> None:
        for values in (
            cast(ValueArray, cast(object, [0.0, 2.0, 8.0])),
            cast(ValueArray, np.array([True, False, True])),
            cast(ValueArray, np.array([1j, 2j, 3j])),
            np.ma.masked_array(VALUES),
        ):
            with self.subTest(values=values), self.assertRaises(TypeError):
                _ = LinearInterpolator().fill(INDICES, values)

    def test_rejects_a_mismatched_length(self) -> None:
        for values in (VALUES[:2], np.zeros((4, 2)), np.array(1.0)):
            with (
                self.subTest(shape=values.shape),
                self.assertRaisesRegex(ValueError, "3 entries"),
            ):
                _ = LinearInterpolator().fill(INDICES, values)

    def test_rejects_integers_that_float64_cannot_represent(self) -> None:
        for values in (
            np.array([0, 2**53 + 1, 0], dtype=np.int64),
            np.array([0, -(2**53) - 1, 0], dtype=np.int64),
            np.array([0, 2**64 - 1, 0], dtype=np.uint64),
        ):
            with (
                self.subTest(values=values),
                self.assertRaisesRegex(ValueError, "2\\*\\*53"),
            ):
                _ = LinearInterpolator().fill(INDICES, values)

    def test_rejects_non_finite_values(self) -> None:
        for bad in (np.nan, np.inf, -np.inf):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "finite"):
                _ = LinearInterpolator().fill(INDICES, np.array([0.0, bad, 8.0]))


class TestSpanValidation(unittest.TestCase):
    def test_rejects_malformed_spans(self) -> None:
        cases: dict[str, range] = {
            "positive step": range(5, 0, -1),
            "empty": range(3, 3),
            "2\\*\\*53": range(0, 2**53 + 2),
        }
        for message, span in cases.items():
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                _ = LinearInterpolator(extrapolation=Extrapolation.HOLD).fill(
                    INDICES, VALUES, span=span
                )

    def test_rejects_what_is_not_a_range(self) -> None:
        for span in ([0, 1, 2], np.arange(3)):
            with (
                self.subTest(span=span),
                self.assertRaisesRegex(TypeError, "span must be a range"),
            ):
                _ = LinearInterpolator().fill(
                    INDICES, VALUES, span=cast(range, cast(object, span))
                )


class TestMinimumObservations(unittest.TestCase):
    def test_methods_that_need_two_observations_say_so(self) -> None:
        for method in (
            LinearInterpolator(extrapolation=Extrapolation.EXTEND),
            CubicSplineInterpolator(extrapolation=Extrapolation.EXTEND),
            PchipInterpolator(extrapolation=Extrapolation.EXTEND),
            AkimaInterpolator(extrapolation=Extrapolation.EXTEND),
        ):
            with (
                self.subTest(method=method),
                self.assertRaisesRegex(ValueError, f"{type(method).__name__} needs"),
            ):
                _ = method.fill(np.array([3]), np.array([7.0]), span=range(5))

    def test_a_single_observation_is_enough_for_a_step_method(self) -> None:
        filled: FilledSeries = NearestInterpolator(
            extrapolation=Extrapolation.EXTEND
        ).fill(np.array([3]), np.array([7.0]), span=range(5))
        np.testing.assert_array_equal(filled.values, np.full(5, 7.0))

    def test_a_single_observation_without_a_span_needs_no_estimate(self) -> None:
        filled: FilledSeries = CubicSplineInterpolator().fill(
            np.array([3]), np.array([[7.0, 1.0]])
        )
        np.testing.assert_array_equal(filled.values, [[7.0, 1.0]])


class TestFields(unittest.TestCase):
    def test_rejects_fields_of_the_wrong_type(self) -> None:
        with self.assertRaisesRegex(TypeError, "extrapolation"):
            _ = LinearInterpolator(
                extrapolation=cast(Extrapolation, cast(object, "hold"))
            )
        with self.assertRaisesRegex(TypeError, "boundary"):
            _ = CubicSplineInterpolator(
                boundary=cast(SplineBoundary, cast(object, "natural"))
            )
        with self.assertRaisesRegex(TypeError, "is_modified"):
            _ = AkimaInterpolator(is_modified=cast(bool, cast(object, 1)))


class TestOverflow(unittest.TestCase):
    def test_estimates_beyond_the_output_dtype_are_rejected(self) -> None:
        indices: NDArray[np.int64] = np.array([0, 1, 9, 10])
        values: NDArray[np.float16] = np.array([0, 60000, 60000, 0], dtype=np.float16)
        with self.assertRaisesRegex(OverflowError, "float16"):
            _ = CubicSplineInterpolator().fill(indices, values)

    def test_extension_beyond_float16_is_rejected(self) -> None:
        with self.assertRaisesRegex(OverflowError, "CubicSplineInterpolator"):
            _ = CubicSplineInterpolator(extrapolation=Extrapolation.EXTEND).fill(
                np.array([0, 1, 2]),
                np.array([0, 1, 8], dtype=np.float16),
                span=range(200),
            )

    def test_differences_beyond_float64_are_rejected_for_every_method(self) -> None:
        for method in (
            LinearInterpolator(),
            NearestInterpolator(),
            PchipInterpolator(),
        ):
            with (
                self.subTest(method=method),
                self.assertRaisesRegex(OverflowError, "differ"),
            ):
                _ = method.fill(np.array([0, 2]), np.array([-1e308, 1e308]))

    def test_tiny_values_are_estimated_without_warnings(self) -> None:
        filled: FilledSeries = PchipInterpolator().fill(
            np.array([0, 2, 5, 7]), np.array([0.0, 1e-310, 0.0, 1e-300])
        )
        self.assertTrue(np.isfinite(filled.values).all())
