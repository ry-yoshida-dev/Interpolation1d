# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "interpolation-1d",
#     "matplotlib-utilities",
#     "numpy>=2.2",
# ]
#
# [tool.uv]
# exclude-newer = "7 days"
#
# [tool.uv.sources]
# interpolation-1d = { path = "..", editable = true }
# matplotlib-utilities = { git = "https://github.com/ry-yoshida-dev/MatplotlibUtilities" }
# ///
"""
Draw the README figure that compares the interpolation methods.

Run ``uv run docs/draw_methods_figure.py`` from the repository root.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib_utilities import (
    GraphAxis,
    GraphLayout,
    GraphParameters,
    LegendLocation,
    LegendParameters,
    Linestyle,
    MatplotGraphMaker,
    PlotParameters,
    ScatterParameters,
)
from numpy.typing import NDArray

from interpolation1d import (
    AkimaInterpolator,
    CubicSplineInterpolator,
    Interpolator,
    LinearInterpolator,
    NearestInterpolator,
    NextInterpolator,
    PchipInterpolator,
    PreviousInterpolator,
)

INK: str = "#0b0b0b"
MUTED: str = "#8a8984"
FILLED: str = "#2a78d6"

FloatSeries = NDArray[np.float64]
Signal = Callable[[FloatSeries], FloatSeries]


@dataclass(frozen=True)
class MethodComparisonFigure:
    """
    A grid with one row per method and one column per situation that separates them.

    Each column is built to expose one property:

    - Step in a gap: the signal jumps between two distant observations, so the step
      methods switch at different frames (Previous late, Next early, Nearest at the
      midpoint), the cubic spline overshoots on both sides, Akima overshoots
      mildly, and PCHIP does not.
    - Peak in a gap: the top of a bump is not observed, so the cubic spline recovers
      it, while PCHIP stops at the higher neighboring observation and Linear cuts
      across it.
    - Spike: the signal rises sharply at one observed frame and is flat elsewhere,
      so the cubic spline ripples far from it, while PCHIP and Akima stay flat
      beyond the neighboring observations.

    Attributes
    ----------
    span : range
        The frames shown in every column.
    value_limits : tuple[float, float]
        The y range shared by every panel.
    """

    span: range = range(0, 41)
    value_limits: tuple[float, float] = (-0.35, 1.35)

    def draw(self, save_path: Path) -> None:
        """
        Draw the figure and save it.

        Parameters
        ----------
        save_path : Path
            Where to write the image.
        """
        methods: list[tuple[str, Interpolator]] = [
            ("Linear", LinearInterpolator()),
            ("Nearest", NearestInterpolator()),
            ("Previous", PreviousInterpolator()),
            ("Next", NextInterpolator()),
            ("Cubic spline", CubicSplineInterpolator()),
            ("PCHIP", PchipInterpolator()),
            ("Akima", AkimaInterpolator()),
        ]
        situations: list[tuple[str, NDArray[np.int64], FloatSeries, Signal]] = [
            self._step_in_gap(),
            self._peak_in_gap(),
            self._spike(),
        ]
        maker: MatplotGraphMaker = MatplotGraphMaker(
            layout=GraphLayout.from_row_column(
                row=len(methods), column=len(situations)
            ),
            parameters=GraphParameters(
                figsize=(12.0, 14.0), dpi=110, font_size=11, w_space=0.15, h_space=0.25
            ),
        )
        fine: FloatSeries = np.linspace(self.span.start, self.span.stop - 1, 800)

        for row, (method_name, method) in enumerate(methods):
            for column, (situation_name, observed, values, signal) in enumerate(
                situations
            ):
                number: int = row * len(situations) + column
                filled = method.fill(observed, values, span=self.span)
                maker.plot(
                    x=fine,
                    y=signal(fine),
                    index=number,
                    subparams=PlotParameters(
                        color=MUTED,
                        linewidth=1.2,
                        linestyle=Linestyle.DASHED,
                        label="True signal",
                        zorder=1,
                    ),
                )
                maker.plot(
                    x=filled.indices,
                    y=filled.values,
                    index=number,
                    subparams=PlotParameters(
                        color=FILLED, linewidth=2.0, label="Filled values", zorder=2
                    ),
                )
                maker.scatter(
                    x=observed,
                    y=values,
                    index=number,
                    subparams=ScatterParameters(
                        c=INK, s=28, label="Observations", zorder=3
                    ),
                )
                maker.set_lim(
                    self.span.start - 0.5,
                    self.span.stop - 0.5,
                    axis=GraphAxis.X,
                    index=number,
                )
                maker.set_lim(*self.value_limits, axis=GraphAxis.Y, index=number)
                if row == 0:
                    maker.set_title(title=situation_name, index=number)
                if column == 0:
                    maker.set_label(label=method_name, axis=GraphAxis.Y, index=number)
                if row == len(methods) - 1:
                    maker.set_label(label="Frame", axis=GraphAxis.X, index=number)

        # One legend for the whole grid, centered below the middle column.
        maker.legend(
            index=len(methods) * len(situations) - 2,
            subparams=LegendParameters(
                loc=LegendLocation.UPPER_CENTER,
                bbox_to_anchor=(0.5, -0.45),
                ncol=3,
                frameon=False,
                fontsize=12,
            ),
        )
        maker.finalize(save_path=str(save_path), is_showing_result_enabled=False)

    def _step_in_gap(self) -> tuple[str, NDArray[np.int64], FloatSeries, Signal]:
        """
        Build a step at frame 25, between the observations at 20 and 30.
        """

        def signal(frames: FloatSeries) -> FloatSeries:
            return np.where(frames >= 24.5, 1.0, 0.0)

        observed: NDArray[np.int64] = np.array([0, 10, 20, 30, 40], dtype=np.int64)
        return "Step in a gap", observed, signal(observed.astype(np.float64)), signal

    def _peak_in_gap(self) -> tuple[str, NDArray[np.int64], FloatSeries, Signal]:
        """
        Build a bump peaking at frame 20, between the observations at 15 and 24.
        """

        def signal(frames: FloatSeries) -> FloatSeries:
            return np.exp(-(((frames - 20.0) / 6.0) ** 2))

        observed: NDArray[np.int64] = np.array(
            [0, 3, 6, 9, 12, 15, 24, 27, 30, 33, 36, 40], dtype=np.int64
        )
        return "Peak in a gap", observed, signal(observed.astype(np.float64)), signal

    def _spike(self) -> tuple[str, NDArray[np.int64], FloatSeries, Signal]:
        """
        Build a narrow spike at frame 20 on a flat signal observed every 4 frames.
        """

        def signal(frames: FloatSeries) -> FloatSeries:
            return np.exp(-(((frames - 20.0) / 1.2) ** 2))

        observed: NDArray[np.int64] = np.arange(0, 41, 4, dtype=np.int64)
        return "Spike", observed, signal(observed.astype(np.float64)), signal


if __name__ == "__main__":
    output: Path = Path(__file__).resolve().parent / "methods.png"
    MethodComparisonFigure().draw(output)
