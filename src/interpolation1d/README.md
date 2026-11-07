# interpolation1d

## Overview

Gap filling for series observed at irregular integer indices. Each `Interpolator` subclass is one method, and its `fill` returns a `FilledSeries` with a value at every requested index.

## Components

| Component | Description |
|-----------|-------------|
| [types.py](./types.py) | Type aliases for observation indices and values. |
| [extrapolation.py](./extrapolation.py) | `Extrapolation`, how to fill indices outside the observations. |
| [spline_boundary.py](./spline_boundary.py) | `SplineBoundary`, the end condition of a cubic spline. |
| [series_validator.py](./series_validator.py) | `SeriesValidator`, runtime checks of observed series and spans. |
| [filled_series.py](./filled_series.py) | `FilledSeries`, the result of filling. |
| [interpolator.py](./interpolator.py) | `Interpolator`, the abstract base class of the methods. |
| [interpolators/](./interpolators/README.md) | One class per interpolation method. |
