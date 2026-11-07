# interpolators

## Overview

The interpolation methods, each a frozen dataclass that validates its fields on construction.

## Components

| Component | Description |
|-----------|-------------|
| [linear.py](./linear.py) | `LinearInterpolator`, a line between neighbors. |
| [nearest.py](./nearest.py) | `NearestInterpolator`, the closest observation. |
| [previous.py](./previous.py) | `PreviousInterpolator`, the latest earlier observation. |
| [next.py](./next.py) | `NextInterpolator`, the earliest later observation. |
| [cubic_spline.py](./cubic_spline.py) | `CubicSplineInterpolator`, a smooth cubic spline. |
| [pchip.py](./pchip.py) | `PchipInterpolator`, a shape-preserving cubic. |
| [akima.py](./akima.py) | `AkimaInterpolator`, a cubic with locally estimated slopes. |
