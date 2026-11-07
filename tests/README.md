# Tests

## Overview

Unit tests for the `interpolation1d` package.

## Components

| Component | Description |
| --- | --- |
| [`test_interpolators.py`](test_interpolators.py) | Tests for the result of each method, value shapes and dtypes, spans, and extrapolation. |
| [`test_validation.py`](test_validation.py) | Tests that malformed indices, values, spans, and fields are rejected with clear errors. |
| [`test_properties.py`](test_properties.py) | Properties of all methods over generated series, such as that observations are kept exactly and lines are reproduced. |

## Examples

```bash
uv run pytest
```
