"""Structural tests for additive Dynamical 2D field helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from tidesurgedata.sources.dynamical import Dynamical, _normalise_field_bounds


def test_field_bounds_validate_and_normalise() -> None:
    assert _normalise_field_bounds(39, 42, -76, -72) == (39.0, 42.0, -76.0, -72.0)


@pytest.mark.parametrize(
    "bounds",
    [
        (42, 39, -76, -72),
        (-91, 42, -76, -72),
        (39, 42, -181, -72),
        (39, 42, -72, -76),
    ],
)
def test_field_bounds_reject_invalid_boxes(bounds) -> None:
    with pytest.raises(ValueError):
        _normalise_field_bounds(*bounds)


def test_spatial_methods_are_additive_to_existing_point_api() -> None:
    source = Dynamical("example", "pressure_surface", lat=40.0, lon=-74.0)
    assert callable(source.fetch)
    assert callable(source.fetch_forecast)
    assert callable(source.fetch_field)
    assert callable(source.fetch_wind_field)
    assert source.lat == 40.0
    assert source.lon == -74.0
    assert source.variable == "pressure_surface"
    assert isinstance(source.latency, pd.Timedelta)
