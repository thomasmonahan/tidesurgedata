"""Structural tests for additive Dynamical 2D field functionality.

These tests deliberately avoid network access. They verify that the new
multi-field API is additive to the existing Dynamical point and forecast
interfaces and that argument validation which can occur before provider
access behaves correctly.

Live retrieval from dynamical.org belongs in the live/provider integration
tests rather than this structural test module.
"""

from __future__ import annotations

import inspect

import pandas as pd
import pytest

from tidesurgedata.sources.dynamical import Dynamical


@pytest.fixture()
def source() -> Dynamical:
    """Construct a Dynamical source without performing network I/O."""
    return Dynamical(
        "example",
        "pressure_surface",
        lat=40.0,
        lon=-74.0,
    )


def test_fetch_fields_is_additive_to_existing_api(source: Dynamical) -> None:
    """Adding spatial fields must not replace the existing source APIs."""
    assert callable(source.fetch)
    assert callable(source.fetch_forecast)
    assert callable(source.init_times)

    assert callable(source.fetch_fields)


def test_existing_source_configuration_is_unchanged(source: Dynamical) -> None:
    """Spatial-field support must not change normal Dynamical construction."""
    assert source.dataset == "example"
    assert source.variable == "pressure_surface"
    assert source.lat == 40.0
    assert source.lon == -74.0
    assert source.method == "nearest"

    assert isinstance(source.latency, pd.Timedelta)


def test_fetch_fields_has_explicit_spatial_bounds(source: Dynamical) -> None:
    """The multi-field API should expose its geographic bounds explicitly."""
    signature = inspect.signature(source.fetch_fields)

    assert "variables" in signature.parameters
    assert "start" in signature.parameters
    assert "end" in signature.parameters
    assert "lat_min" in signature.parameters
    assert "lat_max" in signature.parameters
    assert "lon_min" in signature.parameters
    assert "lon_max" in signature.parameters


def test_fetch_fields_spatial_bounds_are_keyword_only(source: Dynamical) -> None:
    """Bounds should not be accidentally confused with time/variable arguments."""
    signature = inspect.signature(source.fetch_fields)

    for name in ("lat_min", "lat_max", "lon_min", "lon_max"):
        assert (
            signature.parameters[name].kind
            is inspect.Parameter.KEYWORD_ONLY
        )


def test_fetch_fields_rejects_empty_variables_before_network_access(
    source: Dynamical,
) -> None:
    """An empty variable request is invalid without consulting the provider."""
    with pytest.raises(ValueError):
        source.fetch_fields(
            [],
            "2024-01-01T00:00Z",
            "2024-01-02T00:00Z",
            lat_min=39.0,
            lat_max=42.0,
            lon_min=-76.0,
            lon_max=-72.0,
        )


@pytest.mark.parametrize(
    ("lat_min", "lat_max", "lon_min", "lon_max"),
    [
        (42.0, 39.0, -76.0, -72.0),   # reversed latitude
        (-91.0, 42.0, -76.0, -72.0),  # latitude below -90
        (39.0, 91.0, -76.0, -72.0),   # latitude above 90
        (39.0, 42.0, -181.0, -72.0),  # longitude below -180
        (39.0, 42.0, -76.0, 181.0),   # longitude above 180
        (39.0, 42.0, -72.0, -76.0),   # reversed longitude
    ],
)
def test_fetch_fields_rejects_invalid_spatial_bounds_before_network_access(
    source: Dynamical,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> None:
    """Obviously invalid geographic boxes should fail without provider I/O."""
    with pytest.raises(ValueError):
        source.fetch_fields(
            ["pressure_surface"],
            "2024-01-01T00:00Z",
            "2024-01-02T00:00Z",
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
        )


def test_fetch_fields_does_not_mutate_source(source: Dynamical) -> None:
    """Dynamical remains a frozen point-source configuration."""
    before = (
        source.dataset,
        source.variable,
        source.lat,
        source.lon,
        source.method,
    )

    # We deliberately trigger validation before any network request.
    with pytest.raises(ValueError):
        source.fetch_fields(
            [],
            "2024-01-01T00:00Z",
            "2024-01-02T00:00Z",
            lat_min=39.0,
            lat_max=42.0,
            lon_min=-76.0,
            lon_max=-72.0,
        )

    after = (
        source.dataset,
        source.variable,
        source.lat,
        source.lon,
        source.method,
    )

    assert after == before