"""Offline tests for the explorer's pure selection/recipe layer."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from apps.explorer.data import build_recipe, discover_us_sources, nearest_station
from tidesurgedata.meta import SeriesMeta


def _meta(source: str, station: str, variable: str, lat: float, lon: float) -> SeriesMeta:
    return SeriesMeta(
        source=source,
        station_id=station,
        variable=variable,
        lat=lat,
        lon=lon,
        units="m" if variable == "water_level" else "m3 s-1",
        datum="MSL" if variable == "water_level" else None,
        sampling="instantaneous",
        window=None,
        label=None,
        licence="test",
        attribution="test",
        url="https://example.test",
        name=station,
        extra={},
    )


def test_nearest_station_does_not_depend_on_provider_order() -> None:
    far = _meta("test", "far", "water_level", 40.0, -75.0)
    near = _meta("test", "near", "water_level", 40.70, -74.01)
    assert nearest_station([far, near], 40.71, -74.00, "test").station_id == "near"


def test_discovery_uses_provider_find_stations_and_recipe_uses_clicked_coordinates() -> None:
    noaa = _meta("noaa_coops", "NOAA1", "water_level", 40.70, -74.01)
    usgs = _meta("usgs", "USGS1", "discharge", 40.75, -74.05)

    with (
        patch("apps.explorer.data.NOAACoops.find_stations", return_value=[noaa]) as find_noaa,
        patch("apps.explorer.data.USGS.find_stations", return_value=[usgs]) as find_usgs,
    ):
        selection = discover_us_sources(40.71, -74.00, noaa_radius_km=25, usgs_radius_km=80)

    find_noaa.assert_called_once_with(40.71, -74.00, 25, variable="water_level")
    find_usgs.assert_called_once_with(40.71, -74.00, 80, variable="discharge")

    recipe = build_recipe(selection)
    assert recipe.target.station_id == "NOAA1"
    drivers = {driver.name: driver for driver in recipe.drivers}
    assert drivers["discharge"].source.site_id == "USGS1"
    assert all(lag < 0 for lag in drivers["discharge"].lags_hours)
    assert drivers["pressure"].source.lat == pytest.approx(40.71)
    assert drivers["pressure"].source.lon == pytest.approx(-74.00)
    assert drivers["pressure"].forecast.lat == pytest.approx(40.71)
    assert drivers["pressure"].forecast.lon == pytest.approx(-74.00)


def test_discovery_fails_clearly_when_no_station_is_available() -> None:
    with (
        patch("apps.explorer.data.NOAACoops.find_stations", return_value=[]),
        patch("apps.explorer.data.USGS.find_stations", return_value=[]),
        pytest.raises(ValueError, match="NOAA CO-OPS"),
    ):
        discover_us_sources(40.71, -74.00)
