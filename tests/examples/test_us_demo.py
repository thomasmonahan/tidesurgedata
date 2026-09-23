"""tests/examples/test_us_demo.py — offline structural regression test for
BL-19's examples/us_demo.py.

Scope
-----
This test is deliberately narrow. It does NOT re-test:

    * provider correctness (NOAA CO-OPS / USGS / dynamical.org unit
      conversion, metadata parsing, etc.) — owned by BL-05/06/15/16's own
      contract tests.
    * generic Recipe assembly or lead-time semantics — owned by
      tests/spec/test_recipe_frames.py (BL-12/13).

It only checks the one thing that is genuinely specific to this demo: that
``examples/us_demo.py`` wires the intended sources and drivers together
correctly, with the negative-lag rule respected and the recipe JSON
round-tripping. No network access is required or expected — see the
assumption below.

Network boundary in this test
-----------------------------
``build_recipe(lat, lon)`` intentionally performs station discovery through
``NOAACoops.find_stations`` and ``USGS.find_stations``. These tests monkeypatch
only those discovery calls. Source construction, nearest-station selection,
Dynamical location wiring, and Recipe construction remain real.

Loading the module under test
--------------------------------
``examples/`` is not a package, so the module is loaded directly by file
path with ``importlib`` rather than imported by name.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest

from tidesurgedata import Recipe

MODULE_PATH = Path(__file__).resolve().parents[2] / "examples" / "us_demo.py"


def _load_us_demo() -> ModuleType:
    spec = importlib.util.spec_from_file_location("us_demo", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def us_demo() -> ModuleType:
    return _load_us_demo()


@pytest.fixture()
def discovered_stations(us_demo: ModuleType, monkeypatch: pytest.MonkeyPatch) -> tuple[str, str]:
    """Stub provider discovery while leaving the demo's selection logic real."""
    noaa_id = "8518750"
    usgs_id = "01376500"

    def noaa_find(cls, lat, lon, radius_km, variable=None):
        assert variable == "water_level"
        return [SimpleNamespace(station_id=noaa_id, lat=lat + 0.01, lon=lon)]

    def usgs_find(cls, lat, lon, radius_km, variable=None):
        assert variable == "discharge"
        return [SimpleNamespace(station_id=usgs_id, lat=lat + 0.02, lon=lon)]

    monkeypatch.setattr(us_demo.NOAACoops, "find_stations", classmethod(noaa_find))
    monkeypatch.setattr(us_demo.USGS, "find_stations", classmethod(usgs_find))
    return noaa_id, usgs_id


@pytest.fixture()
def recipe(us_demo: ModuleType, discovered_stations: tuple[str, str]) -> Recipe:
    return us_demo.build_recipe(us_demo.DEFAULT_LAT, us_demo.DEFAULT_LON)


def _drivers_by_name(recipe: Recipe) -> dict:
    return {driver.name: driver for driver in recipe.drivers}


def test_target_is_discovered_noaa_coops_water_level(
    recipe: Recipe, discovered_stations: tuple[str, str]
) -> None:
    assert recipe.target.__class__.__name__ == "NOAACoops"
    assert recipe.target.product == "water_level"
    assert recipe.target.station_id == discovered_stations[0]


def test_discharge_driver_is_discovered_usgs_with_negative_lags(
    recipe: Recipe, discovered_stations: tuple[str, str]
) -> None:
    drivers = _drivers_by_name(recipe)
    assert "discharge" in drivers

    discharge_driver = drivers["discharge"]
    assert discharge_driver.source.__class__.__name__ == "USGS"
    assert discharge_driver.source.site_id == discovered_stations[1]

    # The rule this demo exists to demonstrate: an observed driver with no
    # forecast source of its own must use strictly negative lags, or it
    # caps max_lead_time at <= 0 (see the module docstring and ADR 0002).
    assert len(discharge_driver.lags_hours) > 0
    assert all(lag < 0 for lag in discharge_driver.lags_hours)
    assert discharge_driver.forecast is None


def test_pressure_driver_is_dynamical_and_forecast_capable(
    recipe: Recipe, us_demo: ModuleType
) -> None:
    drivers = _drivers_by_name(recipe)
    assert "pressure" in drivers

    pressure_driver = drivers["pressure"]
    assert pressure_driver.source.__class__.__name__ == "Dynamical"
    assert pressure_driver.source.dataset == us_demo.DYNAMICAL_HISTORICAL_DATASET
    assert pressure_driver.source.variable == us_demo.DYNAMICAL_VARIABLE
    assert pressure_driver.source.lat == us_demo.DEFAULT_LAT
    assert pressure_driver.source.lon == us_demo.DEFAULT_LON

    # This is the driver meant to demonstrate forecast-capable mixed-source
    # assembly, so it must carry its own forecast source.
    assert pressure_driver.forecast is not None
    assert pressure_driver.forecast.__class__.__name__ == "Dynamical"
    assert pressure_driver.forecast.dataset == us_demo.DYNAMICAL_FORECAST_DATASET
    assert pressure_driver.forecast.lat == us_demo.DEFAULT_LAT
    assert pressure_driver.forecast.lon == us_demo.DEFAULT_LON


def test_build_recipe_uses_requested_location_for_discovery_and_gridded_sources(
    us_demo: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    lat, lon = 41.25, -72.75
    calls = []

    def noaa_find(cls, got_lat, got_lon, radius_km, variable=None):
        calls.append(("noaa", got_lat, got_lon, radius_km, variable))
        # Deliberately return farther station first: selection must be nearest,
        # not dependent on provider result ordering.
        return [
            SimpleNamespace(station_id="far-noaa", lat=lat + 0.20, lon=lon),
            SimpleNamespace(station_id="near-noaa", lat=lat + 0.01, lon=lon),
        ]

    def usgs_find(cls, got_lat, got_lon, radius_km, variable=None):
        calls.append(("usgs", got_lat, got_lon, radius_km, variable))
        return [
            SimpleNamespace(station_id="far-usgs", lat=lat + 0.30, lon=lon),
            SimpleNamespace(station_id="near-usgs", lat=lat + 0.02, lon=lon),
        ]

    monkeypatch.setattr(us_demo.NOAACoops, "find_stations", classmethod(noaa_find))
    monkeypatch.setattr(us_demo.USGS, "find_stations", classmethod(usgs_find))

    custom_recipe = us_demo.build_recipe(lat, lon)
    pressure_driver = _drivers_by_name(custom_recipe)["pressure"]

    assert custom_recipe.target.station_id == "near-noaa"
    assert _drivers_by_name(custom_recipe)["discharge"].source.site_id == "near-usgs"
    assert calls == [
        ("noaa", lat, lon, us_demo.NOAA_SEARCH_RADIUS_KM, "water_level"),
        ("usgs", lat, lon, us_demo.USGS_SEARCH_RADIUS_KM, "discharge"),
    ]
    assert pressure_driver.source.lat == lat
    assert pressure_driver.source.lon == lon
    assert pressure_driver.forecast is not None
    assert pressure_driver.forecast.lat == lat
    assert pressure_driver.forecast.lon == lon


def test_build_recipe_fails_clearly_when_no_station_is_found(
    us_demo: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        us_demo.NOAACoops, "find_stations", classmethod(lambda cls, *args, **kwargs: [])
    )
    with pytest.raises(ValueError, match="No NOAACoops station serving 'water_level'"):
        us_demo.build_recipe(40.0, -74.0)


def test_max_lead_time_is_positive(recipe: Recipe) -> None:
    assert recipe.max_lead_time > pd.Timedelta(0)


def test_recipe_round_trips_through_json(recipe: Recipe) -> None:
    restored = Recipe.from_json(recipe.to_json())

    assert restored.feature_columns == recipe.feature_columns
    assert restored.target_column == recipe.target_column
    assert restored.max_lead_time == recipe.max_lead_time

    restored_drivers = _drivers_by_name(restored)
    original_drivers = _drivers_by_name(recipe)
    assert set(restored_drivers) == set(original_drivers)

    # The Dynamical pressure driver's forecast configuration is the part
    # most likely to be lost by a careless to_json()/from_json() pair, so
    # check it explicitly rather than relying on the blanket comparisons
    # above.
    assert restored_drivers["pressure"].forecast is not None
    assert (
        restored_drivers["pressure"].forecast.dataset
        == original_drivers["pressure"].forecast.dataset
    )
    assert restored_drivers["pressure"].source.lat == original_drivers["pressure"].source.lat
    assert restored_drivers["pressure"].source.lon == original_drivers["pressure"].source.lon
    assert (
        restored_drivers["pressure"].forecast.lat
        == original_drivers["pressure"].forecast.lat
    )
    assert (
        restored_drivers["pressure"].forecast.lon
        == original_drivers["pressure"].forecast.lon
    )
