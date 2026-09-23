"""Data/discovery layer for the interactive US tidesurgedata explorer.

This module deliberately contains no Streamlit code.  It composes the public
`tidesurgedata` APIs so the UI remains thin and the selection logic is easy to test.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tidesurgedata import Driver, Recipe
from tidesurgedata.meta import SeriesMeta
from tidesurgedata.sources.base import haversine_km
from tidesurgedata.sources.dynamical import Dynamical
from tidesurgedata.sources.noaa_coops import NOAACoops
from tidesurgedata.sources.usgs import USGS

NOAA_SEARCH_RADIUS_KM = 50.0
USGS_SEARCH_RADIUS_KM = 100.0
DYNAMICAL_HISTORICAL_DATASET = "noaa-gfs-analysis"
DYNAMICAL_FORECAST_DATASET = "noaa-gefs-forecast"
DYNAMICAL_VARIABLE = "pressure_surface"


@dataclass(frozen=True)
class SiteSelection:
    """Sources selected for a clicked geographic point."""

    lat: float
    lon: float
    noaa: SeriesMeta
    usgs: SeriesMeta
    noaa_candidates: tuple[SeriesMeta, ...]
    usgs_candidates: tuple[SeriesMeta, ...]


def distance_km(lat: float, lon: float, meta: SeriesMeta) -> float:
    """Distance from a clicked point to a station metadata record."""
    if meta.lat is None or meta.lon is None:
        return float("inf")
    return haversine_km(lat, lon, float(meta.lat), float(meta.lon))


def nearest_station(
    candidates: list[SeriesMeta], lat: float, lon: float, provider: str
) -> SeriesMeta:
    """Return the nearest station, independent of provider result ordering."""
    located = [meta for meta in candidates if meta.lat is not None and meta.lon is not None]
    if not located:
        raise ValueError(f"No {provider} station with coordinates was found near ({lat}, {lon}).")
    return min(located, key=lambda meta: distance_km(lat, lon, meta))


def discover_us_sources(
    lat: float,
    lon: float,
    *,
    noaa_radius_km: float = NOAA_SEARCH_RADIUS_KM,
    usgs_radius_km: float = USGS_SEARCH_RADIUS_KM,
) -> SiteSelection:
    """Discover the nearest CO-OPS water-level and USGS discharge stations."""
    noaa_candidates = NOAACoops.find_stations(
        lat, lon, noaa_radius_km, variable="water_level"
    )
    usgs_candidates = USGS.find_stations(lat, lon, usgs_radius_km, variable="discharge")

    noaa = nearest_station(noaa_candidates, lat, lon, "NOAA CO-OPS water-level")
    usgs = nearest_station(usgs_candidates, lat, lon, "USGS discharge")

    return SiteSelection(
        lat=float(lat),
        lon=float(lon),
        noaa=noaa,
        usgs=usgs,
        noaa_candidates=tuple(noaa_candidates),
        usgs_candidates=tuple(usgs_candidates),
    )


def build_recipe(selection: SiteSelection) -> Recipe:
    """Build the mixed CO-OPS + USGS + Dynamical recipe for a selected point."""
    target = NOAACoops(selection.noaa.station_id, product="water_level", datum="MSL")
    discharge = USGS(selection.usgs.station_id, parameter="discharge")

    pressure_historical = Dynamical(
        DYNAMICAL_HISTORICAL_DATASET,
        DYNAMICAL_VARIABLE,
        lat=selection.lat,
        lon=selection.lon,
    )
    pressure_forecast = Dynamical(
        DYNAMICAL_FORECAST_DATASET,
        DYNAMICAL_VARIABLE,
        lat=selection.lat,
        lon=selection.lon,
    )

    return Recipe(
        target=target,
        drivers=(
            Driver("discharge", discharge, lags_hours=(-24, -12)),
            Driver(
                "pressure",
                pressure_historical,
                lags_hours=(0,),
                forecast=pressure_forecast,
            ),
        ),
        freq="1h",
        target_column="observations",
    )


def utc_day_bounds(start_date: object, end_date: object) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Convert inclusive UI dates to the Recipe's half-open UTC interval."""
    start = pd.Timestamp(start_date, tz="UTC")
    end = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)
    if end <= start:
        raise ValueError("End date must not be before start date.")
    return start, end


def build_weather_field_source(selection: SiteSelection) -> Dynamical:
    """Return the analysis source used for 2D pressure/wind map fields."""
    return Dynamical(
        DYNAMICAL_HISTORICAL_DATASET,
        DYNAMICAL_VARIABLE,
        lat=selection.lat,
        lon=selection.lon,
    )
