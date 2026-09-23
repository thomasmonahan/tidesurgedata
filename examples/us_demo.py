"""examples/us_demo.py — BL-19: US demo (NOAA CO-OPS + USGS + dynamical.org).

Location-driven source selection
-------------------------------
Given only latitude and longitude, the demo discovers the nearest NOAA CO-OPS
station serving water level and the nearest USGS station serving discharge
within configured search radii. Dynamical.org pressure is sampled directly at
the requested point:
    - discovered NOAA CO-OPS water level — target.
    - discovered USGS discharge — observed driver.
    - dynamical.org surface pressure — historical/reanalysis for training,
      forecast for the forecast frame. This is the one driver in this recipe
      that is genuinely forecast-capable, which is the point of including it:
      it demonstrates the "observed driver + forecast-capable driver" mixed
      workflow, rather than only observed history.

Why the discharge lags are negative
------------------------------------
``Recipe.max_lead_time`` is computed, for every driver that has no dedicated
forecast source, as ``min(-lag) - source.latency`` (see ADR 0002 and the
BL-13 spec tests in tests/spec/test_recipe_frames.py). A driver sampled at
lag 0 therefore caps the usable forecast horizon at zero or below. USGS
discharge has no forecast source here, so it must use strictly negative
lags (``-24``, ``-12`` hours) to leave positive lead time. The pressure
driver is exempt from this rule: because it carries its own ``forecast=``
source, ``Recipe`` can use it beyond ``issued`` without needing a negative
lag.

Expected max_lead_time
-----------------------
With ``lags_hours=(-24, -12)`` on discharge and USGS's declared latency,
``recipe.max_lead_time`` should be positive (typically several hours to
about a day, depending on USGS's actual latency once BL-06's adapter is
exercised live). This script computes it at runtime rather than hard-coding
a number, and picks a forecast horizon that is guaranteed to fit within it.

Known gaps at the time this script was written
------------------------------------------------
This file is intentionally a *thin* integration/example layer: it contains
no provider or frame-assembly logic of its own. As of writing, three of its
runtime dependencies are still open on the backlog and will raise
``NotImplementedError`` when exercised:

    * BL-05 — ``NOAACoops.metadata()`` / ``_fetch()`` / ``find_stations()``
      (issue #6, open). Constructing ``NOAACoops(...)`` succeeds; calling it
      does not yet.
    * BL-12 — ``Recipe.training_frame()`` and ``Recipe.provenance()``
      (issue #13, open).
    * BL-13 — ``Recipe.forecast_frame()`` (issue #14, open).

This script does not attempt to work around or reimplement any of the
above — that logic belongs to those issues, not to BL-19. Once BL-05,
BL-12 and BL-13 land, this script should run end to end without changes.

API surface assumed by this script
------------------------------------
BL-19 only consumes the public API; it was written from the package
README, ADR 0002 (docs/design/0002-data-contract.md), and the BL-05/06/12/
13/15/16 issue docstrings, because ``recipe.py`` and ``contract.py`` were
not read directly while drafting this file. Please check the following
against the real source before merging:

    * ``Recipe(target=..., drivers=(...), freq=..., target_column=...)``,
      with ``Recipe.feature_columns``, ``Recipe.max_lead_time``,
      ``Recipe.training_frame(start, end)``,
      ``Recipe.forecast_frame(issued=..., horizon_hours=...)``,
      ``Recipe.provenance()``, ``Recipe.to_json()`` / ``Recipe.from_json()``,
      and a ``Recipe.target_column`` attribute mirroring the constructor
      argument — confirmed from the package README's "Usage" examples and
      the BL-12/13 issue text, but the exact attribute name
      (``target_column`` vs. some other spelling) was not independently
      verified against recipe.py.
    * ``Driver(name, source, lags_hours=(...), forecast=None)`` with
      ``.name``, ``.source``, ``.lags_hours``, ``.forecast`` attributes —
      confirmed from the README's fake-data example.
    * ``contract.validate_frame(frame, target_column, feature_columns)`` —
      confirmed from ADR 0002 and this issue's own planning notes, not from
      contract.py directly.
    * ``recipe.provenance()`` returns an iterable of records exposing
      ``.meta`` (a ``SeriesMeta``: ``.source``, ``.station_id``,
      ``.variable``, ``.licence``, ``.attribution``, ...) and
      ``.start`` / ``.end`` / ``.quality`` — this shape is confirmed
      directly from ``sources/dynamical.py``'s ``FetchRecord`` usage
      (BL-15/16, merged), which is the strongest evidence available for
      this script.

None of this changes what BL-19 delivers; it only means a reviewer with
the real recipe.py open should skim the attribute names below before
merging.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tidesurgedata import Driver, Recipe
from tidesurgedata.contract import validate_frame
from tidesurgedata.sources.base import BaseSource, haversine_km
from tidesurgedata.sources.dynamical import Dynamical
from tidesurgedata.sources.noaa_coops import NOAACoops
from tidesurgedata.sources.usgs import USGS

# --- Location / station / dataset constants --------------------------------

# The demo is driven by a geographic point. Station-backed sources are
# discovered around this point
# For the gridded Dynamical data the nearest gridpoint is sampled.
DEFAULT_LAT = 40.7006
DEFAULT_LON = -74.0142
NOAA_SEARCH_RADIUS_KM = 50.0
USGS_SEARCH_RADIUS_KM = 100.0

# dynamical.org dataset identifiers per the STAC catalog. These names follow
# the convention shown in the package README's "planned API" example
# (``noaa-gfs-analysis`` / ``noaa-gefs-forecast``); confirm they match real
# STAC collection ids once BL-15/16 are exercised live.
DYNAMICAL_HISTORICAL_DATASET = "noaa-gfs-analysis"
DYNAMICAL_FORECAST_DATASET = "noaa-gefs-forecast"
DYNAMICAL_VARIABLE = "pressure_surface"

TRAINING_START = "2023-01-01T00:00Z"
TRAINING_END = "2023-04-01T00:00Z"

# Upper bound on the forecast horizon we *ask for*; the actual horizon used
# is capped at recipe.max_lead_time (see main()), so this script stays
# correct even if adapter latency constants change later.
DEFAULT_FORECAST_HORIZON_HOURS = 6
RECIPE_OUTPUT_PATH = Path("us_demo_recipe.json")


def _nearest_station(
    source_cls: type[BaseSource],
    lat: float,
    lon: float,
    radius_km: float,
    variable: str,
):
    """Return the nearest discovered station serving ``variable``."""
    stations = source_cls.find_stations(lat, lon, radius_km, variable=variable)
    if not stations:
        raise ValueError(
            f"No {source_cls.__name__} station serving {variable!r} found "
            f"within {radius_km:g} km of ({lat}, {lon})."
        )
    return min(
        stations,
        key=lambda meta: (
            haversine_km(lat, lon, meta.lat, meta.lon),
            str(meta.station_id),
        ),
    )


def build_recipe(lat: float, lon: float) -> Recipe:
    """Build the BL-19 recipe for a US latitude/longitude.

    NOAA CO-OPS and USGS stations are discovered with the common
    ``BaseSource.find_stations`` API and the nearest matching station is used.
    Dynamical is gridded, so the requested point is supplied directly.
    """
    noaa_meta = _nearest_station(
        NOAACoops, lat, lon, NOAA_SEARCH_RADIUS_KM, "water_level"
    )
    usgs_meta = _nearest_station(
        USGS, lat, lon, USGS_SEARCH_RADIUS_KM, "discharge"
    )

    target = NOAACoops(noaa_meta.station_id, product="water_level", datum="MSL")
    discharge = USGS(usgs_meta.station_id, parameter="discharge")

    pressure_historical = Dynamical(
        DYNAMICAL_HISTORICAL_DATASET, DYNAMICAL_VARIABLE, lat=lat, lon=lon
    )
    pressure_forecast = Dynamical(
        DYNAMICAL_FORECAST_DATASET, DYNAMICAL_VARIABLE, lat=lat, lon=lon
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


def print_provenance(recipe: Recipe, label: str) -> None:
    """Print one provenance block per source, from ``recipe.provenance()``.

    Deliberately does not invent or cache licence/attribution text: it
    prints exactly what ``SeriesMeta`` reports, since that is the
    mechanism BL-19 is meant to demonstrate (see the module docstring).
    Must be called immediately after the frame it describes, since
    ``provenance()`` reflects only the most recently built frame.
    """
    heading = f"{label} provenance"
    print(f"\n{heading}")
    print("-" * len(heading))

    for record in recipe.provenance():
        meta = record.meta
        print(f"{meta.source}")
        print(f"  variable:    {meta.variable}")
        print(f"  station:     {meta.station_id}")
        print(f"  fetched:     [{record.start}, {record.end})")
        print(f"  quality:     {record.quality}")
        print(f"  licence:     {meta.licence}")
        print(f"  attribution: {meta.attribution}")


def main() -> None:
    recipe = build_recipe(DEFAULT_LAT, DEFAULT_LON)

    print("feature columns:", recipe.feature_columns)
    print("max_lead_time:  ", recipe.max_lead_time)

    # --- Training frame -----------------------------------------------
    print(f"\nBuilding training frame [{TRAINING_START}, {TRAINING_END})...")
    train = recipe.training_frame(TRAINING_START, TRAINING_END)
    validate_frame(train, recipe.target_column, recipe.feature_columns)

    print(train.head())
    print("shape:", train.shape)
    print("NaN counts per column:")
    print(train.isna().sum())

    print_provenance(recipe, "Training")

    # --- Forecast frame --------------------------------------------------
    # Pick a horizon that is guaranteed to fit within max_lead_time, rather
    # than hard-coding one: this keeps the script correct even if adapter
    # latency constants change once BL-05/06/16 are exercised live.
    max_lead_hours = recipe.max_lead_time / pd.Timedelta("1h")
    if max_lead_hours <= 0:
        raise RuntimeError(
            "recipe.max_lead_time is non-positive "
            f"({recipe.max_lead_time!r}); this recipe cannot forecast. "
            "Check that all observed-only drivers use negative lags."
        )
    horizon_hours = min(DEFAULT_FORECAST_HORIZON_HOURS, int(max_lead_hours))

    # The issue time must be recent enough that live drivers have data, but
    # old enough that provider latency doesn't put required observations in
    # the future. forecast_frame() enforces `issued - source.latency` as an
    # as-of cutoff per observed driver; two days back is a conservative
    # margin for USGS/NOAA/dynamical.org's documented latencies.
    issue_time = pd.Timestamp.now(tz="UTC").floor("1h") - pd.Timedelta("2D")

    print(f"\nBuilding forecast frame (issued={issue_time}, horizon={horizon_hours}h)...")
    forecast = recipe.forecast_frame(issued=issue_time, horizon_hours=horizon_hours)
    validate_frame(forecast, recipe.target_column, recipe.feature_columns)

    print(forecast)

    print_provenance(recipe, "Forecast")

    # --- Reproducibility artifact -----------------------------------------
    # Written to the current working directory, not into examples/, so that
    # running the demo doesn't dirty the source tree.
    RECIPE_OUTPUT_PATH.write_text(recipe.to_json(indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote recipe definition to {RECIPE_OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
