"""Standalone Dynamical 2D pressure + wind-field demo.

Exercises the additive ``Dynamical.fetch_field`` and
``Dynamical.fetch_wind_field`` APIs only. It does not use NOAA CO-OPS, USGS,
Recipe, or BL-05.

Run from the repository root after installing the meteorology extra:

    pip install -e ".[met]"
    python examples/dynamical_fields_demo.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tidesurgedata.sources.dynamical import Dynamical

# Example region around New York Harbor. Adjust these constants to inspect a
# different region. Bounds intentionally remain modest to keep remote reads
# and plotting responsive.
LAT = 40.71
LON = -74.01
HALF_WIDTH_DEGREES = 3.0

DATASET = "noaa-gfs-analysis"
PRESSURE_VARIABLE = "pressure_surface"
WIND_U_VARIABLE = "wind_u_10m"
WIND_V_VARIABLE = "wind_v_10m"


def _heatmap(field, *, title: str, colorbar_label: str) -> None:
    """Display one latitude/longitude DataArray as a simple heatmap."""
    field = field.transpose("latitude", "longitude")
    lats = field["latitude"].to_numpy()
    lons = field["longitude"].to_numpy()

    fig, ax = plt.subplots(figsize=(9, 6))
    image = ax.pcolormesh(lons, lats, field.to_numpy(), shading="auto")
    ax.scatter([LON], [LAT], marker="x", label="Requested location", color = 'w')
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.colorbar(image, ax=ax, label=colorbar_label)
    fig.tight_layout()


def main() -> None:
    requested_time = pd.Timestamp.now(tz="UTC").floor("1h") - pd.Timedelta("1D")
    bounds = {
        "lat_min": LAT - HALF_WIDTH_DEGREES,
        "lat_max": LAT + HALF_WIDTH_DEGREES,
        "lon_min": LON - HALF_WIDTH_DEGREES,
        "lon_max": LON + HALF_WIDTH_DEGREES,
    }

    # The source's point coordinates remain useful metadata, but fetch_field()
    # takes explicit spatial bounds and does not alter the normal point-fetch API.
    source = Dynamical(
        dataset=DATASET,
        variable=PRESSURE_VARIABLE,
        lat=LAT,
        lon=LON,
    )

    print(f"Dataset: {DATASET}")
    print(f"Requested time: {requested_time}")
    print(f"Bounds: {bounds}")

    pressure = source.fetch_field(
        requested_time,
        **bounds,
        variable=PRESSURE_VARIABLE,
    )
    wind_u, wind_v = source.fetch_wind_field(
        requested_time,
        **bounds,
        u_variable=WIND_U_VARIABLE,
        v_variable=WIND_V_VARIABLE,
    )

    # U and V share a grid by contract, so wind magnitude is well defined.
    wind_speed = np.hypot(wind_u, wind_v)
    wind_speed.name = "wind_speed_10m"
    wind_speed.attrs = dict(wind_u.attrs)

    selected_time = pressure.attrs.get("selected_time", str(requested_time))
    pressure_units = pressure.attrs.get("units", PRESSURE_VARIABLE)
    wind_units = wind_u.attrs.get("units", "wind speed")

    print(f"Selected Dynamical time: {selected_time}")
    print(f"Pressure grid: {pressure.sizes}")
    print(f"Wind grid: {wind_u.sizes}")

    _heatmap(
        pressure,
        title=f"Dynamical surface pressure\n{selected_time}",
        colorbar_label=str(pressure_units),
    )
    _heatmap(
        wind_speed,
        title=f"Dynamical 10 m wind speed\n{selected_time}",
        colorbar_label=str(wind_units),
    )

    # Add a third view combining wind direction with the wind-speed heatmap.
    speed = wind_speed.transpose("latitude", "longitude")
    u = wind_u.transpose("latitude", "longitude")
    v = wind_v.transpose("latitude", "longitude")
    lats = speed["latitude"].to_numpy()
    lons = speed["longitude"].to_numpy()
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    fig, ax = plt.subplots(figsize=(9, 6))
    image = ax.pcolormesh(lons, lats, speed.to_numpy(), shading="auto")
    # Thin vectors automatically so global/high-resolution grids remain readable.
    stride = max(1, max(len(lats), len(lons)) // 25)
    ax.quiver(
        lon_grid[::stride, ::stride],
        lat_grid[::stride, ::stride],
        u.to_numpy()[::stride, ::stride],
        v.to_numpy()[::stride, ::stride],
    )
    ax.scatter([LON], [LAT], marker="x", label="Requested location")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"10 m wind speed and direction\n{selected_time}")
    ax.legend(loc="best")
    fig.colorbar(image, ax=ax, label=str(wind_units))
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
