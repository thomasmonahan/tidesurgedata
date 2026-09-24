"""Fast 2D Dynamical weather maps for the Streamlit explorer.

Application-only code. This consumes Dynamical.fetch_fields() without
modifying tidesurgedata core functionality.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from tidesurgedata.sources.dynamical import Dynamical


DATASET = "noaa-gfs-analysis"

PRESSURE = "pressure_surface"
WIND_U = "wind_u_10m"
WIND_V = "wind_v_10m"

PRESSURE_CENTER_PA = 100_000.0


def fetch_weather_cube(
    lat: float,
    lon: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
    half_width_degrees: float,
):
    """Fetch pressure/U/V together in one remote operation."""

    bounds = {
        "lat_min": max(-90.0, lat - half_width_degrees),
        "lat_max": min(90.0, lat + half_width_degrees),
        "lon_min": max(-180.0, lon - half_width_degrees),
        "lon_max": min(180.0, lon + half_width_degrees),
    }

    source = Dynamical(
        DATASET,
        PRESSURE,
        lat=lat,
        lon=lon,
    )

    return source.fetch_fields(
        [
            PRESSURE,
            WIND_U,
            WIND_V,
        ],
        start,
        end,
        **bounds,
    )


def available_times(weather) -> pd.DatetimeIndex:
    """Return timestamps containing pressure and both wind components."""

    valid = None

    for variable in (
        PRESSURE,
        WIND_U,
        WIND_V,
    ):
        field = weather[variable]

        spatial_dims = [
            dim
            for dim in field.dims
            if dim != "time"
        ]

        this_valid = field.notnull().any(
            dim=spatial_dims
        )

        valid = (
            this_valid
            if valid is None
            else valid & this_valid
        )

    times = pd.DatetimeIndex(
        pd.to_datetime(
            weather.time.to_numpy(),
            utc=True,
        )
    )

    return times[valid.to_numpy()]


def time_index(
    weather,
    timestamp: pd.Timestamp,
) -> int:
    """Locate timestamp in an already-loaded weather cube."""

    times = pd.DatetimeIndex(
        pd.to_datetime(
            weather.time.to_numpy(),
            utc=True,
        )
    )

    return int(times.get_loc(timestamp))


def _cell_edges(values: np.ndarray) -> np.ndarray:
    """Convert grid centres to pcolormesh cell edges."""

    values = np.asarray(
        values,
        dtype=float,
    )

    if values.ndim != 1 or values.size < 2:
        raise ValueError(
            "At least two grid cells are required "
            "to render a 2D field."
        )

    midpoints = (
        values[:-1]
        + values[1:]
    ) / 2.0

    first = (
        values[0]
        - (midpoints[0] - values[0])
    )

    last = (
        values[-1]
        + (values[-1] - midpoints[-1])
    )

    return np.concatenate(
        ([first], midpoints, [last])
    )


def _pressure_norm(weather) -> TwoSlopeNorm:
    """One fixed pressure scale for the complete cached cube."""

    values = weather[PRESSURE].to_numpy()

    minimum = float(
        np.nanmin(values)
    )

    maximum = float(
        np.nanmax(values)
    )

    span = max(
        PRESSURE_CENTER_PA - minimum,
        maximum - PRESSURE_CENTER_PA,
        1.0,
    )

    return TwoSlopeNorm(
        vmin=PRESSURE_CENTER_PA - span,
        vcenter=PRESSURE_CENTER_PA,
        vmax=PRESSURE_CENTER_PA + span,
    )


def _map_axes(
    *,
    title: str,
    lon_edges: np.ndarray,
    lat_edges: np.ndarray,
):
    """Create the same geographic map convention as the video demo."""

    fig = plt.figure(
        figsize=(10, 7)
    )

    ax = fig.add_subplot(
        1,
        1,
        1,
        projection=ccrs.PlateCarree(),
    )

    ax.set_extent(
        [
            float(np.min(lon_edges)),
            float(np.max(lon_edges)),
            float(np.min(lat_edges)),
            float(np.max(lat_edges)),
        ],
        crs=ccrs.PlateCarree(),
    )

    ax.add_feature(
        cfeature.LAND,
        zorder=0,
    )

    ax.add_feature(
        cfeature.OCEAN,
        zorder=0,
    )

    ax.add_feature(
        cfeature.COASTLINE,
        linewidth=0.8,
        zorder=4,
    )

    ax.add_feature(
        cfeature.BORDERS,
        linewidth=0.4,
        zorder=4,
    )

    ax.add_feature(
        cfeature.LAKES,
        alpha=0.5,
        zorder=1,
    )

    ax.add_feature(
        cfeature.RIVERS,
        linewidth=0.4,
        zorder=4,
    )

    gridlines = ax.gridlines(
        draw_labels=True,
        linewidth=0.3,
        alpha=0.5,
    )

    gridlines.top_labels = False
    gridlines.right_labels = False

    ax.set_title(title)

    return fig, ax


def _add_colorbar(
    fig,
    ax,
    mappable,
    label: str,
):
    """Full-height close-set colorbar matching the video."""

    fig.canvas.draw()

    position = ax.get_position()

    cax = fig.add_axes(
        [
            position.x1 + 0.006,
            position.y0,
            0.022,
            position.height,
        ]
    )

    colorbar = fig.colorbar(
        mappable,
        cax=cax,
    )

    colorbar.set_label(label)

    return colorbar


def _mark_location(
    ax,
    lat: float,
    lon: float,
):
    """High-contrast selected-location marker."""

    transform = ccrs.PlateCarree()

    ax.scatter(
        [lon],
        [lat],
        marker="X",
        s=170,
        facecolor="white",
        edgecolor="black",
        linewidth=2.2,
        transform=transform,
        zorder=20,
        label="Selected location",
    )

    ax.scatter(
        [lon],
        [lat],
        marker="+",
        s=130,
        color="black",
        linewidth=2.5,
        transform=transform,
        zorder=21,
    )


def pressure_figure(
    weather,
    index: int,
    *,
    selected_lat: float,
    selected_lon: float,
):
    """Render one pressure timestep."""

    pressure = (
        weather[PRESSURE]
        .isel(time=index)
        .transpose(
            "latitude",
            "longitude",
        )
    )

    lats = pressure.latitude.to_numpy()
    lons = pressure.longitude.to_numpy()

    lat_edges = _cell_edges(lats)
    lon_edges = _cell_edges(lons)

    timestamp = pd.Timestamp(
        weather.time.to_numpy()[index]
    )

    fig, ax = _map_axes(
        title=(
            "Surface pressure\n"
            f"{timestamp} UTC"
        ),
        lon_edges=lon_edges,
        lat_edges=lat_edges,
    )

    image = ax.pcolormesh(
        lon_edges,
        lat_edges,
        pressure.to_numpy(),
        shading="flat",
        cmap="bwr",
        norm=_pressure_norm(weather),
        transform=ccrs.PlateCarree(),
        zorder=2,
        alpha=0.86,
    )

    _mark_location(
        ax,
        selected_lat,
        selected_lon,
    )

    ax.legend(
        loc="best"
    )

    _add_colorbar(
        fig,
        ax,
        image,
        "Pa",
    )

    return fig

def wind_figure(
    weather,
    index: int,
    *,
    selected_lat: float,
    selected_lon: float,
):
    """Render wind speed and vectors at the native Dynamical grid resolution."""

    # Extract U and V for this time step.
    wind_u = (
        weather[WIND_U]
        .isel(time=index)
        .transpose(
            "latitude",
            "longitude",
        )
    )

    wind_v = (
        weather[WIND_V]
        .isel(time=index)
        .transpose(
            "latitude",
            "longitude",
        )
    )

    # Wind speed at every native grid point.
    speed = np.hypot(
        wind_u,
        wind_v,
    )

    # Native grid-centre coordinates.
    lats = wind_u.latitude.to_numpy()
    lons = wind_u.longitude.to_numpy()

    # Convert centres to edges only for pcolormesh geometry.
    # This does NOT bin or resample the data.
    lat_edges = _cell_edges(lats)
    lon_edges = _cell_edges(lons)

    # Native grid locations for vectors.
    lon_grid, lat_grid = np.meshgrid(
        lons,
        lats,
    )

    timestamp = pd.Timestamp(
        weather.time.to_numpy()[index]
    )

    fig, ax = _map_axes(
        title=(
            "10 m wind speed and direction\n"
            f"{timestamp} UTC"
        ),
        lon_edges=lon_edges,
        lat_edges=lat_edges,
    )

    # -------------------------------------------------------------
    # Wind-speed heatmap
    #
    # Every native grid cell is plotted. No binning, interpolation
    # or spatial downsampling occurs here.
    # -------------------------------------------------------------

    image = ax.pcolormesh(
        lon_edges,
        lat_edges,
        speed.to_numpy(),
        shading="flat",
        transform=ccrs.PlateCarree(),
        zorder=2,
        alpha=0.82,
    )

    # -------------------------------------------------------------
    # Wind vectors
    #
    # Plot every native U/V grid point.
    #
    # There is deliberately NO stride and NO [::stride, ::stride]
    # indexing here.
    # -------------------------------------------------------------

    ax.quiver(
        lon_grid,
        lat_grid,
        wind_u.to_numpy(),
        wind_v.to_numpy(),
        transform=ccrs.PlateCarree(),
        zorder=5,
        angles="xy",
        scale_units="xy",
        width=0.0020,
        headwidth=3.0,
        headlength=4.0,
        headaxislength=3.6,
    )

    _mark_location(
        ax,
        selected_lat,
        selected_lon,
    )

    ax.legend(
        loc="best"
    )

    _add_colorbar(
        fig,
        ax,
        image,
        "Wind speed (m/s)",
    )

    return fig