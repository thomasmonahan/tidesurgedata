"""Standalone Dynamical 2D pressure + wind-field demo with coastline and animation.

Exercises the additive ``Dynamical.fetch_fields`` API only. It does not use
NOAA CO-OPS, USGS, Recipe, or BL-05. Pressure and both wind components are
fetched together as one regional space-time cube, so animation frames require
no additional network requests.

The demo produces:

* a surface-pressure map over a coastline basemap;
* a 10 m wind-speed map over the same basemap;
* a wind-speed map with direction vectors; and
* an animation containing one pressure/wind frame for each available wind
  time bin in the requested animation interval.

Run from the repository root after installing the meteorology extra and the
plotting dependencies::

    pip install -e ".[met]"
    pip install matplotlib cartopy pillow
    python examples/dynamical_fields_demo.py

If ``ffmpeg`` is available on PATH, the animation is written as MP4. Otherwise
Matplotlib falls back to an animated GIF using Pillow.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from tidesurgedata.sources.dynamical import Dynamical

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ImportError as exc:  # pragma: no cover - depends on optional demo dependency
    raise ImportError(
        "dynamical_fields_demo.py requires Cartopy for coastline maps. "
        "Install it with `pip install cartopy`."
    ) from exc

# Example region around New York Harbor. Adjust these constants to inspect a
# different region. Bounds intentionally remain modest to keep remote reads
# and plotting responsive.
LAT = 27.85 #17.521209 #40.71 
LON = -84.84 #-101.346836 #-74.01
HALF_WIDTH_DEGREES = 10.0

DATASET = "noaa-gfs-analysis"
PRESSURE_VARIABLE = "pressure_surface"
WIND_U_VARIABLE = "wind_u_10m"
WIND_V_VARIABLE = "wind_v_10m"

# Fetch and animate the most recent day of analysis data. Pressure and both wind
# components are retrieved together in one regional space-time request.
ANIMATION_LOOKBACK = pd.Timedelta("7d")
PRESSURE_CENTER_PA = 100_000.0
ANIMATION_FPS = 4
ANIMATION_OUTPUT = Path("dynamical_pressure_wind_7d_H_katrina_HR.mp4") #no data back i n2005


def _bounds() -> dict[str, float]:
    return {
        "lat_min": LAT - HALF_WIDTH_DEGREES,
        "lat_max": LAT + HALF_WIDTH_DEGREES,
        "lon_min": LON - HALF_WIDTH_DEGREES,
        "lon_max": LON + HALF_WIDTH_DEGREES,
    }


def _map_axes(*, title: str, bounds: dict[str, float]):
    """Create a geographic axes with land and coastline beneath the field."""
    fig = plt.figure(figsize=(15, 11))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(
        [bounds["lon_min"], bounds["lon_max"], bounds["lat_min"], bounds["lat_max"]],
        crs=ccrs.PlateCarree(),
    )
    ax.add_feature(cfeature.LAND, zorder=0)
    ax.add_feature(cfeature.OCEAN, zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, zorder=3)
    ax.add_feature(cfeature.LAKES, alpha=0.5, zorder=1)
    ax.add_feature(cfeature.RIVERS, linewidth=0.4, zorder=3)
    gridlines = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gridlines.top_labels = False
    gridlines.right_labels = False
    ax.set_title(title)
    return fig, ax


def _field_arrays(field):
    field = field.transpose("latitude", "longitude")
    return (
        field,
        field["latitude"].to_numpy(),
        field["longitude"].to_numpy(),
    )


def _cell_edges(values: np.ndarray) -> np.ndarray:
    """Return cell-edge coordinates from regularly or irregularly spaced centres."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("At least two 1D grid coordinates are required to infer cell edges.")
    midpoints = (values[:-1] + values[1:]) / 2.0
    first = values[0] - (midpoints[0] - values[0])
    last = values[-1] + (values[-1] - midpoints[-1])
    return np.concatenate(([first], midpoints, [last]))


def _field_geometry(field):
    """Return a field plus centre and edge coordinates for correct raster placement."""
    field, lats, lons = _field_arrays(field)
    return field, lats, lons, _cell_edges(lats), _cell_edges(lons)


def _pressure_norm(values: np.ndarray) -> TwoSlopeNorm:
    """Return one fixed symmetric pressure scale centred on 100,000 Pa.

    The limits are derived from the complete supplied pressure dataset and are
    symmetric about ``PRESSURE_CENTER_PA``.  For animation this function is
    called once using all frames, so the colour scale never changes with time.
    """
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    span = max(PRESSURE_CENTER_PA - vmin, vmax - PRESSURE_CENTER_PA)
    if span <= 0 or not np.isfinite(span):
        span = 1.0
    return TwoSlopeNorm(
        vmin=PRESSURE_CENTER_PA - span,
        vcenter=PRESSURE_CENTER_PA,
        vmax=PRESSURE_CENTER_PA + span,
    )


def _add_full_height_colorbar(fig, ax, mappable, label: str):
    """Add a colorbar matching the map axes height with only a narrow gap."""
    fig.canvas.draw()
    pos = ax.get_position()
    gap = 0.006
    width = 0.022
    cax = fig.add_axes([pos.x1 + gap, pos.y0, width, pos.height])
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(label)
    return cbar


def _mark_requested_location(ax) -> None:
    """Draw a high-contrast requested-location marker above all weather layers."""
    transform = ccrs.PlateCarree()
    ax.scatter(
        [LON], [LAT], marker="X", s=190, facecolor="white", edgecolor="black",
        linewidth=2.2, transform=transform, zorder=20, label="Requested location",
    )
    ax.scatter(
        [LON], [LAT], marker="+", s=150, color="black", linewidth=2.8,
        transform=transform, zorder=21,
    )


def _heatmap(
    field, *, title: str, colorbar_label: str, bounds: dict[str, float], pressure: bool = False
) -> None:
    """Display one latitude/longitude DataArray over a coastline map."""
    field, lats, lons, lat_edges, lon_edges = _field_geometry(field)
    fig, ax = _map_axes(title=title, bounds=bounds)
    kwargs = {}
    if pressure:
        kwargs.update(cmap="bwr", norm=_pressure_norm(field.to_numpy()))
    image = ax.pcolormesh(
        lon_edges,
        lat_edges,
        field.to_numpy(),
        shading="flat",
        transform=ccrs.PlateCarree(),
        zorder=2,
        alpha=0.86,
        **kwargs,
    )
    # Match the visible map to the actual raster cell edges, avoiding clipped
    # half-cells and apparent gaps at the perimeter.
    ax.set_extent(
        [float(np.min(lon_edges)), float(np.max(lon_edges)),
         float(np.min(lat_edges)), float(np.max(lat_edges))],
        crs=ccrs.PlateCarree(),
    )
    _mark_requested_location(ax)
    ax.legend(loc="best")
    _add_full_height_colorbar(fig, ax, image, colorbar_label)



def _wind_map(wind_u, wind_v, *, title: str, colorbar_label: str, bounds: dict[str, float]):
    """Plot wind speed as a heatmap and U/V components as arrows."""
    speed = np.hypot(wind_u, wind_v)
    speed.name = "wind_speed_10m"
    speed.attrs = dict(wind_u.attrs)

    speed, lats, lons, lat_edges, lon_edges = _field_geometry(speed)
    u = wind_u.transpose("latitude", "longitude")
    v = wind_v.transpose("latitude", "longitude")
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    fig, ax = _map_axes(title=title, bounds=bounds)
    image = ax.pcolormesh(
        lon_edges,
        lat_edges,
        speed.to_numpy(),
        shading="flat",
        transform=ccrs.PlateCarree(),
        zorder=2,
        alpha=0.82,
    )
    stride = max(1, max(len(lats), len(lons)) // 25)
    ax.quiver(
        lon_grid,
        lat_grid,
        u.to_numpy(),
        v.to_numpy(),
        transform=ccrs.PlateCarree(),
        zorder=4,
        angles="xy",
        scale_units="xy",
        width=0.0016,
        headwidth=3.2,
        headlength=4.2,
        headaxislength=3.8,
    )
    ax.set_extent(
        [float(np.min(lon_edges)), float(np.max(lon_edges)),
         float(np.min(lat_edges)), float(np.max(lat_edges))],
        crs=ccrs.PlateCarree(),
    )
    _mark_requested_location(ax)
    ax.legend(loc="best")
    _add_full_height_colorbar(fig, ax, image, colorbar_label)
    return fig


def _fetch_weather_cube(
    source: Dynamical,
    start: pd.Timestamp,
    end: pd.Timestamp,
    bounds: dict[str, float],
):
    """Fetch pressure and both wind components in one remote request."""
    print(f"Fetching weather cube [{start}, {end})...")
    weather = source.fetch_fields(
        [PRESSURE_VARIABLE, WIND_U_VARIABLE, WIND_V_VARIABLE],
        start,
        end,
        **bounds,
    )
    print(f"Weather cube loaded: {dict(weather.sizes)}")
    return weather


def _available_wind_times(weather) -> pd.DatetimeIndex:
    """Return UTC timestamps having both U and V wind data in the loaded cube."""
    u = weather[WIND_U_VARIABLE]
    v = weather[WIND_V_VARIABLE]

    # A time bin is usable when each component contains at least one finite
    # grid value. This is evaluated locally after the cube has been loaded.
    spatial_dims = [dim for dim in u.dims if dim != "time"]
    u_ok = u.notnull().any(dim=spatial_dims)
    v_ok = v.notnull().any(dim=spatial_dims)
    usable = (u_ok & v_ok).to_numpy()

    times = pd.DatetimeIndex(pd.to_datetime(weather.time.to_numpy(), utc=True))
    return times[usable]


def _animation_frames(weather, times: pd.DatetimeIndex):
    """Build lightweight local views for each usable animation timestamp."""
    frames = []
    cube_times = pd.DatetimeIndex(pd.to_datetime(weather.time.to_numpy(), utc=True))
    positions = {timestamp: i for i, timestamp in enumerate(cube_times)}

    for timestamp in times:
        i = positions[timestamp]
        frames.append(
            (
                timestamp,
                weather[PRESSURE_VARIABLE].isel(time=i),
                weather[WIND_U_VARIABLE].isel(time=i),
                weather[WIND_V_VARIABLE].isel(time=i),
            )
        )
    return frames


def _save_animation(frames, *, bounds: dict[str, float], output: Path) -> Path:
    """Save pressure heatmap + wind vectors as MP4, falling back to GIF."""
    if not frames:
        raise ValueError("No wind time bins were available for animation.")

    # Use global limits so colours mean the same thing from frame to frame.
    pressure_values = np.concatenate([frame[1].to_numpy().ravel() for frame in frames])
    speed_values = np.concatenate(
        [np.hypot(frame[2], frame[3]).to_numpy().ravel() for frame in frames]
    )
    pressure_norm = _pressure_norm(pressure_values)
    speed_min, speed_max = np.nanmin(speed_values), np.nanmax(speed_values)

    fig = plt.figure(figsize=(15, 11))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(
        [bounds["lon_min"], bounds["lon_max"], bounds["lat_min"], bounds["lat_max"]],
        crs=ccrs.PlateCarree(),
    )
    ax.add_feature(cfeature.LAND, zorder=0)
    ax.add_feature(cfeature.OCEAN, zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=4)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, zorder=4)

    first_time, first_pressure, first_u, first_v = frames[0]
    first_pressure, lats, lons, lat_edges, lon_edges = _field_geometry(first_pressure)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    image = ax.pcolormesh(
        lon_edges,
        lat_edges,
        first_pressure.to_numpy(),
        shading="flat",
        transform=ccrs.PlateCarree(),
        zorder=2,
        alpha=0.86,
        cmap="bwr",
        norm=pressure_norm,
    )
    speed = np.hypot(first_u, first_v).transpose("latitude", "longitude")
    # Wind speed is represented by arrow length; pressure remains the heatmap.
    quiver = ax.quiver(
        lon_grid,
        lat_grid,
        first_u.transpose("latitude", "longitude").to_numpy(),
        first_v.transpose("latitude", "longitude").to_numpy(),
        transform=ccrs.PlateCarree(),
        zorder=5,
        angles="xy",
        scale_units="xy",
        width=0.0014,
        headwidth=3.0,
        headlength=4.0,
        headaxislength=3.6,
    )
    ax.set_extent(
        [float(np.min(lon_edges)), float(np.max(lon_edges)),
         float(np.min(lat_edges)), float(np.max(lat_edges))],
        crs=ccrs.PlateCarree(),
    )
    _mark_requested_location(ax)
    pressure_units = first_pressure.attrs.get("units", PRESSURE_VARIABLE)
    _add_full_height_colorbar(fig, ax, image, str(pressure_units))
    title = ax.set_title(f"Surface pressure and 10 m wind\n{first_time}")

    # Keep these values visible in the console; they also prove the wind field
    # contributes to every frame even though pressure owns the colour scale.
    print(f"Animation wind-speed range: {speed_min:.3f} to {speed_max:.3f}")

    def update(frame_index: int):
        timestamp, pressure, wind_u, wind_v = frames[frame_index]
        pressure = pressure.transpose("latitude", "longitude")
        u = wind_u.transpose("latitude", "longitude")
        v = wind_v.transpose("latitude", "longitude")
        image.set_array(pressure.to_numpy().ravel())
        quiver.set_UVC(
            u.to_numpy(),
            v.to_numpy(),
        )
        title.set_text(f"Surface pressure and 10 m wind\n{timestamp}")
        return image, quiver, title

    movie = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=1000 / ANIMATION_FPS,
        blit=False,
    )

    output = output.resolve()
    if animation.writers.is_available("ffmpeg"):
        movie.save(output, writer="ffmpeg", fps=ANIMATION_FPS, dpi=180)
    else:
        output = output.with_suffix(".gif")
        movie.save(output, writer="pillow", fps=ANIMATION_FPS, dpi=180)

    plt.close(fig)
    return output


def main() -> None:
    requested_time = pd.Timestamp.now(tz="UTC").floor("1h") - pd.Timedelta("1D")
    bounds = _bounds()

    source = Dynamical(
        dataset=DATASET,
        variable=PRESSURE_VARIABLE,
        lat=LAT,
        lon=LON,
    )

    # One cube supplies both the static plots and every animation frame.
    animation_end = requested_time + pd.Timedelta("1h")
    animation_start = animation_end - ANIMATION_LOOKBACK

    print(f"Dataset: {DATASET}")
    print(f"Requested time: {requested_time}")
    print(f"Bounds: {bounds}")

    weather = _fetch_weather_cube(source, animation_start, animation_end, bounds)
    times = _available_wind_times(weather)
    print(f"Available wind time bins: {len(times)}")

    if not len(times):
        raise ValueError("No wind data were available in the requested animation interval.")

    # Use the latest usable wind time for the static maps. This avoids a second
    # remote request and guarantees pressure/U/V are taken from the same cube.
    selected_time = times[-1]
    cube_times = pd.DatetimeIndex(pd.to_datetime(weather.time.to_numpy(), utc=True))
    selected_index = cube_times.get_loc(selected_time)

    pressure = weather[PRESSURE_VARIABLE].isel(time=selected_index)
    wind_u = weather[WIND_U_VARIABLE].isel(time=selected_index)
    wind_v = weather[WIND_V_VARIABLE].isel(time=selected_index)

    wind_speed = np.hypot(wind_u, wind_v)
    wind_speed.name = "wind_speed_10m"
    wind_speed.attrs = dict(wind_u.attrs)

    pressure_units = pressure.attrs.get("units", PRESSURE_VARIABLE)
    wind_units = wind_u.attrs.get("units", "wind speed")

    print(f"Selected Dynamical time: {selected_time}")
    print(f"Pressure grid: {pressure.sizes}")
    print(f"Wind grid: {wind_u.sizes}")

    _heatmap(
        pressure,
        title=f"Dynamical surface pressure\n{selected_time}",
        colorbar_label=str(pressure_units),
        bounds=bounds,
        pressure=True,
    )
    _heatmap(
        wind_speed,
        title=f"Dynamical 10 m wind speed\n{selected_time}",
        colorbar_label=str(wind_units),
        bounds=bounds,
    )
    _wind_map(
        wind_u,
        wind_v,
        title=f"10 m wind speed and direction\n{selected_time}",
        colorbar_label=str(wind_units),
        bounds=bounds,
    )

    frames = _animation_frames(weather, times)
    print("Rendering animation locally; no further Dynamical requests are required.")
    output = _save_animation(frames, bounds=bounds, output=ANIMATION_OUTPUT)
    print(f"Wrote animation: {output}")

    plt.show()


if __name__ == "__main__":
    main()
