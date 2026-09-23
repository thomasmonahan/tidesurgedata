"""Folium map helpers for the explorer."""

from __future__ import annotations

import folium

from data import SiteSelection


def make_map(
    lat: float,
    lon: float,
    selection: SiteSelection | None = None,
    *,
    zoom_start: int = 8,
) -> folium.Map:
    """Create a clickable map showing the requested point and selected stations."""
    fmap = folium.Map(location=[lat, lon], zoom_start=zoom_start, control_scale=True)

    folium.Marker(
        [lat, lon],
        tooltip="Selected point",
        icon=folium.Icon(color="blue", icon="crosshairs", prefix="fa"),
    ).add_to(fmap)

    if selection is not None:
        if selection.noaa.lat is not None and selection.noaa.lon is not None:
            folium.Marker(
                [selection.noaa.lat, selection.noaa.lon],
                tooltip=f"NOAA CO-OPS: {selection.noaa.name or selection.noaa.station_id}",
                popup=f"NOAA CO-OPS water level — {selection.noaa.station_id}",
                icon=folium.Icon(color="green", icon="water", prefix="fa"),
            ).add_to(fmap)
        if selection.usgs.lat is not None and selection.usgs.lon is not None:
            folium.Marker(
                [selection.usgs.lat, selection.usgs.lon],
                tooltip=f"USGS: {selection.usgs.name or selection.usgs.station_id}",
                popup=f"USGS discharge — {selection.usgs.station_id}",
                icon=folium.Icon(color="orange", icon="droplet", prefix="fa"),
            ).add_to(fmap)

    return fmap
