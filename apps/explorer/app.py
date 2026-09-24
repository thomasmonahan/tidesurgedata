"""Interactive map explorer for tidesurgedata.

Run from the repository root with:
    streamlit run apps/explorer/app.py
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st
from data import (
    NOAA_SEARCH_RADIUS_KM,
    USGS_SEARCH_RADIUS_KM,
    SiteSelection,
    build_recipe,
    discover_us_sources,
    distance_km,
    utc_day_bounds,
)
from map_view import make_map
from plots import line_figure
from streamlit_folium import st_folium

DEFAULT_LAT = 40.7069
DEFAULT_LON = -74.0092


@st.cache_data(ttl=3600, show_spinner=False)
def cached_discovery(
    lat: float, lon: float, noaa_radius: float, usgs_radius: float
) -> SiteSelection:
    """Cache station discovery; rounded coordinates avoid near-duplicate map requests."""
    return discover_us_sources(
        round(lat, 4),
        round(lon, 4),
        noaa_radius_km=noaa_radius,
        usgs_radius_km=usgs_radius,
    )


def provenance_table(recipe) -> pd.DataFrame:
    """Flatten provenance from the most recently built frame for display."""
    rows = []
    for record in recipe.provenance():
        meta = record.meta
        rows.append(
            {
                "source": meta.source,
                "station": meta.station_id,
                "variable": meta.variable,
                "licence": meta.licence,
                "attribution": meta.attribution,
                "start": record.start,
                "end": record.end,
                "quality": record.quality,
            }
        )
    return pd.DataFrame(rows)


def station_summary(selection: SiteSelection) -> None:
    """Render selected station metadata."""
    st.subheader("Selected data sources")
    left, right = st.columns(2)
    with left:
        st.markdown("**NOAA CO-OPS water level**")
        st.write(selection.noaa.name or selection.noaa.station_id)
        st.caption(
            f"Station {selection.noaa.station_id} · "
            f"{distance_km(selection.lat, selection.lon, selection.noaa):.1f} km from click"
        )
    with right:
        st.markdown("**USGS discharge**")
        st.write(selection.usgs.name or selection.usgs.station_id)
        st.caption(
            f"Station {selection.usgs.station_id} · "
            f"{distance_km(selection.lat, selection.lon, selection.usgs):.1f} km from click"
        )
    st.caption(
        "Dynamical pressure is sampled at the clicked latitude/longitude; "
        "historical pressure is used for training and forecast pressure for forecast frames."
    )


def main() -> None:
    st.set_page_config(page_title="TideSurgeData Explorer", layout="wide")
    st.title("TideSurgeData Explorer")
    st.caption(
        "Click a US coastal location, discover nearby gauges, then retrieve and plot aligned data."
    )

    if "point" not in st.session_state:
        st.session_state.point = (DEFAULT_LAT, DEFAULT_LON)
    if "selection" not in st.session_state:
        st.session_state.selection = None

    with st.sidebar:
        st.header("Data request")
        today = date.today()
        start_date = st.date_input("Start date", today - timedelta(days=7))
        end_date = st.date_input("End date", today - timedelta(days=2))
        noaa_radius = st.number_input(
            "NOAA search radius (km)", min_value=1.0, value=NOAA_SEARCH_RADIUS_KM, step=5.0
        )
        usgs_radius = st.number_input(
            "USGS search radius (km)", min_value=1.0, value=USGS_SEARCH_RADIUS_KM, step=5.0
        )
        st.caption("Station discovery uses the package's existing find_stations() APIs.")

    lat, lon = st.session_state.point
    fmap = make_map(lat, lon, st.session_state.selection)
    map_state = st_folium(
        fmap, height=500, use_container_width=True, returned_objects=["last_clicked"]
    )

    clicked = map_state.get("last_clicked") if map_state else None
    if clicked:
        new_point = (float(clicked["lat"]), float(clicked["lng"]))
        if new_point != st.session_state.point:
            st.session_state.point = new_point
            st.session_state.selection = None
            st.rerun()

    lat, lon = st.session_state.point
    st.write(f"Selected point: **{lat:.5f}, {lon:.5f}**")

    if st.button("Discover nearby data sources", type="primary"):
        try:
            with st.spinner("Discovering NOAA CO-OPS and USGS stations..."):
                st.session_state.selection = cached_discovery(lat, lon, noaa_radius, usgs_radius)
        except Exception as exc:
            st.error(f"Station discovery failed: {exc}")

    selection = st.session_state.selection
    if selection is None:
        st.info("Click the map, then discover nearby data sources.")
        return

    station_summary(selection)
    recipe = build_recipe(selection)
    st.write("Feature columns:", recipe.feature_columns)
    st.write("Maximum lead time:", recipe.max_lead_time)

    if st.button("Load and plot observations"):
        try:
            start, end = utc_day_bounds(start_date, end_date)
            with st.spinner("Retrieving and aligning NOAA, USGS and Dynamical data..."):
                frame = recipe.training_frame(start, end)
            st.session_state.training_frame = frame
            st.session_state.training_provenance = provenance_table(recipe)
        except Exception as exc:
            st.error(f"Data retrieval failed: {exc}")

    frame = st.session_state.get("training_frame")
    if frame is not None:
        st.subheader("Aligned time series")
        target = recipe.target_column
        discharge_cols = [c for c in frame.columns if c.startswith("discharge_")]
        pressure_cols = [c for c in frame.columns if c.startswith("pressure_")]
        st.plotly_chart(line_figure(frame, [target], "Water level"), use_container_width=True)
        st.plotly_chart(
            line_figure(frame, discharge_cols, "River discharge"), use_container_width=True
        )
        st.plotly_chart(
            line_figure(frame, pressure_cols, "Surface pressure"), use_container_width=True
        )
        with st.expander("Aligned frame"):
            st.dataframe(frame, use_container_width=True)
        with st.expander("Provenance, licence and attribution", expanded=True):
            st.dataframe(
                st.session_state.training_provenance, use_container_width=True, hide_index=True
            )

    st.divider()
    st.subheader("Forecast inputs")
    horizon_cap = max(0, int(recipe.max_lead_time / pd.Timedelta("1h")))
    if horizon_cap <= 0:
        st.warning("This source combination currently has no positive forecast lead time.")
        return

    horizon = st.slider("Forecast horizon (hours)", 1, min(24, horizon_cap), min(6, horizon_cap))
    if st.button("Load forecast frame"):
        try:
            issued = pd.Timestamp.now(tz="UTC").floor("1h") - pd.Timedelta("2D")
            with st.spinner("Retrieving forecast inputs..."):
                forecast = recipe.forecast_frame(issued=issued, horizon_hours=horizon)
            st.session_state.forecast_frame = forecast
            st.session_state.forecast_provenance = provenance_table(recipe)
            st.session_state.forecast_issued = issued
        except Exception as exc:
            st.error(f"Forecast retrieval failed: {exc}")

        forecast = st.session_state.get("forecast_frame")
        if forecast is not None:
            st.caption(f"Issued: {st.session_state.forecast_issued}")

            # Keep physically different driver variables on separate y-axes/plots.
            # Discharge and pressure have very different units and numerical scales,
            # so plotting them together makes the discharge series difficult to see.
            discharge_cols = [
                column for column in forecast.columns if column.startswith("discharge_")
            ]

            pressure_cols = [
                column for column in forecast.columns if column.startswith("pressure_")
            ]

            if discharge_cols:
                st.plotly_chart(
                    line_figure(
                        forecast,
                        discharge_cols,
                        "Forecast-frame river discharge",
                        yaxis_title="Discharge",
                    ),
                    use_container_width=True,
                )

            if pressure_cols:
                st.plotly_chart(
                    line_figure(
                        forecast,
                        pressure_cols,
                        "Forecast-frame surface pressure",
                        yaxis_title="Pressure (Pa)",
                    ),
                    use_container_width=True,
                )

            with st.expander("Forecast frame"):
                st.dataframe(
                    forecast,
                    use_container_width=True,
                )

            with st.expander(
                "Forecast provenance, licence and attribution",
                expanded=True,
            ):
                st.dataframe(
                    st.session_state.forecast_provenance,
                    use_container_width=True,
                    hide_index=True,
                )


if __name__ == "__main__":
    main()
