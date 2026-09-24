"""Interactive map explorer for tidesurgedata.

Run from the repository root with:

    streamlit run apps/explorer/app.py
"""

from __future__ import annotations

from datetime import date, timedelta

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

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
from weather_fields import (
    available_times,
    fetch_weather_cube,
    pressure_figure,
    time_index,
    wind_figure,
)


DEFAULT_LAT = 40.7069
DEFAULT_LON = -74.0092


# ---------------------------------------------------------------------------
# Cached network operations
# ---------------------------------------------------------------------------


@st.cache_data(
    ttl=3600,
    show_spinner=False,
)
def cached_discovery(
    lat: float,
    lon: float,
    noaa_radius: float,
    usgs_radius: float,
) -> SiteSelection:
    """Cache station discovery.

    Coordinates are rounded to avoid almost-identical map clicks creating
    separate NOAA/USGS discovery requests.
    """

    return discover_us_sources(
        round(lat, 4),
        round(lon, 4),
        noaa_radius_km=noaa_radius,
        usgs_radius_km=usgs_radius,
    )


@st.cache_data(
    ttl=1800,
    show_spinner=False,
)
def cached_weather_cube(
    lat: float,
    lon: float,
    start_iso: str,
    end_iso: str,
    half_width_degrees: float,
):
    """Fetch and cache one regional Dynamical weather cube.

    This is deliberately the ONLY place in the application where the 2D
    weather cube is fetched.

    Once returned, changing the time slider uses the xarray Dataset already
    stored in session state and makes no additional Dynamical request.
    """

    return fetch_weather_cube(
        round(lat, 4),
        round(lon, 4),
        pd.Timestamp(start_iso),
        pd.Timestamp(end_iso),
        half_width_degrees,
    )


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


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


def station_summary(
    selection: SiteSelection,
) -> None:
    """Render selected station metadata."""

    st.subheader("Selected data sources")

    left, right = st.columns(2)

    with left:
        st.markdown(
            "**NOAA CO-OPS water level**"
        )

        st.write(
            selection.noaa.name
            or selection.noaa.station_id
        )

        st.caption(
            f"Station {selection.noaa.station_id} · "
            f"{distance_km(selection.lat, selection.lon, selection.noaa):.1f} "
            "km from click"
        )

    with right:
        st.markdown(
            "**USGS discharge**"
        )

        st.write(
            selection.usgs.name
            or selection.usgs.station_id
        )

        st.caption(
            f"Station {selection.usgs.station_id} · "
            f"{distance_km(selection.lat, selection.lon, selection.usgs):.1f} "
            "km from click"
        )

    st.caption(
        "Dynamical pressure is sampled at the clicked latitude/longitude; "
        "historical pressure is used for training and forecast pressure for "
        "forecast frames."
    )


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------


def main() -> None:

    st.set_page_config(
        page_title="TideSurgeData Explorer",
        layout="wide",
    )

    st.title(
        "TideSurgeData Explorer"
    )

    st.caption(
        "Click a US coastal location, discover nearby gauges, "
        "then retrieve and plot aligned data."
    )

    # ------------------------------------------------------------------
    # Session state
    # ------------------------------------------------------------------

    if "point" not in st.session_state:
        st.session_state.point = (
            DEFAULT_LAT,
            DEFAULT_LON,
        )

    if "selection" not in st.session_state:
        st.session_state.selection = None

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------

    with st.sidebar:

        st.header(
            "Data request"
        )

        today = date.today()

        start_date = st.date_input(
            "Start date",
            today - timedelta(days=7),
        )

        end_date = st.date_input(
            "End date",
            today - timedelta(days=2),
        )

        noaa_radius = st.number_input(
            "NOAA search radius (km)",
            min_value=1.0,
            value=NOAA_SEARCH_RADIUS_KM,
            step=5.0,
        )

        usgs_radius = st.number_input(
            "USGS search radius (km)",
            min_value=1.0,
            value=USGS_SEARCH_RADIUS_KM,
            step=5.0,
        )

        st.caption(
            "Station discovery uses the package's existing "
            "find_stations() APIs."
        )

        # --------------------------------------------------------------
        # 2D weather controls
        # --------------------------------------------------------------

        st.divider()

        st.subheader(
            "2D weather"
        )

        weather_half_width = st.slider(
            "Field half-width (degrees)",
            min_value=1.0,
            max_value=5.0,
            value=2.0,
            step=0.5,
            help=(
                "Size of the Dynamical region around the selected point. "
                "Smaller regions retrieve and render faster."
            ),
        )

        weather_hours = st.slider(
            "2D weather lookback (hours)",
            min_value=6,
            max_value=48,
            value=24,
            step=6,
            help=(
                "Amount of historical weather loaded into the time slider. "
                "Shorter periods retrieve faster."
            ),
        )

    # ------------------------------------------------------------------
    # Interactive map
    # ------------------------------------------------------------------

    lat, lon = st.session_state.point

    fmap = make_map(
        lat,
        lon,
        st.session_state.selection,
    )

    map_state = st_folium(
        fmap,
        height=500,
        use_container_width=True,
        returned_objects=[
            "last_clicked"
        ],
    )

    clicked = (
        map_state.get("last_clicked")
        if map_state
        else None
    )

    if clicked:

        new_point = (
            float(clicked["lat"]),
            float(clicked["lng"]),
        )

        if new_point != st.session_state.point:

            st.session_state.point = new_point
            st.session_state.selection = None

            # Do not reuse a weather cube belonging to the old point.
            st.session_state.pop(
                "weather_cube",
                None,
            )

            st.session_state.pop(
                "weather_point",
                None,
            )

            st.rerun()

    lat, lon = st.session_state.point

    st.write(
        f"Selected point: **{lat:.5f}, {lon:.5f}**"
    )

    # ------------------------------------------------------------------
    # Station discovery
    # ------------------------------------------------------------------

    if st.button(
        "Discover nearby data sources",
        type="primary",
    ):

        try:

            with st.spinner(
                "Discovering NOAA CO-OPS and USGS stations..."
            ):

                st.session_state.selection = (
                    cached_discovery(
                        lat,
                        lon,
                        noaa_radius,
                        usgs_radius,
                    )
                )

        except Exception as exc:

            st.error(
                f"Station discovery failed: {exc}"
            )

    selection = (
        st.session_state.selection
    )

    if selection is None:

        st.info(
            "Click the map, then discover nearby data sources."
        )

        return

    # ------------------------------------------------------------------
    # Recipe
    # ------------------------------------------------------------------

    station_summary(
        selection
    )

    recipe = build_recipe(
        selection
    )

    st.write(
        "Feature columns:",
        recipe.feature_columns,
    )

    st.write(
        "Maximum lead time:",
        recipe.max_lead_time,
    )

    # ------------------------------------------------------------------
    # Training / observations
    # ------------------------------------------------------------------

    if st.button(
        "Load and plot observations"
    ):

        try:

            start, end = utc_day_bounds(
                start_date,
                end_date,
            )

            with st.spinner(
                "Retrieving and aligning NOAA, USGS and Dynamical data..."
            ):

                frame = (
                    recipe.training_frame(
                        start,
                        end,
                    )
                )

            st.session_state.training_frame = (
                frame
            )

            st.session_state.training_provenance = (
                provenance_table(
                    recipe
                )
            )

        except Exception as exc:

            st.error(
                f"Data retrieval failed: {exc}"
            )

    frame = st.session_state.get(
        "training_frame"
    )

    if frame is not None:

        st.subheader(
            "Aligned time series"
        )

        target = (
            recipe.target_column
        )

        discharge_cols = [
            column
            for column in frame.columns
            if column.startswith(
                "discharge_"
            )
        ]

        pressure_cols = [
            column
            for column in frame.columns
            if column.startswith(
                "pressure_"
            )
        ]

        st.plotly_chart(
            line_figure(
                frame,
                [target],
                "Water level",
                yaxis_title="Water level (m)",
            ),
            use_container_width=True,
        )

        st.plotly_chart(
            line_figure(
                frame,
                discharge_cols,
                "River discharge",
                yaxis_title="Discharge (m³/s)",
            ),
            use_container_width=True,
        )

        st.plotly_chart(
            line_figure(
                frame,
                pressure_cols,
                "Surface pressure",
                yaxis_title="Pressure (Pa)",
            ),
            use_container_width=True,
        )

        with st.expander(
            "Aligned frame"
        ):

            st.dataframe(
                frame,
                use_container_width=True,
            )

        with st.expander(
            "Provenance, licence and attribution",
            expanded=True,
        ):

            st.dataframe(
                st.session_state.training_provenance,
                use_container_width=True,
                hide_index=True,
            )

    # ==================================================================
    # 2D WEATHER
    # ==================================================================
    #
    # IMPORTANT:
    #
    # Network retrieval happens ONLY when the button below is clicked.
    #
    # The time slider further below NEVER calls fetch_weather_cube().
    #
    # ==================================================================

    st.divider()

    st.subheader(
        "2D pressure and wind fields"
    )

    st.caption(
        "Pressure and wind are retrieved together once for a small region. "
        "Changing the time slider then uses the cached weather cube and "
        "does not make another Dynamical request."
    )

    # ------------------------------------------------------------------
    # Explicit network-loading button
    # ------------------------------------------------------------------

    if st.button(
        "Load 2D weather fields"
    ):

        try:

            # Use the selected observation end date as the weather endpoint.
            _, field_end = utc_day_bounds(
                start_date,
                end_date,
            )

            field_start = (
                field_end
                - pd.Timedelta(
                    hours=weather_hours
                )
            )

            with st.spinner(
                "Retrieving regional Dynamical pressure and wind..."
            ):

                # ------------------------------------------------------
                # EXPENSIVE OPERATION
                #
                # This is the only 2D Dynamical network call.
                # ------------------------------------------------------

                weather = (
                    cached_weather_cube(
                        lat,
                        lon,
                        field_start.isoformat(),
                        field_end.isoformat(),
                        weather_half_width,
                    )
                )

            # Store the returned xarray Dataset in session state.
            #
            # Slider reruns retrieve this object rather than calling
            # Dynamical again.

            st.session_state.weather_cube = (
                weather
            )

            st.session_state.weather_point = (
                lat,
                lon,
            )

            st.session_state.weather_start = (
                field_start
            )

            st.session_state.weather_end = (
                field_end
            )

        except Exception as exc:

            st.error(
                f"2D weather retrieval failed: {exc}"
            )

    # ------------------------------------------------------------------
    # Retrieve cube from session state
    #
    # NO NETWORK ACCESS OCCURS HERE.
    # ------------------------------------------------------------------

    weather = (
        st.session_state.get(
            "weather_cube"
        )
    )

    weather_point = (
        st.session_state.get(
            "weather_point"
        )
    )

    # ------------------------------------------------------------------
    # Time slider
    # ------------------------------------------------------------------

    if (
        weather is not None
        and weather_point == (lat, lon)
    ):

        times = available_times(
            weather
        )

        if len(times):

            st.caption(
                f"Loaded {len(times)} weather time bins from "
                f"{times[0]:%Y-%m-%d %H:%M} to "
                f"{times[-1]:%Y-%m-%d %H:%M} UTC."
            )

            # ----------------------------------------------------------
            # FAST OPERATION
            #
            # Moving this slider causes Streamlit to rerun, but the
            # weather data come from session_state. There is no call to
            # cached_weather_cube() or Dynamical.fetch_fields().
            # ----------------------------------------------------------

            selected_weather_time = (
                st.select_slider(
                    "2D field time (UTC)",
                    options=list(times),
                    value=times[-1],
                    format_func=lambda value: (
                        value.strftime(
                            "%Y-%m-%d %H:%M"
                        )
                    ),
                )
            )

            # This simply finds the already-loaded xarray time index.
            index = time_index(
                weather,
                selected_weather_time,
            )

            pressure_column, wind_column = st.columns(
                2,
                gap="small",
            )

            with pressure_column:
                pressure_fig = pressure_figure(
                    weather,
                    index,
                    selected_lat=lat,
                    selected_lon=lon,
                )

                st.pyplot(
                    pressure_fig,
                    use_container_width=True,
                    clear_figure=False,
                )

                plt.close(pressure_fig)


            with wind_column:
                wind_fig = wind_figure(
                    weather,
                    index,
                    selected_lat=lat,
                    selected_lon=lon,
                )

                st.pyplot(
                    wind_fig,
                    use_container_width=True,
                    clear_figure=False,
                )

                plt.close(wind_fig)

        else:

            st.warning(
                "No usable pressure/wind time bins "
                "were found in the loaded weather cube."
            )

    elif weather is not None:

        st.info(
            "The selected map location has changed. "
            "Load the 2D weather fields again for the new point."
        )

    # ------------------------------------------------------------------
    # Forecast
    # ------------------------------------------------------------------

    st.divider()

    st.subheader(
        "Forecast inputs"
    )

    horizon_cap = max(
        0,
        int(
            recipe.max_lead_time
            / pd.Timedelta("1h")
        ),
    )

    if horizon_cap <= 0:

        st.warning(
            "This source combination currently has no "
            "positive forecast lead time."
        )

        return

    horizon = st.slider(
        "Forecast horizon (hours)",
        1,
        min(
            24,
            horizon_cap,
        ),
        min(
            6,
            horizon_cap,
        ),
    )

    if st.button(
        "Load forecast frame"
    ):

        try:

            issued = (
                pd.Timestamp.now(
                    tz="UTC"
                ).floor("1h")
                - pd.Timedelta("2D")
            )

            with st.spinner(
                "Retrieving forecast inputs..."
            ):

                forecast = (
                    recipe.forecast_frame(
                        issued=issued,
                        horizon_hours=horizon,
                    )
                )

            st.session_state.forecast_frame = (
                forecast
            )

            st.session_state.forecast_provenance = (
                provenance_table(
                    recipe
                )
            )

            st.session_state.forecast_issued = (
                issued
            )

        except Exception as exc:

            st.error(
                f"Forecast retrieval failed: {exc}"
            )

    forecast = (
        st.session_state.get(
            "forecast_frame"
        )
    )

    if forecast is not None:

        st.caption(
            f"Issued: "
            f"{st.session_state.forecast_issued}"
        )

        discharge_cols = [
            column
            for column in forecast.columns
            if column.startswith(
                "discharge_"
            )
        ]

        pressure_cols = [
            column
            for column in forecast.columns
            if column.startswith(
                "pressure_"
            )
        ]

        if discharge_cols:

            st.plotly_chart(
                line_figure(
                    forecast,
                    discharge_cols,
                    "Forecast-frame river discharge",
                    yaxis_title="Discharge (m³/s)",
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

        with st.expander(
            "Forecast frame"
        ):

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