"""Plotting helpers for aligned tidesurgedata frames."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def line_figure(frame: pd.DataFrame, columns: list[str], title: str) -> go.Figure:
    """Build a simple synchronized-time line figure for selected columns."""
    fig = go.Figure()
    for column in columns:
        if column in frame.columns:
            fig.add_trace(go.Scatter(x=frame.index, y=frame[column], mode="lines", name=column))
    fig.update_layout(
        title=title,
        xaxis_title="Time (UTC)",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=45, b=20),
    )
    return fig


def weather_field_figure(
    pressure,
    wind_u=None,
    wind_v=None,
    *,
    selected_lat: float | None = None,
    selected_lon: float | None = None,
    wind_stride: int = 4,
) -> go.Figure:
    """Plot a 2D pressure field with optional U/V wind vectors."""
    import numpy as np

    lats = pressure["latitude"].values
    lons = pressure["longitude"].values
    z = pressure.transpose("latitude", "longitude").values

    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=lons,
            y=lats,
            z=z,
            contours=dict(showlabels=True),
            colorbar=dict(title=pressure.attrs.get("units", "pressure")),
            name="Surface pressure",
            hovertemplate="lon=%{x:.2f}<br>lat=%{y:.2f}<br>pressure=%{z:.2f}<extra></extra>",
        )
    )

    if wind_u is not None and wind_v is not None:
        stride = max(1, int(wind_stride))
        u = wind_u.transpose("latitude", "longitude").values
        v = wind_v.transpose("latitude", "longitude").values
        # Scale arrows relative to grid spacing while retaining vector direction.
        dx = float(np.median(np.abs(np.diff(lons)))) if len(lons) > 1 else 0.1
        dy = float(np.median(np.abs(np.diff(lats)))) if len(lats) > 1 else 0.1
        speed = np.hypot(u, v)
        finite = speed[np.isfinite(speed) & (speed > 0)]
        scale = 0.7 * min(dx, dy) / float(np.nanmedian(finite)) if finite.size else 0.0

        xs, ys = [], []
        for i in range(0, len(lats), stride):
            for j in range(0, len(lons), stride):
                if not np.isfinite(u[i, j]) or not np.isfinite(v[i, j]):
                    continue
                x0, y0 = float(lons[j]), float(lats[i])
                x1, y1 = x0 + float(u[i, j]) * scale, y0 + float(v[i, j]) * scale
                xs.extend([x0, x1, None])
                ys.extend([y0, y1, None])
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name="10 m wind",
                hoverinfo="skip",
                line=dict(width=1),
            )
        )

    if selected_lat is not None and selected_lon is not None:
        fig.add_trace(
            go.Scatter(
                x=[selected_lon],
                y=[selected_lat],
                mode="markers",
                name="Selected location",
                marker=dict(size=10, symbol="x"),
            )
        )

    selected_time = pressure.attrs.get("selected_time", "")
    fig.update_layout(
        title=f"Dynamical 2D surface pressure and wind{f' · {selected_time}' if selected_time else ''}",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        margin=dict(l=20, r=20, t=55, b=20),
    )
    return fig
