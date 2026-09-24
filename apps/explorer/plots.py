"""Plotting helpers for aligned tidesurgedata frames."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def line_figure(
    frame: pd.DataFrame,
    columns: list[str],
    title: str,
    *,
    yaxis_title: str | None = None,
) -> go.Figure:
    """Build a synchronized-time line figure for selected columns."""
    fig = go.Figure()

    for column in columns:
        if column in frame.columns:
            fig.add_trace(
                go.Scatter(
                    x=frame.index,
                    y=frame[column],
                    mode="lines",
                    name=column,
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="Time (UTC)",
        yaxis_title=yaxis_title,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=45, b=20),
    )

    return fig
