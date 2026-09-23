"""As-of hindcasts: replay forecast frames at past issue times without leakage (ADR 0006)."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING

import pandas as pd

from tidesurgedata.align import to_grid
from tidesurgedata.timeutil import TimeLike, to_utc

if TYPE_CHECKING:
    from tidesurgedata.recipe import Recipe

__all__ = ["hindcast_frames"]


def hindcast_frames(
    recipe: Recipe, issue_times: Iterable[TimeLike], horizon_hours: float
) -> Iterator[tuple[pd.Timestamp, pd.DataFrame, pd.Series]]:
    """Replay forecast frames as of past issue times.

    For each issue time: (issue_time, forecast_frame as-of that time, observed target over the
    horizon). Model-agnostic: users apply any predictor. Must never leak data unavailable at the
    issue time.

    Parameters
    ----------
    recipe : Recipe
        Dataset definition.
    issue_times : iterable of time-like
        Timezone-aware issue times, processed in the order given.
    horizon_hours : float
        Forecast horizon in hours; must not exceed ``recipe.max_lead_time``.

    Yields
    ------
    issue_time : pandas.Timestamp
        The issue time, in UTC.
    frame : pandas.DataFrame
        ``recipe.forecast_frame(issue_time, horizon_hours)``: built only from data available by
        ``issue_time - latency`` for each source and forecasts initialised no later than that.
    observed : pandas.Series
        The target on the same grid as ``frame`` (``recipe.target_how``, never interpolated),
        named ``recipe.target_column_name``, for scoring predictions.

    Raises
    ------
    ValueError
        If any issue time is naive or the horizon exceeds ``recipe.max_lead_time``.

    Notes
    -----
    Data sources return today's version of historical data. Where providers revise data
    (preliminary to verified), a hindcast can only approximate what was known at the issue time;
    the frame's provenance records report the quality actually used.
    """
    max_lead = recipe.max_lead_time
    horizon = pd.Timedelta(hours=horizon_hours)

    if max_lead is not None and horizon > max_lead:
        raise ValueError(f"horizon {horizon} exceeds recipe max lead time {max_lead}.")

    for issue_time in issue_times:
        issued = to_utc(issue_time)

        frame = recipe.forecast_frame(issued, horizon_hours)

        target = recipe.target.fetch(
            frame.index[0],
            frame.index[-1] + pd.Timedelta(recipe.freq),
        )

        observed = to_grid(
            target,
            recipe.target.metadata(),
            recipe.freq,
            recipe.target_how,
            max_gap=None,
        ).reindex(frame.index)

        observed.name = recipe.target_column_name

        yield issued, frame, observed
