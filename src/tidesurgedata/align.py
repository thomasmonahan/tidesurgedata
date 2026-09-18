"""Alignment of series onto a regular grid and materialisation of lagged driver columns.

See ``docs/design/0004-resampling-convention.md``. Grid timestamps are *centred*: a value at grid
time ``t`` with spacing ``freq`` represents either the instant ``t`` or the window
``[t - freq/2, t + freq/2)``, so no phase shift is introduced relative to instantaneous data.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd

from tidesurgedata.meta import SeriesMeta

__all__ = ["lag_column_name", "materialise_lags", "to_grid"]


def lag_column_name(driver: str, lag_hours: float) -> str:
    """Column name for a lagged driver.

    'discharge', -24 -> 'discharge_lag-24h'; 0 -> 'discharge_lag0h';
    -0.25 -> 'discharge_lag-0.25h'. Shortest exact decimal representation; reject NaN/inf.
    """
    lag = float(lag_hours)
    if not math.isfinite(lag):
        raise ValueError(f"lag_hours must be finite, got {lag_hours!r}.")
    # lag == 0 also normalises -0.0
    text = "0" if lag == 0 else np.format_float_positional(lag, trim="-")
    return f"{driver}_lag{text}h"


def to_grid(
    series: pd.Series,
    meta: SeriesMeta,
    freq: str | pd.Timedelta,
    how: Literal["instant", "mean"],
    max_gap: str | pd.Timedelta | None = None,
) -> pd.Series:
    """Resample a contract-valid series onto a regular, centred UTC grid.

    Parameters
    ----------
    series : pandas.Series
        Contract-valid input series (see :func:`tidesurgedata.contract.validate_series`).
    meta : SeriesMeta
        Metadata of ``series``; ``sampling``, ``window`` and ``label`` define its time
        convention.
    freq : str or pandas.Timedelta
        Grid spacing. Grid points are multiples of ``freq`` since the epoch, from the first
        (relabelled) timestamp floored to ``freq`` through the last ceiled to ``freq``,
        inclusive. An empty input gives an empty output.
    how : {"instant", "mean"}
        ``"instant"``: the value at the grid time if present, else the nearest sample within
        ``freq / 10``, else NaN.
        ``"mean"``: mean of samples in the centred window ``[t - freq/2, t + freq/2)``; NaN
        unless at least 50 % of the expected samples in the window are valid (non-NaN). The
        expected count is the window length divided by the series' native spacing.
    max_gap : str or pandas.Timedelta, optional
        After gridding, fill NaN runs by linear interpolation in time **only** when the gap
        between the bounding valid grid values is ``<= max_gap``. Never extrapolate beyond the
        first or last valid value. ``None`` disables interpolation.

    Returns
    -------
    pandas.Series
        Contract-valid series on the grid, same name as the input, ``float64``.

    Notes
    -----
    Window-mean inputs (``meta.sampling == "window_mean"``) are first relabelled to centre
    labels using ``meta.window`` and ``meta.label``: a ``"start"`` label is shifted by
    ``+window/2``, an ``"end"`` label by ``-window/2``. For example, an hourly mean labelled at
    the end of its hour would otherwise lag a tide by 30 minutes.

    Raises
    ------
    ContractError
        If ``series`` breaks the data contract.
    ValueError
        If ``how`` is invalid or ``freq`` / ``max_gap`` are not positive.
    """

    from tidesurgedata.contract import validate_series

    # Check whether series breaks the data contract
    validate_series(series, meta)

    # Protects against invalid how, freq and max_gap
    if how not in {"instant", "mean"}:
        raise ValueError(f"how must be 'instant' or 'mean', got {how!r}.")
    try:
        grid_freq = pd.Timedelta(freq)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"freq must be a positive duration, got {freq!r}.") from exc
    if grid_freq <= pd.Timedelta(0):
        raise ValueError(f"freq must be positive, got {freq!r}.")
    if max_gap is not None:
        try:
            gap = pd.Timedelta(max_gap)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"max_gap must be a positive duration, got {max_gap!r}.") from exc
        if gap <= pd.Timedelta(0):
            raise ValueError(f"max_gap must be positive, got {max_gap!r}.")

    if series.empty:
        return pd.Series(
            np.array([], dtype="float64"),
            index=pd.DatetimeIndex([], tz="UTC", name="time"),
            name=series.name,
        )

    aligned = series.copy()
    if meta.sampling == "window_mean":
        if meta.label == "start":
            aligned.index = aligned.index + meta.window / 2
        elif meta.label == "end":
            aligned.index = aligned.index - meta.window / 2

    first = aligned.index[0].floor(grid_freq)
    last = aligned.index[-1].ceil(grid_freq)
    grid = pd.date_range(first, last, freq=grid_freq, tz="UTC", name="time")

    if how == "instant":
        result = pd.Series(np.nan, index=grid, dtype="float64", name=series.name)
        positions = grid.searchsorted(aligned.index)
        exact = positions < len(grid)
        exact_positions = positions[exact]
        exact_index = aligned.index[exact]
        exact_match = grid[exact_positions] == exact_index
        result.iloc[exact_positions[exact_match]] = aligned.iloc[np.flatnonzero(exact)[exact_match]]

        tolerance = grid_freq / 10
        nearest = aligned.reindex(grid, method="nearest", tolerance=tolerance)
        result = result.fillna(nearest)
    else:
        differences = np.diff(aligned.index.as_unit("ns").asi8)
        positive = differences[differences > 0]
        if len(positive) == 0:
            native_spacing = grid_freq
        else:
            native_spacing = pd.to_timedelta(np.median(positive), unit="ns")
        expected = grid_freq / native_spacing
        values = np.full(len(grid), np.nan, dtype="float64")
        for position, timestamp in enumerate(grid):
            start = timestamp - grid_freq / 2
            stop = timestamp + grid_freq / 2
            left = aligned.index.searchsorted(start, side="left")
            right = aligned.index.searchsorted(stop, side="left")
            window_values = aligned.iloc[left:right]
            valid = window_values.dropna()
            if len(valid) >= expected / 2:
                values[position] = valid.mean()
        result = pd.Series(values, index=grid, name=series.name)

    if max_gap is not None:
        values = result.to_numpy(copy=True)
        valid_positions = np.flatnonzero(~np.isnan(values))
        for left, right in zip(valid_positions[:-1], valid_positions[1:], strict=True):
            if right == left + 1 or grid[right] - grid[left] > gap:
                continue
            missing = np.arange(left + 1, right)
            values[missing] = np.interp(
                missing,
                [left, right],
                [values[left], values[right]],
            )
        result = pd.Series(values, index=grid, name=series.name)

    validate_series(result, meta)
    return result


def materialise_lags(
    series_on_grid: pd.Series, driver_name: str, lags_hours: Sequence[float]
) -> pd.DataFrame:
    """Build one column per lag from a gridded driver series.

    For example, if lags_hours = (-1, 0, 2) and series_on_grid is

    time    discharge
    00:00   0
    01:00   1
    02:00   2
    03:00   3

    The output should be:
    
    time    lag-1h  lag0h  lag2h
    00:00    NaN      0      2
    01:00     0       1      3
    02:00     1       2     NaN
    03:00     2       3     NaN


    Parameters
    ----------
    series_on_grid : pandas.Series
        Driver series on a regular UTC grid (output of :func:`to_grid`). The grid spacing is
        inferred from the index.
    driver_name : str
        Driver name used in column names.
    lags_hours : sequence of float
        Lags in hours; negative means the past.

    Returns
    -------
    pandas.DataFrame
        Same index as ``series_on_grid``; columns named ``lag_column_name(driver_name, lag)`` in
        the order of ``lags_hours``; the value in column ``lag`` at time ``t`` is
        ``series(t + lag)`` (NaN where ``t + lag`` is outside the series); ``float64``.

    Raises
    ------
    ValueError
        If any lag is not an exact multiple of the grid step, or the index is not regular.
    """

    # Check that the time series is long enough
    if len(series_on_grid.index) < 2:
        raise ValueError("at least two timestamps are required to infer grid spacing")

    steps = np.diff(series_on_grid.index.as_unit("ns").asi8)

    if not np.all(steps == steps[0]):
        raise ValueError("index is not regular")

    # Keep the step in the same integer-nanosecond unit as the lag values.
    step = series_on_grid.index[1] - series_on_grid.index[0]

    lags = [pd.Timedelta(hours=float(lag_hour)) for lag_hour in lags_hours]
    lag_ns = np.asarray([lag.value for lag in lags], dtype=np.int64)
    step_ns = step.value

    if not np.all(lag_ns % step_ns == 0):
        raise ValueError("some lags are not an exact multiple of the grid step")

    values = series_on_grid.to_numpy(dtype="float64", copy=False)
    result = {}
<<<<<<< HEAD
    for lag_hour, lag_value in zip(lags_hours, lag_ns):
        offset = int(lag_value // step_ns)  # Converts each lag into an integer number of grid positions
        shifted = np.full(values.shape, np.nan, dtype="float64") # shift values
=======
    for lag_hour, lag_value in zip(lags_hours, lag_ns, strict=True):
        offset = int(lag_value // step_ns)
        shifted = np.full(values.shape, np.nan, dtype="float64")
>>>>>>> origin/main
        if offset >= 0:
            if offset < len(values):
                shifted[: len(values) - offset] = values[offset:]
        elif -offset < len(values):
            shifted[-offset:] = values[: len(values) + offset]
        result[lag_column_name(driver_name, lag_hour)] = shifted

    return pd.DataFrame(result, index=series_on_grid.index)
