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

    raise NotImplementedError("BL-10: align.to_grid resampling and gap policy")


def materialise_lags(
    series_on_grid: pd.Series, driver_name: str, lags_hours: Sequence[float]
) -> pd.DataFrame:
    """Build one column per lag from a gridded driver series.

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
    raise NotImplementedError("BL-11: align.materialise_lags")
