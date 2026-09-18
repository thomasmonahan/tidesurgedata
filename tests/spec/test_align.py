"""Spec tests for align.to_grid (BL-10) and align.materialise_lags (BL-11)."""

import numpy as np
import pandas as pd
import pytest

from tidesurgedata.align import lag_column_name, materialise_lags, to_grid
from tidesurgedata.contract import ContractError, validate_series

from ..test_meta import make_meta

T0 = pd.Timestamp("2024-01-01T00:00Z")


def minutes_series(freq, periods, start=T0, name="water_level"):
    """Series whose value is minutes since T0 at each timestamp."""
    index = pd.date_range(start, periods=periods, freq=freq, tz="UTC", name="time")
    values = (index - T0) / pd.Timedelta("1min")
    return pd.Series(np.asarray(values, dtype="float64"), index=index, name=name)


def discharge_meta(**kwargs):
    return make_meta(variable="discharge", units="m3 s-1", datum=None, **kwargs)


# --- to_grid: instant ---------------------------------------------------------------------------


def test_instant_exact_times():
    s = minutes_series("6min", 10 * 24)
    out = to_grid(s, make_meta(), "1h", how="instant")
    validate_series(out, make_meta())
    assert out.name == "water_level"
    assert (out.index.as_unit("ns").asi8 % pd.Timedelta("1h").value == 0).all()
    expected = s.reindex(pd.date_range(T0, periods=24, freq="1h", tz="UTC"))
    pd.testing.assert_series_equal(
        out.reindex(expected.index), expected, check_names=False, check_freq=False
    )


def test_instant_nearest_within_tenth_of_freq():
    s = minutes_series("1h", 12, start=T0 + pd.Timedelta("5min"))  # 00:05, 01:05, ...
    out = to_grid(s, make_meta(), "1h", how="instant")
    assert out[T0 + pd.Timedelta("1h")] == 65.0  # 6 min tolerance: nearest sample used
    late = minutes_series("1h", 12, start=T0 + pd.Timedelta("7min"))
    out = to_grid(late, make_meta(), "1h", how="instant")
    assert np.isnan(out[T0 + pd.Timedelta("1h")])  # 7 min away: NaN


def test_instant_is_not_interpolated_without_max_gap():
    s = minutes_series("1h", 10)
    s.iloc[4] = np.nan
    out = to_grid(s, make_meta(), "1h", how="instant")
    assert np.isnan(out[s.index[4]])


# --- to_grid: mean ------------------------------------------------------------------------------


def test_mean_uses_centred_window():
    s = minutes_series("15min", 4 * 24, name="discharge")
    out = to_grid(s, discharge_meta(), "1h", how="mean")
    t = T0 + pd.Timedelta("5h")
    # window [04:30, 05:30): samples at 04:30, 04:45, 05:00, 05:15 -> mean 5h in minutes - 7.5
    assert out[t] == pytest.approx(300.0 - 7.5)


def test_mean_relabels_end_labelled_window_means():
    s = minutes_series("15min", 4 * 24, name="discharge")  # value = label time in minutes
    meta = discharge_meta(sampling="window_mean", window=pd.Timedelta("15min"), label="end")
    out = to_grid(s, meta, "1h", how="mean")
    t = T0 + pd.Timedelta("5h")
    # centres = labels - 7.5 min; window [04:30, 05:30) holds labels 04:45, 05:00, 05:15, 05:30
    assert out[t] == pytest.approx(np.mean([285.0, 300.0, 315.0, 330.0]))


def test_mean_relabels_start_labelled_window_means():
    s = minutes_series("15min", 4 * 24, name="discharge")
    meta = discharge_meta(sampling="window_mean", window=pd.Timedelta("15min"), label="start")
    out = to_grid(s, meta, "1h", how="mean")
    t = T0 + pd.Timedelta("5h")
    # centres = labels + 7.5 min; window [04:30, 05:30) holds labels 04:30, 04:45, 05:00, 05:15
    assert out[t] == pytest.approx(np.mean([270.0, 285.0, 300.0, 315.0]))


def test_mean_centre_labelled_window_means_unchanged():
    s = minutes_series("15min", 4 * 24, name="discharge")
    meta = discharge_meta(sampling="window_mean", window=pd.Timedelta("15min"), label="centre")
    out = to_grid(s, meta, "1h", how="mean")
    assert out[T0 + pd.Timedelta("5h")] == pytest.approx(300.0 - 7.5)


def test_mean_hourly_end_labelled_does_not_lag_tide():
    """An hourly mean labelled at the end of its hour lags a tide by 30 min unless relabelled."""
    index = pd.date_range(T0, periods=24 * 60 * 3, freq="1min", tz="UTC")
    signal = np.cos(2 * np.pi * ((index - T0) / pd.Timedelta("12.42h")))
    minute = pd.Series(signal.astype("float64"), index=index, name="discharge")
    end_labelled = minute.resample("1h", label="right", closed="left").mean()
    meta = discharge_meta(sampling="window_mean", window=pd.Timedelta("1h"), label="end")
    out = to_grid(end_labelled, meta, "30min", how="instant")
    centres = end_labelled.index - pd.Timedelta("30min")
    np.testing.assert_allclose(out.reindex(centres).to_numpy(), end_labelled.to_numpy())


def test_mean_fifty_percent_rule():
    s = minutes_series("15min", 4 * 24, name="discharge")
    t = T0 + pd.Timedelta("5h")
    window = s.index[(s.index >= t - pd.Timedelta("30min")) & (s.index < t + pd.Timedelta("30min"))]
    two_missing = s.copy()
    two_missing[window[:2]] = np.nan
    out = to_grid(two_missing, discharge_meta(), "1h", how="mean")
    assert out[t] == pytest.approx(s[window[2:]].mean())  # 2 of 4 valid: 50 % is enough
    three_missing = s.copy()
    three_missing[window[:3]] = np.nan
    out = to_grid(three_missing, discharge_meta(), "1h", how="mean")
    assert np.isnan(out[t])  # 1 of 4 valid
    removed = s.drop(window[:3])  # absent samples count as missing too
    out = to_grid(removed, discharge_meta(), "1h", how="mean")
    assert np.isnan(out[t])


# --- to_grid: gaps ------------------------------------------------------------------------------


def test_max_gap_interpolates_short_gaps_only():
    s = minutes_series("1h", 24)
    s.iloc[5] = np.nan  # bounding values 2 h apart
    s.iloc[10:12] = np.nan  # bounding values 3 h apart
    out = to_grid(s, make_meta(), "1h", how="instant", max_gap="2h")
    assert out[s.index[5]] == pytest.approx(300.0)
    assert np.isnan(out[s.index[10]]) and np.isnan(out[s.index[11]])
    out = to_grid(s, make_meta(), "1h", how="instant", max_gap=pd.Timedelta("3h"))
    assert out[s.index[10]] == pytest.approx(600.0)
    assert out[s.index[11]] == pytest.approx(660.0)


def test_max_gap_never_extrapolates():
    s = minutes_series("1h", 24)
    s.iloc[:2] = np.nan
    s.iloc[-1] = np.nan
    out = to_grid(s, make_meta(), "1h", how="instant", max_gap="6h")
    assert out.iloc[:2].isna().all()
    assert np.isnan(out[s.index[-1]])


# --- to_grid: errors ----------------------------------------------------------------------------


def test_to_grid_rejects_contract_violations():
    s = minutes_series("1h", 5)
    s.index = s.index.tz_localize(None)
    with pytest.raises(ContractError):
        to_grid(s, make_meta(), "1h", how="instant")


@pytest.mark.parametrize("kwargs", [dict(how="median"), dict(freq="0h"), dict(max_gap="-1h")])
def test_to_grid_invalid_arguments(kwargs):
    params = dict(freq="1h", how="instant", max_gap=None)
    params.update(kwargs)
    with pytest.raises(ValueError):
        to_grid(minutes_series("1h", 5), make_meta(), **params)


# --- materialise_lags ---------------------------------------------------------------------------


def hourly(n=48, name="discharge"):
    return minutes_series("1h", n, name=name) / 60.0  # value = hours since T0


def test_materialise_lags_values_and_names():
    s = hourly()
    out = materialise_lags(s, "discharge", (-24, -1, 0, 2))
    assert list(out.columns) == [lag_column_name("discharge", lag) for lag in (-24, -1, 0, 2)]
    assert out.index.equals(s.index)
    assert (out.dtypes == "float64").all()
    t = s.index[30]
    assert out.loc[t, "discharge_lag-24h"] == 6.0
    assert out.loc[t, "discharge_lag-1h"] == 29.0
    assert out.loc[t, "discharge_lag0h"] == 30.0
    assert out.loc[t, "discharge_lag2h"] == 32.0
    assert out["discharge_lag-24h"].iloc[:24].isna().all()
    assert out["discharge_lag2h"].iloc[-2:].isna().all()


def test_materialise_lags_sub_hourly_grid():
    s = minutes_series("15min", 20, name="discharge")
    out = materialise_lags(s, "discharge", (-0.25, -0.5))
    assert list(out.columns) == ["discharge_lag-0.25h", "discharge_lag-0.5h"]
    assert out.iloc[5, 0] == 60.0 and out.iloc[5, 1] == 45.0


def test_materialise_lags_non_multiple_raises():
    with pytest.raises(ValueError, match="multiple"):
        materialise_lags(hourly(), "discharge", (-1.5,))
