"""Spec tests for Recipe.training_frame / provenance (BL-12), forecast_frame (BL-13) and
forecast_frames (BL-22)."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from tidesurgedata import Driver, Recipe, validate_frame
from tidesurgedata.sources.fake import FakeRiver, FakeTideGauge
from tidesurgedata.timeutil import to_utc

BL12 = pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="BL-12")
BL13 = pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="BL-13")
BL22 = pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="BL-22")

START, END = "2024-01-10T00:00Z", "2024-01-12T00:00Z"
ISSUED = pd.Timestamp("2024-01-20T00:00Z")


@dataclasses.dataclass(frozen=True)
class GappyTideGauge(FakeTideGauge):
    """Fake tide gauge with missing values over [gap_start, gap_end)."""

    gap_start: str = "2024-01-10T06:00Z"
    gap_end: str = "2024-01-10T09:00Z"

    def _fetch(self, start, end):
        s, q, r = super()._fetch(start, end)
        s = s.copy()
        s[(s.index >= to_utc(self.gap_start)) & (s.index < to_utc(self.gap_end))] = np.nan
        return s, q, r


@dataclasses.dataclass(frozen=True)
class TruncatedRiver(FakeRiver):
    """Fake river that has no data after ``cutoff`` (simulates what was available then)."""

    cutoff: str = "2100-01-01T00:00Z"

    def _fetch(self, start, end):
        s, q, r = super()._fetch(start, end)
        return s[s.index <= to_utc(self.cutoff)], q, r


# --- training_frame (BL-12) ---------------------------------------------------------------------


def test_training_frame_shape_and_columns(fake_recipe):
    df = fake_recipe.training_frame(START, END)
    validate_frame(df, fake_recipe.target_column_name, fake_recipe.feature_columns)
    assert len(df) == 48
    assert df.index[0] == to_utc(START)
    assert df.index[-1] == to_utc(END) - pd.Timedelta("1h")
    assert list(df.columns) == [
        "water_level",
        "discharge_lag-24h",
        "discharge_lag-12h",
        "pressure_lag0h",
    ]


def test_training_frame_values(fake_recipe, fake_tide_gauge, fake_river):
    df = fake_recipe.training_frame(START, END)
    # target: instant values on the grid, never interpolated
    target = fake_tide_gauge.fetch(START, END)
    pd.testing.assert_series_equal(
        df["water_level"], target.reindex(df.index), check_names=False, check_freq=False
    )
    # drivers fetched over the window widened by their lags: no NaN at the start
    assert df.notna().all().all()
    # lag -24 h at t is the hourly mean of discharge around t - 24 h
    t = df.index[5]
    around = fake_river.fetch(
        t - pd.Timedelta("24h") - pd.Timedelta("30min"),
        t - pd.Timedelta("24h") + pd.Timedelta("30min"),
    )
    assert df.loc[t, "discharge_lag-24h"] == pytest.approx(around.mean())


def test_training_frame_keeps_nan_rows(fake_recipe):
    recipe = dataclasses.replace(fake_recipe, target=GappyTideGauge())
    df = recipe.training_frame(START, END)
    assert len(df) == 48
    gap = df.loc["2024-01-10T06:00Z":"2024-01-10T08:00Z", "water_level"]
    assert len(gap) == 3 and gap.isna().all()  # target never interpolated, rows kept


def test_training_frame_custom_target_column(fake_recipe):
    recipe = dataclasses.replace(fake_recipe, target_column="observations")
    df = recipe.training_frame(START, END)
    assert df.columns[0] == "observations"
    validate_frame(df, "observations", recipe.feature_columns)


def test_training_frame_resolves_gridded_drivers(fake_recipe, fake_met):
    unlocated = fake_met.with_location(None, None)
    recipe = dataclasses.replace(
        fake_recipe,
        drivers=(fake_recipe.drivers[0], Driver("pressure", unlocated, (0,), forecast=unlocated)),
    )
    df = recipe.training_frame(START, END)
    expected = fake_recipe.training_frame(START, END)
    pd.testing.assert_frame_equal(df, expected)

def test_training_frame_rejects_bad_ranges(fake_recipe):
    with pytest.raises(ValueError):
        fake_recipe.training_frame("2024-01-10", END)
    with pytest.raises(ValueError):
        fake_recipe.training_frame(END, START)


def test_provenance_after_training_frame(fake_recipe):
    fake_recipe.training_frame(START, END)
    records = fake_recipe.provenance()
    sources = [r.meta.source for r in records]
    assert sources[0] == "fake_tide_gauge"
    assert "fake_river" in sources and "fake_met" in sources
    river = next(r for r in records if r.meta.source == "fake_river")
    assert river.start <= to_utc(START) - pd.Timedelta("24h")


# --- forecast_frame (BL-13) ---------------------------------------------------------------------


def test_forecast_frame_index_and_nan_target(fake_recipe):
    df = fake_recipe.forecast_frame(ISSUED, horizon_hours=11)
    validate_frame(df, fake_recipe.target_column_name, fake_recipe.feature_columns)
    expected = pd.date_range(ISSUED + pd.Timedelta("1h"), ISSUED + pd.Timedelta("11h"), freq="1h")
    assert list(df.index) == list(expected)  # (issued, issued + horizon]
    assert df[fake_recipe.target_column_name].isna().all()
    assert df[fake_recipe.feature_columns].notna().all().all()


def test_forecast_frame_uses_forecast_after_issue(fake_recipe, fake_met):
    df = fake_recipe.forecast_frame(ISSUED, horizon_hours=11)
    fc = fake_met.fetch_forecast(ISSUED, pd.Timedelta("11h"))
    control = fc.values["control"].reindex(df.index)
    pd.testing.assert_series_equal(
        df["pressure_lag0h"], control, check_names=False, check_freq=False
    )
    member = fake_recipe.forecast_frame(ISSUED, horizon_hours=11, member="3")
    pd.testing.assert_series_equal(
        member["pressure_lag0h"],
        fc.values["3"].reindex(df.index),
        check_names=False,
        check_freq=False,
    )


def test_forecast_frame_no_data_after_issued_minus_latency(fake_recipe):
    cutoff = ISSUED - FakeRiver.latency
    truncated = TruncatedRiver(cutoff=cutoff.isoformat())
    leaky = fake_recipe.forecast_frame(ISSUED, horizon_hours=11)
    recipe = dataclasses.replace(
        fake_recipe,
        drivers=(
            dataclasses.replace(fake_recipe.drivers[0], source=truncated),
            fake_recipe.drivers[1],
        ),
    )
    honest = recipe.forecast_frame(ISSUED, horizon_hours=11)
    pd.testing.assert_frame_equal(leaky, honest)


def test_forecast_frame_horizon_beyond_max_lead_time_raises(fake_recipe):
    assert fake_recipe.max_lead_time == pd.Timedelta("11h")
    with pytest.raises(ValueError, match="lead"):
        fake_recipe.forecast_frame(ISSUED, horizon_hours=12)


def test_forecast_frame_non_positive_lead_time_raises(fake_tide_gauge, fake_met):
    recipe = Recipe(target=fake_tide_gauge, drivers=(Driver("pressure", fake_met, (0,)),))
    assert recipe.max_lead_time <= pd.Timedelta(0)
    with pytest.raises(ValueError):
        recipe.forecast_frame(ISSUED, horizon_hours=1)


def test_forecast_frame_rejects_naive_issue_time(fake_recipe):
    with pytest.raises(ValueError):
        fake_recipe.forecast_frame("2024-01-20T00:00", horizon_hours=6)


# --- forecast_frames (BL-22) --------------------------------------------------------------------


@BL22
def test_forecast_frames_per_member(fake_recipe):
    frames = fake_recipe.forecast_frames(ISSUED, horizon_hours=11)
    assert set(frames) == {"control", *[str(i) for i in range(1, 11)]}
    for member, df in frames.items():
        pd.testing.assert_frame_equal(
            df, fake_recipe.forecast_frame(ISSUED, horizon_hours=11, member=member)
        )
    assert not frames["1"]["pressure_lag0h"].equals(frames["2"]["pressure_lag0h"])
    pd.testing.assert_series_equal(
        frames["1"]["discharge_lag-12h"], frames["2"]["discharge_lag-12h"]
    )
