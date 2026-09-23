"""Spec tests for hindcast_frames (BL-14)."""

import dataclasses

import pandas as pd
import pytest

from tidesurgedata import hindcast_frames

from .test_recipe_frames import TruncatedRiver

ISSUES = pd.date_range("2024-01-20T00:00Z", periods=4, freq="6h")


def test_hindcast_frames_structure(fake_recipe, fake_tide_gauge):
    results = list(hindcast_frames(fake_recipe, ISSUES, horizon_hours=11))
    assert [r[0] for r in results] == list(ISSUES)
    for issued, frame, observed in results:
        pd.testing.assert_frame_equal(frame, fake_recipe.forecast_frame(issued, 11))
        assert observed.index.equals(frame.index)
        assert observed.name == fake_recipe.target_column_name
        truth = fake_tide_gauge.fetch(frame.index[0], frame.index[-1] + pd.Timedelta("1h"))
        pd.testing.assert_series_equal(
            observed, truth.reindex(frame.index), check_names=False, check_freq=False
        )


def test_hindcast_frames_no_leakage(fake_recipe):
    for issued, frame, _ in hindcast_frames(fake_recipe, ISSUES, horizon_hours=11):
        cutoff = issued - TruncatedRiver.latency
        truncated = dataclasses.replace(
            fake_recipe.drivers[0], source=TruncatedRiver(cutoff=cutoff.isoformat())
        )
        honest = dataclasses.replace(fake_recipe, drivers=(truncated, fake_recipe.drivers[1]))
        pd.testing.assert_frame_equal(frame, honest.forecast_frame(issued, 11))


def test_hindcast_frames_horizon_beyond_max_lead_time(fake_recipe):
    with pytest.raises(ValueError):
        list(hindcast_frames(fake_recipe, ISSUES, horizon_hours=24))
