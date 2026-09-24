"""Spec and contract tests for the dynamical.org adapter (BL-15 analysis, BL-16 forecasts).

The unit tests run **offline** (ADR 0005): `Dynamical._collection` and `Dynamical._open_dataset`
are replaced with a small synthetic STAC collection and xarray dataset shaped like the real ones
(analysis: ``time``; forecast: ``init_time``, ``ensemble_member``, ``lead_time``). The tests
marked ``live`` are the only ones that reach dynamical.org and run in the nightly job.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from tidesurgedata.contract import validate_forecast
from tidesurgedata.sources.base import ForecastSource, GriddedSource
from tidesurgedata.sources.dynamical import ArchiveCoverageWarning, Dynamical

from ..contract_suite import ForecastSourceContractTests, SourceContractTests

xr = pytest.importorskip("xarray", reason="needs the 'met' extra")

LAT, LON = 51.5, -3.0
LATITUDES = np.array([52.0, 51.5, 51.0])
LONGITUDES = np.array([-3.5, -3.0, -2.5])
ANALYSIS_START = "2023-12-25"
FORECAST_INITS = pd.date_range("2024-01-05", "2024-01-12", freq="1D")
LEAD_TIMES = pd.timedelta_range("0h", "96h", freq="3h")


def _signal(times: pd.DatetimeIndex, offset: float = 0.0) -> np.ndarray:
    """Smooth synthetic surface pressure in Pa."""
    hours = np.asarray(
        (times - pd.Timestamp(ANALYSIS_START, tz="UTC")) / pd.Timedelta("1h"), dtype="float64"
    )
    return (101325.0 + 800.0 * np.sin(2 * np.pi * hours / 74.0) + offset).astype("float32")


def analysis_dataset() -> "xr.Dataset":
    times = pd.date_range(ANALYSIS_START, "2024-01-20", freq="1h", tz="UTC")
    values = _signal(times)[:, None, None] * np.ones((1, len(LATITUDES), len(LONGITUDES)))
    return xr.Dataset(
        {"pressure_surface": (("time", "latitude", "longitude"), values.astype("float32"))},
        coords={
            "time": times.tz_localize(None),
            "latitude": LATITUDES,
            "longitude": LONGITUDES,
        },
    )


def forecast_dataset(n_members: int = 5) -> "xr.Dataset":
    members = np.arange(n_members)
    shape = (len(FORECAST_INITS), n_members, len(LEAD_TIMES), len(LATITUDES), len(LONGITUDES))
    values = np.empty(shape, dtype="float32")
    for i, init in enumerate(FORECAST_INITS):
        valid = pd.DatetimeIndex(init.tz_localize("UTC") + LEAD_TIMES)
        for m in members:
            # Each member is the truth plus a member-specific offset growing with lead time.
            growth = np.asarray(LEAD_TIMES / pd.Timedelta("24h"), dtype="float32")
            values[i, m] = (_signal(valid) + 20.0 * m * growth)[:, None, None]
    return xr.Dataset(
        {
            "pressure_surface": (
                ("init_time", "ensemble_member", "lead_time", "latitude", "longitude"),
                values,
            )
        },
        coords={
            "init_time": FORECAST_INITS,
            "ensemble_member": members,
            "lead_time": LEAD_TIMES,
            "latitude": LATITUDES,
            "longitude": LONGITUDES,
        },
    )


class FakeCollection:
    """Minimal stand-in for the STAC collection of a dynamical.org dataset."""

    def __init__(self, dataset: str):
        self.id = dataset
        self.title = f"Synthetic {dataset}"
        self.extra_fields = {
            "cube:variables": {"pressure_surface": {"unit": "Pa", "type": "data"}},
            "attribution": "Synthetic data for tests",
        }

    def get_self_href(self) -> str:
        return f"https://stac.dynamical.org/{self.id}/collection.json"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Serve synthetic data instead of dynamical.org, so unit tests need no network."""
    datasets = {
        "noaa-gfs-analysis": analysis_dataset(),
        "noaa-gefs-forecast-35-day": forecast_dataset(),
    }
    monkeypatch.setattr(Dynamical, "_collection", lambda self: FakeCollection(self.dataset))
    monkeypatch.setattr(
        Dynamical, "_open_dataset", lambda self, collection=None: datasets[self.dataset]
    )
    return datasets


def analysis():
    return Dynamical("noaa-gfs-analysis", "pressure_surface", lat=LAT, lon=LON)


def forecast():
    return Dynamical("noaa-gefs-forecast-35-day", "pressure_surface", lat=LAT, lon=LON)


def test_is_gridded_forecast_source():
    assert isinstance(analysis(), GriddedSource)
    assert isinstance(analysis(), ForecastSource)
    unlocated = Dynamical("noaa-gfs-analysis", "pressure_surface")
    assert unlocated.with_location(1.0, 2.0) == Dynamical(
        "noaa-gfs-analysis", "pressure_surface", lat=1.0, lon=2.0
    )


class TestDynamicalContract(SourceContractTests):
    @pytest.fixture
    def source(self):
        return analysis()

    @pytest.fixture
    def window(self):
        return ("2024-01-01T00:00Z", "2024-01-03T00:00Z")


class TestDynamicalForecastContract(ForecastSourceContractTests):
    @pytest.fixture
    def source(self):
        return forecast()

    @pytest.fixture
    def issued(self):
        return "2024-01-10T12:00Z"

    @pytest.fixture
    def horizon(self):
        return pd.Timedelta("48h")


# --- BL-15: analysis ----------------------------------------------------------------------------


def test_metadata():
    meta = analysis().metadata()
    assert meta.source == "dynamical"
    assert meta.variable == "pressure_surface"
    assert meta.units == "Pa"
    assert (meta.lat, meta.lon) == pytest.approx((LAT, LON), abs=0.5)
    assert "CC" in meta.licence and "BY" in meta.licence


def test_metadata_requires_location():
    with pytest.raises(ValueError, match="location"):
        Dynamical("noaa-gfs-analysis", "pressure_surface").metadata()


def test_fetch_before_archive_start_warns():
    start = pd.Timestamp(ANALYSIS_START, tz="UTC") - pd.Timedelta("10D")
    with pytest.warns(ArchiveCoverageWarning, match="archive starts at 2023-12-25"):
        series = analysis().fetch(start, start + pd.Timedelta("12D"))
    assert series.index[0] == pd.Timestamp(ANALYSIS_START, tz="UTC")


def test_fetch_within_archive_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", ArchiveCoverageWarning)
        analysis().fetch("2024-01-01T00:00Z", "2024-01-02T00:00Z")


def test_fetch_is_half_open():
    start, end = pd.Timestamp("2024-01-01T00:00Z"), pd.Timestamp("2024-01-02T00:00Z")
    series = analysis().fetch(start, end)
    assert series.index[0] == start
    assert series.index[-1] == end - pd.Timedelta("1h")
    assert len(series) == 24


# --- BL-16: forecasts ---------------------------------------------------------------------------


def test_forecast_members():
    fc = forecast().fetch_forecast("2024-01-10T12:00Z", pd.Timedelta("24h"))
    assert list(fc.values.columns) == ["control", "1", "2", "3", "4"]
    assert all(isinstance(c, str) for c in fc.values.columns)
    validate_forecast(fc, forecast().metadata())


def test_forecast_uses_latest_available_initialisation():
    fc = forecast().fetch_forecast("2024-01-10T12:00Z", pd.Timedelta("24h"))
    assert fc.init_time == pd.Timestamp("2024-01-10T00:00Z")
    assert fc.values.index[0] == fc.init_time
    assert fc.values.index[-1] <= pd.Timestamp("2024-01-11T12:00Z")


def test_forecast_member_subset():
    fc = forecast().fetch_forecast("2024-01-10T12:00Z", pd.Timedelta("24h"), members=["control"])
    assert list(fc.values.columns) == ["control"]
    everything = forecast().fetch_forecast("2024-01-10T12:00Z", pd.Timedelta("24h"))
    pd.testing.assert_series_equal(fc.values["control"], everything.values["control"])


def test_forecast_unknown_member():
    with pytest.raises(ValueError, match="Unknown members"):
        forecast().fetch_forecast("2024-01-10T12:00Z", pd.Timedelta("24h"), members=["99"])


def test_forecast_rejects_naive_issue_time():
    with pytest.raises(ValueError, match="[Nn]aive"):
        forecast().fetch_forecast("2024-01-10T12:00", pd.Timedelta("24h"))


def test_forecast_before_archive_start():
    with pytest.raises(ValueError, match="no initialisation available"):
        forecast().fetch_forecast("2019-01-01T00:00Z", pd.Timedelta("24h"))


def test_members_differ_and_grow_with_lead_time():
    fc = forecast().fetch_forecast("2024-01-10T12:00Z", pd.Timedelta("72h"))
    spread = fc.values.std(axis=1)
    assert spread.iloc[0] == pytest.approx(0.0, abs=1e-6)  # all members agree at lead 0
    assert spread.iloc[-1] > spread.iloc[1]


def test_init_times_half_open():
    times = forecast().init_times("2024-01-06T00:00Z", "2024-01-09T00:00Z")
    assert list(times) == list(pd.date_range("2024-01-06", "2024-01-08", freq="1D", tz="UTC"))
    assert times.name == "init_time"


# --- live smoke tests (nightly only) -------------------------------------------------------------


@pytest.mark.live
@pytest.mark.enable_socket
def test_live_smoke_analysis(monkeypatch):
    monkeypatch.undo()  # use the real catalog and store
    end = pd.Timestamp.now(tz="UTC").floor("1D") - pd.Timedelta("3D")
    assert len(analysis().fetch(end - pd.Timedelta("1D"), end)) > 0


@pytest.mark.live
@pytest.mark.enable_socket
def test_live_smoke_forecast(monkeypatch):
    monkeypatch.undo()  # use the real catalog and store
    issued = pd.Timestamp.now(tz="UTC").floor("1D") - pd.Timedelta("3D")
    fc = forecast().fetch_forecast(issued, pd.Timedelta("24h"))
    assert len(fc.values) > 0
    assert "control" in fc.values.columns
