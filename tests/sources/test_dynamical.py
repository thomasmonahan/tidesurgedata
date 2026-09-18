"""Spec and contract tests for the dynamical.org adapter (BL-15 analysis, BL-16 forecasts)."""

import pandas as pd
import pytest

from tidesurgedata.sources.base import ForecastSource, GriddedSource
from tidesurgedata.sources.dynamical import Dynamical

from ..contract_suite import ForecastSourceContractTests, SourceContractTests

BL15 = pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="BL-15")
BL16 = pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="BL-16")


def analysis():
    return Dynamical("noaa-gfs-analysis", "pressure_surface", lat=51.5, lon=-3.0)


def forecast():
    return Dynamical("noaa-gefs-forecast-35-day", "pressure_surface", lat=51.5, lon=-3.0)


def test_is_gridded_forecast_source():
    assert isinstance(analysis(), GriddedSource)
    assert isinstance(analysis(), ForecastSource)
    unlocated = Dynamical("noaa-gfs-analysis", "pressure_surface")
    assert unlocated.with_location(1.0, 2.0) == Dynamical(
        "noaa-gfs-analysis", "pressure_surface", lat=1.0, lon=2.0
    )


# @pytest.mark.skip(reason="BL-15: needs cassette")
@pytest.mark.vcr
class TestDynamicalContract(SourceContractTests):
    @pytest.fixture
    def source(self):
        return analysis()

    @pytest.fixture
    def window(self):
        return ("2024-01-01T00:00Z", "2024-01-03T00:00Z")


# @pytest.mark.skip(reason="BL-16: needs cassette")
@pytest.mark.enable_socket
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


# @BL15
def test_metadata():
    meta = analysis().metadata()
    assert meta.source == "dynamical"
    assert meta.variable == "pressure_surface"
    assert meta.units == "Pa"
    assert (meta.lat, meta.lon) == pytest.approx((51.5, -3.0), abs=0.5)
    assert "CC" in meta.licence and "BY" in meta.licence


# @BL16
def test_forecast_members():
    fc = forecast().fetch_forecast("2024-01-10T12:00Z", pd.Timedelta("24h"))
    assert "control" in fc.values.columns or len(fc.values.columns) > 1
    assert all(isinstance(c, str) for c in fc.values.columns)


@pytest.mark.live
@pytest.mark.enable_socket
# @BL15
def test_live_smoke_analysis():
    end = pd.Timestamp.now(tz="UTC").floor("1D") - pd.Timedelta("3D")
    assert len(analysis().fetch(end - pd.Timedelta("1D"), end)) > 0


@pytest.mark.live
@pytest.mark.enable_socket
# @BL16
def test_live_smoke_forecast():
    issued = pd.Timestamp.now(tz="UTC").floor("1D") - pd.Timedelta("3D")
    fc = forecast().fetch_forecast(issued, pd.Timedelta("24h"))
    assert len(fc.values) > 0
