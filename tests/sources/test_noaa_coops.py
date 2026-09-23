"""Spec and contract tests for the NOAACoops adapter (BL-05)."""

import pandas as pd
import pytest

from tidesurgedata.meta import SeriesMeta
from tidesurgedata.sources.noaa_coops import NOAACoops

from ..contract_suite import SourceContractTests

STUB = pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="BL-05")


def make_source():
    return NOAACoops("8518750", product="water_level", datum="MSL")


@pytest.mark.vcr
class TestNOAACoopsContract(SourceContractTests):
    @pytest.fixture
    def source(self):
        return make_source()

    @pytest.fixture
    def window(self):
        return ("2024-01-01T00:00Z", "2024-01-03T00:00Z")


@pytest.mark.vcr
def test_metadata():
    meta = make_source().metadata()
    assert meta.source == "noaa_coops"
    assert meta.variable == "water_level"
    assert meta.units == "m"
    assert meta.datum == "MSL"
    assert "NOAA" in meta.attribution or "NOAA" in meta.licence


@pytest.mark.live
@pytest.mark.enable_socket
def test_find_stations():
    stations = NOAACoops.find_stations(40.70, -74.01, radius_km=5)
    assert stations and all(isinstance(m, SeriesMeta) for m in stations)
    assert all(m.source == "noaa_coops" for m in stations)


@pytest.mark.live
@pytest.mark.enable_socket
def test_live_smoke():
    end = pd.Timestamp.now(tz="UTC").floor("1h") - pd.Timedelta("2D")
    series = make_source().fetch(end - pd.Timedelta("1D"), end)
    assert len(series) > 0
