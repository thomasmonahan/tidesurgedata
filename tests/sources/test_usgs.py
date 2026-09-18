"""Spec and contract tests for the USGS adapter (BL-06)."""

import pandas as pd
import pytest

from tidesurgedata.meta import SeriesMeta
from tidesurgedata.sources.usgs import USGS

from ..contract_suite import SourceContractTests

pytest.importorskip("dataretrieval", reason='requires the "usgs" extra')

STUB = pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="BL-06")


def make_source():
    return USGS("05427718", parameter="discharge")


# @pytest.mark.skip(reason="BL-06: needs cassette")
@pytest.mark.vcr
class TestUSGSContract(SourceContractTests):
    @pytest.fixture
    def source(self):
        return make_source()

    @pytest.fixture
    def window(self):
        return ("2024-03-01T00:00Z", "2024-03-02T00:00Z")


# @STUB
@pytest.mark.vcr
def test_metadata():
    meta = make_source().metadata()
    assert meta.source == "usgs"
    assert meta.variable == "discharge"
    assert meta.units == "m3 s-1"
    assert meta.datum is None


@pytest.mark.live
@pytest.mark.enable_socket
def test_fetch():
    source = USGS("05427718", parameter="discharge")

    series = source.fetch(
        "2024-03-01T00:00Z",
        "2024-03-02T00:00Z",
    )

    assert len(series) == 96
    assert series.name == "discharge"
    assert str(series.index.tz) == "UTC"


@pytest.mark.live
@pytest.mark.enable_socket
def test_fetch_stage():
    source = USGS("05427718", parameter="stage")

    series = source.fetch(
        "2024-03-01T00:00Z",
        "2024-03-02T00:00Z",
    )

    assert len(series) == 96
    assert series.name == "stage"
    assert str(series.index.tz) == "UTC"


# @STUB
@pytest.mark.vcr
def test_find_stations():
    stations = USGS.find_stations(40.73, -74.10, radius_km=10, variable="discharge")
    assert stations and all(isinstance(m, SeriesMeta) for m in stations)
    assert all(m.source == "usgs" for m in stations)


@pytest.mark.live
@pytest.mark.enable_socket
# @STUB
def test_live_smoke():
    end = pd.Timestamp.now(tz="UTC").floor("1h") - pd.Timedelta("2D")
    series = make_source().fetch(end - pd.Timedelta("1D"), end)
    assert len(series) > 0
