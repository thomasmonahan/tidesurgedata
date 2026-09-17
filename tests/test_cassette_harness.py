"""Example cassette proving the recording harness end to end (BL-02).

This test calls a public Environment Agency endpoint directly with ``requests``; it is not an
adapter (adapters are BL-05…BL-09). It sends a dummy API-key header so the test can prove that
``vcr_config`` filters credentials out of recorded cassettes.

Re-record (normally never needed)::

    pytest tests/test_cassette_harness.py --record-mode=rewrite --force-enable-socket
"""

import pytest
import requests

from .cassette_check import CASSETTE_ROOT

STATION_URL = "https://environment.data.gov.uk/flood-monitoring/id/stations/E72639"
DUMMY_KEY = "dummy-harness-key-not-a-secret"
CASSETTE = CASSETTE_ROOT / "test_cassette_harness" / "test_replays_recorded_response.yaml"


@pytest.mark.vcr
def test_replays_recorded_response():
    response = requests.get(STATION_URL, headers={"X-Api-Key": DUMMY_KEY}, timeout=30)
    response.raise_for_status()
    station = response.json()["items"]
    assert station["stationReference"] == "E72639"
    assert -90 <= station["lat"] <= 90


def test_cassette_does_not_contain_filtered_credentials():
    text = CASSETTE.read_text()
    assert DUMMY_KEY not in text
    assert "X-Api-Key" not in text


@pytest.mark.filterwarnings("ignore:A test tried to use socket:UserWarning")
def test_network_is_blocked_without_cassette():
    import pytest_socket

    with pytest.raises(pytest_socket.SocketBlockedError):
        requests.get(STATION_URL, timeout=5)
