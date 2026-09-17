"""Committed cassettes are free of secrets and within the size limit; the checker works."""

import pytest

from .cassette_check import CASSETTE_ROOT, MAX_BYTES, check_file, check_paths


def test_committed_cassettes_are_clean():
    assert check_paths([CASSETTE_ROOT]) == []


CLEAN = """\
interactions:
- request:
    body: null
    headers:
      User-Agent: [python-requests/2.34]
    method: GET
    uri: https://example.org/data?station=1&start=2024-01-01
  response:
    body: {string: '{"ok": true}'}
    headers: {}
    status: {code: 200, message: OK}
version: 1
"""


@pytest.fixture
def cassette(tmp_path):
    def write(text):
        path = tmp_path / "cassette.yaml"
        path.write_text(text)
        return path

    return write


def test_clean_cassette(cassette):
    assert check_file(cassette(CLEAN)) == []


def test_unfiltered_header(cassette):
    text = CLEAN.replace("User-Agent", "Authorization")
    assert any("'authorization' not filtered" in p for p in check_file(cassette(text)))


def test_unfiltered_query_parameter(cassette):
    text = CLEAN.replace("station=1", "api_key=abc123")
    assert any("'api_key' not filtered" in p for p in check_file(cassette(text)))


def test_token_pattern(cassette):
    text = CLEAN.replace('{"ok": true}', "ghp_" + "a" * 36)
    assert any("GitHub token" in p for p in check_file(cassette(text)))


def test_secret_environment_value(cassette, monkeypatch):
    monkeypatch.setenv("API_USGS_PAT", "s3cret-value-123")
    text = CLEAN.replace('{"ok": true}', "s3cret-value-123")
    assert any("API_USGS_PAT" in p for p in check_file(cassette(text)))


def test_size_limit(cassette):
    text = CLEAN.replace('{"ok": true}', "x" * (MAX_BYTES + 1))
    assert any("byte limit" in p for p in check_file(cassette(text)))


def test_invalid_yaml(cassette):
    assert any("not valid YAML" in p for p in check_file(cassette("interactions: [")))
