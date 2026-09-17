# Recording cassettes

Unit tests never touch the network (ADR 0005). Adapter tests replay HTTP responses recorded with
[pytest-recording](https://github.com/kiwicom/pytest-recording) (built on vcrpy).

## Where cassettes live

`tests/cassettes/<test module name>/<test name>.yaml`, e.g.
`tests/cassettes/test_noaa_coops/TestNOAACoopsContract.test_fetch_obeys_contract.yaml`. The
directory is set by the `vcr_cassette_dir` fixture in `tests/conftest.py`.

## Worked example

`tests/test_cassette_harness.py` is a complete, minimal example: a `@pytest.mark.vcr` test that
calls a public Environment Agency endpoint, its recorded cassette in
`tests/cassettes/test_cassette_harness/`, and tests proving that credentials are filtered and that
the network is blocked without a cassette. Copy its pattern for adapter tests.

## Recording

1. Remove the `@pytest.mark.skip(reason="BL-xx: needs cassette")` marker from the adapter's
   `Test…Contract` class (it already has `@pytest.mark.vcr`).
2. Choose a **small** window (a day or two) and a stable, verified period.
3. Record, allowing sockets for this run only (`vcr_config` deliberately does not set
   `record_mode`, so the command-line option takes effect; without it, cassettes only replay):

   ```bash
   pytest tests/sources/test_noaa_coops.py --record-mode=once --force-enable-socket
   ```

   For APIs needing a key, export it first (e.g. `export API_USGS_PAT=...`); never write it to a
   file in the repository.
4. Re-run **without** network to confirm replay works:

   ```bash
   pytest tests/sources/test_noaa_coops.py
   ```

## Reviewing before committing

Run the cassette checker:

```bash
python tests/cassette_check.py
```

It fails on any cassette that is larger than 500 kB, contains an unfiltered credential header or
API-key query parameter, contains a well-known token format (GitHub, AWS, Slack, bearer tokens,
private keys), or contains the value of a secret environment variable that is set in your shell
(`API_USGS_PAT` or any variable ending `_PAT`, `_TOKEN`, `_API_KEY`, `_SECRET`, `_PASSWORD`). The
same check runs as a pre-commit hook on changed cassettes and in CI (`tests/test_cassettes.py`), so
record with your API keys still exported so the environment-variable check is meaningful.

Also review by eye:

- **Size:** keep each cassette small; shrink the window or the number of stations if needed.
- **Content:** no personal data; responses are provider data covered by the provider's licence.

## Filtering secrets

`vcr_config` in `tests/conftest.py` removes the credential headers and query parameters listed
in `tests/cassette_check.py` (`FILTER_HEADERS`: `authorization`, `x-api-key`, `api-key`, `cookie`,
`proxy-authorization`; `FILTER_QUERY_PARAMETERS`: `api_key`, `apikey`, `token`, `key`,
`access_token`). If a provider uses another name, add it to those tuples (so the checker also
enforces it) or override `vcr_config` in that adapter's test module:

```python
@pytest.fixture(scope="module")
def vcr_config(vcr_config):
    return {**vcr_config, "filter_headers": [*vcr_config["filter_headers"], "x-provider-key"]}
```

## Re-recording

Delete the affected cassette files and record again with `--record-mode=once`. If an adapter's
output changes, bump its `adapter_version`.

## Live smoke tests

Each adapter also has a `@pytest.mark.live` smoke test that hits the real API. They run in the
nightly job (`pytest -m live`), never in required checks.
