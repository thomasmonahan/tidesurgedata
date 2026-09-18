# Inputs and outputs: US MVP issues

A practical companion to the docstrings for the issues on the US path. For each one: what your
code is handed, what it must give back, a worked example with real output, the common errors, and
how to know you are done.

| Issue | What you implement | Spec tests |
|---|---|---|
| [#6 BL-05](https://github.com/thomasmonahan/tidesurgedata/issues/6) | `sources/noaa_coops.py` | `tests/sources/test_noaa_coops.py` |
| [#7 BL-06](https://github.com/thomasmonahan/tidesurgedata/issues/7) | `sources/usgs.py` | `tests/sources/test_usgs.py` |
| [#13 BL-12](https://github.com/thomasmonahan/tidesurgedata/issues/13) | `Recipe.training_frame`, `Recipe.provenance` | `tests/spec/test_recipe_frames.py` |
| [#14 BL-13](https://github.com/thomasmonahan/tidesurgedata/issues/14) | `Recipe.forecast_frame` | `tests/spec/test_recipe_frames.py` |
| [#20 BL-19](https://github.com/thomasmonahan/tidesurgedata/issues/20) | `examples/us_demo.py` | none (runs live) |
| [#16 BL-15](https://github.com/thomasmonahan/tidesurgedata/issues/16) | `sources/dynamical.py` analysis | `tests/sources/test_dynamical.py` |
| [#17 BL-16](https://github.com/thomasmonahan/tidesurgedata/issues/17) | `sources/dynamical.py` forecasts | `tests/sources/test_dynamical.py` |
| [#18 BL-17](https://github.com/thomasmonahan/tidesurgedata/issues/18) | `discovery.find_stations` | `tests/spec/test_discovery.py` |

The docstring in the code is the specification; this guide adds shape, dtype and worked examples.
Where they disagree, the docstring wins — tell the channel so this file gets fixed.

---

## 0. The shared rules, concretely

Everything in the package moves as one of three objects.

**A series** — one variable, one station:

```
time
2024-06-01 00:00:00+00:00    0.755834
2024-06-01 00:06:00+00:00    0.823905
2024-06-01 00:12:00+00:00    0.917377
Freq: 6min, Name: water_level, dtype: float64
```

- index: `DatetimeIndex`, **UTC**, strictly increasing, no duplicates (microsecond or nanosecond
  resolution both fine)
- values: `float64`, missing is `NaN`, never `inf`
- `series.name` **equals** `meta.variable`

**`SeriesMeta`** — what the series *is* (frozen dataclass, validated on construction):

```python
SeriesMeta(
    source="noaa_coops",  # your registry name
    station_id="8518750",
    variable="water_level",  # also the series name
    lat=40.7003,
    lon=-74.0139,  # lat [-90, 90], lon [-180, 360)
    units="m",  # canonical only: m, m s-1, m3 s-1, Pa, K, kg m-2 s-1, W m-2, 1
    datum="MSL",  # required for water_level and stage, else None
    sampling="instantaneous",  # or "window_mean" -> then window and label are required
    window=None,
    label=None,  # e.g. pd.Timedelta("15min"), "start"|"centre"|"end"
    licence="US Government Work (public domain)",
    attribution="Data from NOAA/NOS/CO-OPS",
    url="https://tidesandcurrents.noaa.gov/stationhome.html?id=8518750",
    name="The Battery, NY",
    extra={"product": "water_level", "interval": "6"},  # str -> str only
)
```

**`FetchRecord`** — provenance of one retrieval. `BaseSource` builds it for you:

```json
{
  "start": "2024-06-01T00:00:00+00:00",
  "end": "2024-06-03T00:00:00+00:00",
  "retrieved_at": "2026-09-18T09:58:38+00:00",
  "quality": "verified",
  "n_values": 480,
  "n_missing": 0,
  "request": {"station": "FAKE-TG", "begin": "...", "end": "...", "n_requests": "1"}
}
```

Two conventions that are easy to get wrong:

- **Ranges are half-open**, `[start, end)`. `end` itself is never included.
- **Lags are in hours and negative means the past.** A column `discharge_lag-24h` holds, at time
  `t`, the discharge at `t - 24h`.

Check your own output any time:

```python
from tidesurgedata.contract import validate_series, validate_frame

validate_series(series, meta)  # raises ContractError naming the broken rule
validate_frame(df, "observations", feature_columns)
```

### Canonical variable names

Agreed at kickoff. Use these names so recipes are portable across providers; add to this table
rather than introducing a synonym.

| Quantity | Variable | Units | Notes |
|---|---|---|---|
| Sea level | `water_level` | `m` | needs `datum` |
| River stage | `stage` | `m` | needs `datum` |
| River discharge | `discharge` | `m3 s-1` | |
| Surface pressure | `pressure_surface` | `Pa` | |
| Wind components | `wind_u_10m`, `wind_v_10m` | `m s-1` | never speed/direction |
| Air temperature | `air_temperature` | `K` | |

---

## 1. Adapters: what `BaseSource` does for you

Both adapter issues implement the same three methods. **You do not write chunking, retries,
sorting, de-duplication, slicing or validation** — `BaseSource.fetch_with_record` does all of it:

```
fetch(start, end)
  └─ to_utc(start), to_utc(end)            reject naive times
  └─ split_range(..., cls.max_request)     one chunk per provider request limit
  └─ your _fetch(chunk_start, chunk_end)   ← the only network code you write
  └─ concat, drop duplicate timestamps (keep last), sort, slice to [start, end)
  └─ set series.name = meta.variable
  └─ combine chunk qualities ("mixed" if they differ)
  └─ validate_series(...)                  ← your output is checked here
  └─ build FetchRecord
```

So `_fetch` may return slightly too much data, in any order — but it must return the right
**units**, **time zone** and **dtype**.

### `metadata(self) -> SeriesMeta`

| | |
|---|---|
| **Input** | nothing but `self` (the constructor parameters) |
| **Output** | one `SeriesMeta` describing the series this instance produces |
| **May it call the network?** | yes (station metadata endpoint); cache it on the instance if you like |

### `_fetch(self, start, end) -> tuple[pd.Series, Quality, dict]`

| | |
|---|---|
| **Input** | `start`, `end`: UTC `pd.Timestamp`, half-open, already chunked to `max_request` |
| **Output 1** | `pd.Series`, UTC index, `float64`, **canonical units**, name irrelevant (overwritten) |
| **Output 2** | `Quality`: `"verified"`, `"preliminary"`, `"mixed"` or `"unknown"` |
| **Output 3** | `dict[str, str]` of **non-secret** request parameters — never an API key |

Missing values come back as `NaN` rows, not as absent timestamps, where the provider tells you a
value is missing.

### `find_stations(cls, lat, lon, radius_km, variable=None) -> list[SeriesMeta]`

A **classmethod**: it runs without an instance. Returns one `SeriesMeta` per station (and variable)
within `radius_km` of the point, `[]` if none. `tidesurgedata.sources.base.haversine_km` computes
the distance.

### Class attributes to set

| Attribute | Meaning | If you get it wrong |
|---|---|---|
| `max_request` | longest span in one provider request | requests fail or silently truncate |
| `latency` | typical delay before data is available | forecast frames leak data or refuse valid horizons |
| `adapter_version` | bump when output could change | stale cache entries later (BL-04) |

---

## 2. #6 BL-05 — NOAA CO-OPS

```python
NOAACoops(station_id="8518750", product="water_level", datum="MSL", interval=None)
```

| Parameter | Type | Notes |
|---|---|---|
| `station_id` | `str` | CO-OPS station, e.g. `"8518750"` (The Battery, NY) |
| `product` | `str` | `water_level`, `wind`, `air_pressure`; `predictions` for reference only |
| `datum` | `str` | mandatory for water level: `MSL`, `MLLW`, `NAVD`, … |
| `interval` | `str \| None` | provider interval; `None` = product default |

All parameters must stay **JSON-serialisable** — recipes are saved with `to_spec()`.

**Products to variables** (fill the gaps as you go):

| `product` | `variable` | Provider units | Convert to |
|---|---|---|---|
| `water_level` | `water_level` | m (metric) | `m`, with `datum` |
| `air_pressure` | `pressure_surface` | hPa/mbar | `Pa` (`units.convert(x, "hPa", "Pa")`) |
| `wind` | `wind_u_10m`, `wind_v_10m` | speed + direction | `m s-1` components — **never** speed/direction |

**Request limits** — chunk by interval: 1-minute data 4 days, 6-minute data 1 month, hourly data
1 year. `max_request` is a class attribute but the limit depends on `interval`, so override it as a
property on the dataclass and note which rule applied.

**Quality:** water level is `"preliminary"` until verified (monthly), then `"verified"`. Let
`BaseSource` produce `"mixed"` by returning different qualities per chunk.

**Other facts:** request times in GMT (`time_zone=gmt`); no API key; attribution to NOAA CO-OPS;
US Government data.

**Worked example of what `metadata()` must return** (this exact object passes validation):

```python
SeriesMeta(
    source="noaa_coops",
    station_id="8518750",
    variable="water_level",
    lat=40.7003,
    lon=-74.0139,
    units="m",
    datum="MSL",
    sampling="instantaneous",
    window=None,
    label=None,
    licence="US Government Work (public domain)",
    attribution="Data from NOAA/NOS/CO-OPS",
    url="https://tidesandcurrents.noaa.gov/stationhome.html?id=8518750",
    name="The Battery, NY",
    extra={"product": "water_level", "interval": "6"},
)
```

and the series `_fetch` returns for that station:

```
time
2024-06-01 00:00:00+00:00    0.412
2024-06-01 00:06:00+00:00    0.401
2024-06-01 00:12:00+00:00    0.388
Name: water_level, dtype: float64
```

**Done when:** `tests/sources/test_noaa_coops.py` passes with the `xfail`/`skip` markers removed,
the contract suite replays from a recorded cassette (see `docs/dev/recording-cassettes.md`), and the
nightly live smoke test passes.

---

## 3. #7 BL-06 — USGS

```python
USGS(site_id="01376500", parameter="discharge")
```

| Parameter | Type | Notes |
|---|---|---|
| `site_id` | `str` | USGS site number, keep the leading zero — it is a string |
| `parameter` | `str` | `"discharge"` → `m3 s-1`, `"stage"` → `m` (needs a datum) |

**Use `dataretrieval.waterdata`** (extra `usgs`, `dataretrieval>=1.1.0`). **Not** the legacy `nwis`
module: WaterServices is scheduled for decommissioning in Q1 2027.

**Lazy import**, so `import tidesurgedata` stays light — this is enforced by
`tests/test_import_isolation.py`:

```python
def _waterdata():
    try:
        from dataretrieval import waterdata
    except ImportError as exc:
        raise ImportError(
            "The USGS adapter needs the 'usgs' extra: pip install 'tidesurgedata[usgs]'"
        ) from exc
    return waterdata
```

**Units:** convert ft³/s → `m3 s-1` and ft → `m` with `tidesurgedata.units.convert`.

**Quality:** approved → `"verified"`; provisional → `"preliminary"`.

**Secrets:** the API key comes from the environment variable `API_USGS_PAT` only. It must never
appear in `to_spec()` output, in `FetchRecord.request`, in a cassette or in a log line. The cassette
checker (`python tests/cassette_check.py`) fails if the value of `API_USGS_PAT` shows up in a
recording, so record with the variable exported.

**Still to confirm and document in your PR:** `max_request` and `latency` for the endpoints you use.

**Done when:** `tests/sources/test_usgs.py` passes with markers removed, a cassette with no key in
it, and the live smoke test passes nightly.

---

## 4. #13 BL-12 — `training_frame` and `provenance`

```python
Recipe.training_frame(self, start, end) -> pd.DataFrame
```

| | |
|---|---|
| **Input** | `start`, `end`: timezone-aware, half-open `[start, end)`; naive input raises `ValueError` |
| **Output** | one row per grid point, target first, then features in `recipe.feature_columns` order |

Real output for the fake recipe in `tests/conftest.py`:

```
                           observations  discharge_lag-24h  discharge_lag-12h  pressure_lag0h
time
2024-06-01 00:00:00+00:00         0.756             22.777             22.621      101391.463
2024-06-01 01:00:00+00:00         1.387             22.764             22.608      101328.193
2024-06-01 02:00:00+00:00         1.668             22.751             22.596      101264.139

shape: (168, 4)   dtypes: all float64   index: datetime64[us, UTC], freq 1h
first: 2024-06-01 00:00:00+00:00   last: 2024-06-07 23:00:00+00:00
```

Rules, in the order they apply:

1. **Resolve first:** call `self.resolved()` so gridded drivers with no location take the target's
   location.
2. **Grid:** `timeutil.regular_grid(start, end, self.freq)` — points are multiples of `freq` since
   the epoch, so frames from different windows line up.
3. **Target column** is named `self.target_column_name` (the target's variable name, or
   `target_column` if set — RTide wants `"observations"`). Put it on the grid with
   `to_grid(..., how=self.target_how, max_gap=None)`. **Never interpolate the target.**
4. **Drivers:** fetch each over the window **widened by its lags** (and by a grid step, so the
   centred window at the edges is complete), then `to_grid(..., d.how, d.max_gap)`, then
   `materialise_lags(gridded, d.name, d.lags_hours)`, then reindex onto the grid.
5. **Keep NaN rows.** Consumers decide what to drop; the frame must still have one row per grid
   point.
6. **Validate** with `validate_frame(df, self.target_column_name, self.feature_columns)` before
   returning.

`provenance()` returns the `FetchRecord`s of the **last** frame built, target first:

```
fake_tide_gauge    water_level       2024-05-31 23:00 -> 2024-06-08 01:00  n=1700 quality=verified
fake_river         discharge         2024-05-30 23:00 -> 2024-06-08 01:00  n=776  quality=verified
fake_met           pressure_surface  2024-05-31 23:00 -> 2024-06-08 01:00  n=170  quality=verified
```

Note the river was fetched from 24 h before `start`: that is the lag widening in rule 4. `Recipe` is
a frozen dataclass, so store the records with
`object.__setattr__(self, "_provenance", records)`.

**Pitfalls**

- Building the frame column by column with `pd.DataFrame(dict)` preserves insertion order — build
  the target first, then features in `feature_columns` order, or `validate_frame` will reject it.
- An all-NaN column indicates a fetch window that is too narrow; check the lag widening rather
  than filling the values.
- Empty or inverted ranges (`end <= start`) raise `ValueError`.

---

## 5. #14 BL-13 — `forecast_frame`

```python
Recipe.forecast_frame(self, issued, horizon_hours, member=None) -> pd.DataFrame
```

| | |
|---|---|
| **Input** | `issued`: timezone-aware issue time; `horizon_hours`: float; `member`: ensemble member, default `"control"` |
| **Output** | same columns as the training frame, index over **`(issued, issued + horizon]`**, target column all `NaN` |

```
                           observations  discharge_lag-24h  discharge_lag-12h  pressure_lag0h
time
2024-06-08 01:00:00+00:00           NaN             20.713             20.577      100154.352
2024-06-08 02:00:00+00:00           NaN             20.701             20.565      100174.113
2024-06-08 03:00:00+00:00           NaN             20.690             20.554      100200.228

shape: (11, 4)   first: 2024-06-08 01:00:00+00:00   last: 2024-06-08 11:00:00+00:00
```

Note the index differs from `training_frame`: the issue time itself is **excluded**, and
`issued + horizon` is **included**.

Rules:

1. **Refuse impossible horizons first:**
   ```python
   recipe.forecast_frame(issued, horizon_hours=12)
   # ValueError: horizon exceeds max lead time   (max_lead_time is 11h for the fake recipe)
   ```
   `max_lead_time` is already implemented (ADR 0003): for each driver without a forecast source,
   `min(-lag) - latency`, minimised over those drivers; `None` means unlimited.
2. **As-of discipline (ADR 0006):** for each driver, the cutoff is `issued - source.latency`.
   Observed and analysis data may be used **only up to that cutoff** — truncate the fetched series
   at the cutoff before gridding, not after lagging.
3. **Forecast drivers:** for valid times after the cutoff, take values from
   `driver.forecast.fetch_forecast(issued, horizon, members=[member or "control"])`, whose
   `.values` is a DataFrame of member columns indexed by valid time. Concatenate with the truncated
   observations, keeping the forecast where both exist.
4. **Target column is all NaN**, but must be present, `float64`, and first.
5. **Validate** with `validate_frame` before returning.

**Common error:** `materialise_lags` keeps its input index. If the gridded driver series ends at
the cutoff, every lagged value at future valid times is NaN. **Extend the gridded series to the end
of the forecast grid (with NaN) before lagging.**

**How leakage is tested:** the spec test builds the frame twice, once with the normal source and
once with a source truncated at `issued - latency`, and requires the two frames to be identical. If
your implementation reaches past the cutoff, they differ and the test fails.

---

## 6. #20 BL-19 — US demo script

`examples/us_demo.py`, runnable end to end against live APIs:

```bash
python examples/us_demo.py
```

It should:

1. Build a recipe: CO-OPS water level as target, USGS discharge and CO-OPS air pressure as drivers,
   `freq="1h"`, `target_column="observations"`.
2. Print `recipe.feature_columns` and `recipe.max_lead_time` so the lead-time limit is visible.
3. Build a training frame over a few months and print `df.head()`, `df.shape` and the NaN count per
   column.
4. Build a forecast frame at a recent issue time within `max_lead_time` and print it.
5. Print the provenance: per source, the range fetched, quality, and **the licence and attribution
   text** — the demo doubles as our attribution example.
6. Save `recipe.to_json()` next to the outputs so the dataset definition is reproducible.

Keep it a script with a `main()`, no argument parsing needed, station identifiers as constants at
the top with a comment on why that pair was chosen (the river should plausibly affect that gauge).

**Expected horizon:** every driver without a forecast source caps the lead time at
`min(-lag) - latency`. A driver at lag 0 therefore caps it at `-latency`, i.e. hindcast only, however
fast the provider reports. So give each observed driver a negative lag (e.g. pressure at `-12`), and
state the resulting `max_lead_time` in the script's docstring. Long horizons need forecast drivers
(BL-15/16).

---

## 7. #16 BL-15 — dynamical.org analysis at a point

```python
Dynamical(
    dataset="noaa-gfs-analysis", variable="pressure_surface", lat=51.5, lon=-3.0, method="nearest"
)
```

| Parameter | Type | Notes |
|---|---|---|
| `dataset` | `str` | dataset identifier from the STAC catalog, e.g. `"noaa-gfs-analysis"` |
| `variable` | `str` | **canonical** name (section 0), not the provider's variable name |
| `lat`, `lon` | `float \| None` | sample point; `None` means "not yet located" |
| `method` | `str` | `"nearest"` first; `"linear"` later if wanted |

`Dynamical` subclasses `GriddedSource`, so it inherits `with_location(lat, lon)`, which returns a
copy with a new location. `Recipe.resolved()` calls it for any gridded driver left unlocated, so a
recipe can say "pressure at the target station" without repeating coordinates. Keep `lat`/`lon` as
plain floats: `to_spec()` must stay JSON-serialisable.

**Lazy imports.** `xarray`, `zarr`, `icechunk` and `pystac` must be imported **inside** functions,
never at module level, and a missing dependency must name the extra:

```python
def _open(dataset: str):
    try:
        import icechunk, xarray as xr  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The dynamical.org adapter needs the 'met' extra: pip install 'tidesurgedata[met]'"
        ) from exc
    ...
```

`tests/test_import_isolation.py` imports every module in a subprocess and fails if any of those
packages is loaded.

### `metadata()`

The returned `SeriesMeta` describes **the grid cell actually sampled**, not the point requested:

| Field | Value |
|---|---|
| `source` | `"dynamical"` |
| `station_id` | a stable label for the grid point, e.g. `"noaa-gfs-analysis@51.500,-3.000"` |
| `variable` | canonical name; `units` the matching canonical unit (`Pa`, `K`, `m s-1`) |
| `lat`, `lon` | coordinates of the selected cell (the spec test allows 0.5° from the request) |
| `datum` | `None` |
| `sampling` | `"instantaneous"` for instantaneous fields; `"window_mean"` for averaged or accumulated fields, then `window` and `label` are required and must come from the dataset attributes |
| `licence` | `"CC-BY-4.0"`, plus ECMWF Terms of Use for ECMWF datasets |
| `attribution` | dataset attribution text from the catalog |
| `extra` | useful provenance: requested point, grid resolution, STAC item id |

Getting `sampling`, `window` and `label` right matters: `to_grid` shifts window means onto centred
labels, and a wrong label silently shifts a driver in time (ADR 0004).

### `_fetch(start, end)`

Same contract as any adapter (section 1): a `float64`, UTC-indexed series in canonical units for
`[start, end)`, its quality, and the non-secret request parameters. Open the dataset lazily, select
the nearest cell, slice the time axis, then convert units. Record `dataset`, `variable`, `method`
and the selected coordinates in the request dictionary.

Also set, with evidence in the PR:

- `max_request` — chunk long ranges so a single read stays reasonable.
- `latency` — how long after valid time the analysis appears. This feeds the lead-time rule.
- `adapter_version` — bump if a change alters returned values.

Each dataset has a different **archive start date**; a request before it should come back empty or
raise a clear error, not silently return NaN for years.

**Done when:** the BL-15 tests in `tests/sources/test_dynamical.py` pass with markers removed, the
contract suite replays from a recorded fixture, and `tests/test_import_isolation.py` still passes.

---

## 8. #17 BL-16 — dynamical.org forecasts and ensembles

Two more methods on the same class, making it a `ForecastSource`. Forecast datasets are indexed by
`init_time` and `lead_time` (and a member dimension for ensembles), where
`valid_time = init_time + lead_time`.

### `init_times(start, end) -> pd.DatetimeIndex`

Initialisation times available in `[start, end)`: UTC, sorted, unique, named `init_time`. Half-open,
so a run exactly at `end` is excluded. The contract suite checks this directly.

### `fetch_forecast(issued, horizon, members=None) -> Forecast`

| | |
|---|---|
| **`issued`** | timezone-aware issue time |
| **`horizon`** | `pd.Timedelta`; valid times must extend to at least `issued + horizon` |
| **`members`** | list of member labels, or `None` for all |

Returns a `Forecast` dataclass:

```python
Forecast(
    values=...,  # DataFrame: index = valid_time (UTC), columns = member labels (str)
    init_time=...,  # the initialisation actually used (UTC Timestamp)
    record=...,  # FetchRecord, meta equal to self.metadata()
)
```

**The rule that matters (ADR 0006):** use the **latest initialisation available by
`issued - self.latency`**, and never one after it. In code:

```python
cutoff = to_utc(issued) - self.latency
candidates = self.init_times(cutoff - pd.Timedelta("10D"), cutoff + pd.Timedelta(1, "ns"))
init = candidates.max()
```

The contract suite asserts exactly this, so an off-by-one on the half-open bound will fail.

**Member labels are strings.** A deterministic dataset uses the single column `"control"`; GEFS-style
ensembles use `"control"` plus `"1"`, `"2"`, … Requesting an unknown member raises `ValueError`.
Validate before returning:

```python
from tidesurgedata.contract import validate_forecast

validate_forecast(forecast, self.metadata())
```

`FakeMet` in `tests/conftest.py` is a working reference implementation of this behaviour: 6-hourly
initialisations, 4-hour latency, `"control"` plus ten members. Read it before starting, and compare
your output shape against `FakeMet(...).fetch_forecast("2024-01-10T12:00Z", pd.Timedelta("48h"))`.

**Open question to settle in the PR:** how to test this without large downloads. VCR cassettes
suit plain HTTP, but Icechunk/Zarr reads are many ranged requests and may record badly. A small
committed fixture (one point, a few times, saved as netCDF or a tiny Zarr store) is likely better.
Whatever you choose, unit tests must not touch the network, and the cassette or fixture must stay
well under 500 kB (`python tests/cassette_check.py`).

**Done when:** the BL-16 tests pass with markers removed, `TestDynamicalForecastContract` runs from
a fixture, and the live smoke test passes nightly.

---

## 9. #18 BL-17 — cross-provider `find_stations`

```python
tidesurgedata.discovery.find_stations(lat, lon, radius_km, variables=None, sources=None)
```

| Parameter | Type | Notes |
|---|---|---|
| `lat`, `lon` | `float` | search centre in degrees |
| `radius_km` | `float` | positive search radius |
| `variables` | `Sequence[str] \| None` | keep only these canonical variable names; `None` = all |
| `sources` | `Sequence[str] \| None` | registry names to query; `None` = all registered; unknown name raises `KeyError` |

**Returns** a `DataFrame` with one row per station **and variable**:

- columns: every `SeriesMeta` field in dataclass order, then `distance_km`;
- `window` stays a `pd.Timedelta` (this is a DataFrame, not JSON);
- sorted by `distance_km` ascending, ties broken by `source` then `station_id`;
- a fresh `RangeIndex`;
- when nothing is found: an **empty DataFrame with those same columns**, not an empty one with no
  columns.

```python
columns = [f.name for f in dataclasses.fields(SeriesMeta)] + ["distance_km"]
```

**Iterating the registry:**

```python
from tidesurgedata.sources.registry import registered_sources
from tidesurgedata.sources.base import haversine_km
```

`registered_sources()` returns `{name: class}` for every registered adapter, including stubs. Any
adapter whose `find_stations` raises `NotImplementedError` is **skipped with a `UserWarning`**, not
propagated: discovery must keep working while adapters are unimplemented. That behaviour is asserted
by `tests/spec/test_discovery.py`, which runs against the fake sources.

Use `haversine_km` for `distance_km` so it matches the adapters' own radius filtering.

**Done when:** `tests/spec/test_discovery.py` passes with markers removed, and the result is
demonstrated across two real adapters once BL-05 and BL-06 land.

---

## 10. Checking your work

```bash
pytest -m "not live and not rtide"    # everything, no network
pytest tests/spec/test_recipe_frames.py -k training   # just your part
ruff check --fix . && ruff format .   # before every commit; pre-commit install does this for you
```

Reading order when a test fails:

1. `ContractError` names the rule and the source: `[noaa_coops:8518750:water_level] contract rule
   'values.float64' violated: ...`.
2. Spec tests in `tests/spec/` and `tests/sources/` are the acceptance criteria — read the test, not
   just its failure.
3. `@pytest.mark.xfail(..., strict=True)` means CI fails once your code works. **Removing those
   markers is part of your PR.**

Fake sources are the fastest way to experiment without network access:

```python
from tidesurgedata.sources.fake import FakeMet, FakeRiver, FakeTideGauge

gauge = FakeTideGauge(pressure=FakeMet(), river=FakeRiver())
series, record = gauge.fetch_with_record("2024-06-01T00:00Z", "2024-06-03T00:00Z")
```

They are deterministic: the same window always gives the same values, and fetching `[a, c)` equals
fetching `[a, b)` then `[b, c)`.
