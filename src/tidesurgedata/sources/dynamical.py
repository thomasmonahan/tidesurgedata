"""dynamical.org weather data adapter (BL-15 analysis, BL-16 forecasts/ensembles).

Provider facts
--------------
- dynamical.org cloud-optimised weather data (GFS, GEFS, HRRR, ECMWF IFS ENS, AIFS) stored as
  Icechunk Zarr.
- Datasets are discovered through the STAC catalog ``https://stac.dynamical.org/catalog.json`` and
  opened with xarray (install the ``met`` extra: ``pip install "tidesurgedata[met]"``).
- Forecast datasets have ``init_time``, ``lead_time`` and (for ensembles) member dimensions.
- Licence CC BY 4.0, plus the ECMWF Terms of Use for ECMWF datasets.
- Archive start dates differ per dataset.

Implementation checklist
------------------------
- [ ] Import ``xarray``, ``zarr``, ``icechunk`` and ``pystac`` inside functions; raise
      ``ImportError`` naming the ``met`` extra.
- [ ] BL-15 ``metadata()``/``_fetch()``: analysis at a point (``method="nearest"``, later
      ``"linear"``); canonical units and variable names; vectors as ``u``/``v``; sampling
      convention from dataset attributes; licence and attribution per dataset.
- [ ] BL-15: confirm ``max_request`` and ``latency`` per dataset; archive start handling.
- [ ] BL-16 ``init_times()`` and ``fetch_forecast()``: latest ``init_time`` available by
      ``issued - latency``; ``valid_time = init_time + lead_time``; members as string labels
      (deterministic: ``"control"``).
- [ ] Small recorded fixture; contract and forecast contract tests; remove markers; CHANGELOG.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from tidesurgedata.contract import validate_forecast
from tidesurgedata.meta import Quality, SeriesMeta
from tidesurgedata.sources.base import FetchRecord, Forecast, GriddedSource
from tidesurgedata.sources.registry import register_source
from tidesurgedata.timeutil import TimeLike, to_utc

__all__ = ["CATALOG_URL", "Dynamical"]

#: STAC catalog describing every dynamical.org dataset.
CATALOG_URL = "https://stac.dynamical.org/catalog.json"

#: Dimension names used by ensemble datasets for their member axis.
MEMBER_DIMENSIONS = ("ensemble_member", "realization", "member", "number")

#: Label for the unperturbed member.
CONTROL = "control"


def _require_met() -> tuple:
    """Import the optional ``met`` dependencies, naming the extra if they are missing."""
    try:
        import icechunk
        import pystac
        import xarray as xr
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "The dynamical.org adapter needs the 'met' extra: pip install 'tidesurgedata[met]'"
        ) from exc
    return icechunk, pystac, xr


def _member_label(value: object) -> str:
    """Ensemble member coordinate value -> label: member 0 is the control member."""
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(value)
    return CONTROL if number == 0 else str(number)


@register_source("dynamical")
@dataclass(frozen=True)
class Dynamical(GriddedSource):
    """dynamical.org dataset sampled at a point; also a ``ForecastSource`` for forecast datasets.

    Parameters
    ----------
    dataset : str
        dynamical.org dataset identifier from the STAC catalog.
    variable : str
        Canonical variable name, e.g. ``"pressure_surface"``, ``"wind_u_10m"``.
    lat, lon : float, optional
        Sample location; ``None`` until located (``Recipe.resolved`` uses the target location).
    method : {"nearest", "linear"}
        Spatial sampling method.
    """

    dataset: str
    variable: str
    lat: float | None = None
    lon: float | None = None
    method: str = "nearest"

    # No documented request limit or guaranteed analysis latency.
    max_request = None
    latency = pd.Timedelta(0)

    # --- dataset access -------------------------------------------------------------------

    @classmethod
    def _catalog(cls):
        """Open the dynamical.org STAC catalog."""
        _, pystac, _ = _require_met()
        return pystac.Catalog.from_file(CATALOG_URL)

    def _collection(self):
        """STAC collection for this dataset.

        Tests replace this method to run offline.
        """
        collection = self._catalog().get_child(self.dataset)
        if collection is None:
            raise ValueError(f"Unknown dynamical.org dataset: {self.dataset}")
        return collection

    def _open_dataset(self, collection=None):
        """Open this dataset's Icechunk Zarr store as an xarray Dataset.

        Tests replace this method to run offline.
        """
        icechunk, _, xr = _require_met()
        collection = collection if collection is not None else self._collection()
        asset = collection.assets["icechunk-https"]
        repo = icechunk.Repository.open(icechunk.http_storage(asset.href))
        return xr.open_zarr(repo.readonly_session("main").store, chunks=None)

    def _located(self) -> tuple[float, float]:
        if self.lat is None or self.lon is None:
            raise ValueError(
                f"{type(self).__name__} has no location; call with_location(lat, lon) first."
            )
        return float(self.lat), float(self.lon)

    def metadata(self) -> SeriesMeta:
        """Return metadata for the selected dynamical.org variable and location."""
        lat, lon = self._located()
        collection = self._collection()
        variables = collection.extra_fields.get("cube:variables", {})
        if self.variable not in variables:
            raise ValueError(f"Variable {self.variable!r} is not available in {self.dataset!r}")

        variable = variables[self.variable]

        return SeriesMeta(
            source="dynamical",
            station_id=f"{self.dataset}:{lat},{lon}",
            variable=self.variable,
            lat=lat,
            lon=lon,
            units=variable["unit"],
            datum=None,
            sampling="instantaneous",
            window=None,
            label=None,
            licence=collection.extra_fields.get("license", "CC-BY-4.0"),
            attribution=collection.extra_fields.get("attribution", ""),
            url=collection.get_self_href() or "https://stac.dynamical.org/catalog.json",
            name=collection.title or self.dataset,
            extra={"dataset": self.dataset},
        )

    def _fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Series, Quality, dict]:
        """Fetch analysis data from dynamical.org for the user specified location and time range."""
        lat, lon = self._located()
        ds = self._open_dataset()

        # xarray uses timezone-naive datetime64 coordinates; tidesurgedata uses aware UTC.
        # The half-open [start, end) range is honoured by dropping the closing endpoint below.
        start_naive = start.tz_convert("UTC").tz_localize(None)
        end_naive = end.tz_convert("UTC").tz_localize(None)

        # Select only the requested variable, time range, and nearest grid point.
        data = (
            ds[self.variable]
            .sel(time=slice(start_naive, end_naive))
            .sel(latitude=lat, longitude=lon, method="nearest")
        )

        # Convert the one-dimensional xarray result to the pandas Series expected
        series = data.to_series().astype("float64")
        series.index = pd.to_datetime(series.index, utc=True)
        series = series[series.index < end]  # [start, end) is half-open
        series.name = self.variable

        request = {
            "dataset": self.dataset,
            "variable": self.variable,
            "lat": str(lat),
            "lon": str(lon),
            "method": self.method,
        }

        return series, "unknown", request

    @classmethod
    def find_stations(
        cls, lat: float, lon: float, radius_km: float, variable: str | None = None
    ) -> list[SeriesMeta]:
        """Return the nearest grid point of each dataset within ``radius_km`` of a location.

        One entry per dataset and variable. Datasets whose nearest grid point falls outside the
        radius are skipped.
        """
        from tidesurgedata.sources.base import haversine_km

        stations: list[SeriesMeta] = []
        for collection in cls._catalog().get_children():
            variables = collection.extra_fields.get("cube:variables", {})
            names = [variable] if variable is not None else list(variables)
            if variable is not None and variable not in variables:
                continue
            probe = cls(collection.id, names[0]) if names else None
            if probe is None:
                continue
            ds = probe._open_dataset(collection)

            # Nearest grid cell to the requested point.
            grid_lat = float(ds.latitude.sel(latitude=lat, method="nearest"))
            grid_lon = float(ds.longitude.sel(longitude=lon, method="nearest"))
            if haversine_km(lat, lon, grid_lat, grid_lon) > radius_km:
                continue

            for name in names:
                source = cls(collection.id, name, lat=grid_lat, lon=grid_lon)
                stations.append(source.metadata())
        return stations

    def init_times(self, start: TimeLike, end: TimeLike) -> pd.DatetimeIndex:
        """Initialisation times of the forecast dataset in the half-open range ``[start, end)``.

        Returns
        -------
        pandas.DatetimeIndex
            UTC, sorted, unique, named ``init_time``. Empty if the dataset has no runs in range.
        """
        times = self._all_init_times(self._open_dataset())
        window = times[(times >= to_utc(start)) & (times < to_utc(end))]
        return window.rename("init_time")

    @staticmethod
    def _all_init_times(ds) -> pd.DatetimeIndex:
        """Every initialisation time in the dataset, as a sorted unique UTC index."""
        times = pd.DatetimeIndex(pd.to_datetime(ds.init_time.values, utc=True))
        return times.sort_values().unique()

    def fetch_forecast(
        self, issued: TimeLike, horizon: pd.Timedelta, members: Sequence[str] | None = None
    ) -> Forecast:
        """Forecast from the latest initialisation available by ``issued``.

        Use the latest initialisation time available by ``issued`` (respecting
        :attr:`latency`). Never use data initialised after that.

        Parameters
        ----------
        issued : time-like
            Timezone-aware issue time; naive input is rejected.
        horizon : pandas.Timedelta
            Valid times run from the initialisation time to ``issued + horizon``.
        members : sequence of str, optional
            Member labels to return (``"control"`` and ``"1"``, ``"2"``, … for ensembles);
            default all members of the dataset.

        Returns
        -------
        Forecast
            ``values`` indexed by valid time with one ``float64`` column per member, the
            ``init_time`` used, and a :class:`~tidesurgedata.meta.FetchRecord`. Validated with
            :func:`tidesurgedata.contract.validate_forecast`.

        Raises
        ------
        ValueError
            If ``issued`` is naive, ``horizon`` is negative, no initialisation is available by
            ``issued - latency``, or an unknown member is requested.
        """
        lat, lon = self._located()
        issued_utc = to_utc(issued)
        horizon = pd.Timedelta(horizon)
        if horizon < pd.Timedelta(0):
            raise ValueError(f"horizon must be non-negative, got {horizon!r}.")

        ds = self._open_dataset()

        # The latest run whose data was available at the issue time.
        cutoff = issued_utc - self.latency
        available = self._all_init_times(ds)
        available = available[available <= cutoff]
        if len(available) == 0:
            raise ValueError(
                f"{self.dataset} has no initialisation available by {cutoff.isoformat()}."
            )
        init_time = available[-1]

        data = (
            ds[self.variable]
            .sel(init_time=init_time.tz_localize(None))
            .sel(latitude=lat, longitude=lon, method="nearest")
        )

        # valid_time = init_time + lead_time, truncated at the requested horizon.
        valid_time = init_time + pd.to_timedelta(data.lead_time.values)
        keep = valid_time <= issued_utc + horizon
        data = data.isel(lead_time=keep)
        valid_time = pd.DatetimeIndex(valid_time[keep], name="valid_time")

        member_dims = [dim for dim in data.dims if dim in MEMBER_DIMENSIONS]
        if member_dims:
            member_dim = member_dims[0]
            data = data.transpose("lead_time", member_dim)
            labels = [_member_label(value) for value in data[member_dim].values]
            frame = pd.DataFrame(data.values, index=valid_time, columns=labels)
        else:
            frame = pd.DataFrame({CONTROL: data.values}, index=valid_time)

        if members is not None:
            requested = [str(member) for member in members]
            unknown = [member for member in requested if member not in frame.columns]
            if unknown or not requested:
                raise ValueError(
                    f"Unknown members {unknown}; {self.dataset} provides {list(frame.columns)}."
                )
            frame = frame[requested]

        frame = frame.astype("float64")  # dynamical.org stores float32

        meta = self.metadata()
        record = FetchRecord(
            meta=meta,
            start=init_time,
            end=issued_utc + horizon,
            retrieved_at=pd.Timestamp.now(tz="UTC"),
            quality="unknown",
            n_values=int(frame.size),
            n_missing=int(frame.isna().to_numpy().sum()),
            request={
                "dataset": self.dataset,
                "variable": self.variable,
                "lat": str(lat),
                "lon": str(lon),
                "init_time": init_time.isoformat(),
                "members": ",".join(frame.columns),
            },
        )
        forecast = Forecast(values=frame, init_time=init_time, record=record)
        validate_forecast(forecast, meta)
        return forecast
