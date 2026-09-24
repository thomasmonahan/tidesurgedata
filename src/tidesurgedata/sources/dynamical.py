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

from tidesurgedata.meta import Quality, SeriesMeta
from tidesurgedata.sources.base import FetchRecord, Forecast, GriddedSource
from tidesurgedata.sources.registry import register_source
from tidesurgedata.timeutil import TimeLike

__all__ = ["Dynamical"]


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

    def metadata(self) -> SeriesMeta:
        """Return metadata for the selected dynamical.org variable and location."""

        # Import pystac here so the optional ``met`` dependency stays lazy.
        import pystac
        # try:
        # except: ImportError("import pystac required")

        if self.lat is None or self.lon is None:
            raise ValueError(
                "lat and lon must be set before requesting metadata"
            )  # check lat-long req

        # Open the dynamical.org STAC catalogue.
        catalog = pystac.Catalog.from_file(
            "https://stac.dynamical.org/catalog.json"
        )  ### opens Dynamical's STAC catalogue -
        # below we will select the dataset passed to the class (Dynamical)

        # Find the collection requested when Dynamical(...) was constructed.
        collection = catalog.get_child(self.dataset)

        if collection is None:
            raise ValueError(
                f"Unknown dynamical.org dataset: {self.dataset}"
            )  # Check for valid dataset

        variables = collection.extra_fields.get("cube:variables", {})
        if self.variable not in variables:
            raise ValueError(f"Variable {self.variable!r} is not available in {self.dataset!r}")

        variable = variables[self.variable]

        return SeriesMeta(
            source="dynamical",
            station_id=f"{self.dataset}:{self.lat},{self.lon}",
            variable=self.variable,
            lat=self.lat,
            lon=self.lon,
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

    # def _fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Series, Quality, dict]:
    #    raise NotImplementedError("BL-15")

    def _fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Series, Quality, dict]:
        """Fetch analysis data from dynamical.org for the user specified location and time range."""

        # Imports for this function
        import icechunk
        import pystac
        import xarray as xr

        if self.lat is None or self.lon is None:
            raise ValueError("lat and lon must be set before fetching data")  # check lat-long req

        # Find the requested dataset in the dynamical.org STAC catalogue.
        catalog = pystac.Catalog.from_file("https://stac.dynamical.org/catalog.json")
        collection = catalog.get_child(
            self.dataset
        )  # same synbtax here to access dynamics's STAC as in metadata()

        if collection is None:
            raise ValueError(f"Unknown dynamical.org dataset: {self.dataset}")

        # STAC tells us where the Icechunk repository is stored.
        asset = collection.assets["icechunk-https"]

        # Open the remote Icechunk repository read-only.
        repo = icechunk.Repository.open(icechunk.http_storage(asset.href))
        session = repo.readonly_session("main")

        # Open the repository as an xarray Dataset.
        ds = xr.open_zarr(session.store, chunks=None)

        # Have to be careful at htis point: xarray uses timezone-naive datetime64 coordinates
        # , while tidesurgedata supplies timezone-aware UTC timestamps.
        # so here I am converting to UTC to conform with the rest of tidesurge.
        #  This neeeds a global check
        start = start.tz_convert("UTC").tz_localize(None)
        end = end.tz_convert("UTC").tz_localize(None)

        # Select time independently from space. xarray cannot apply
        # method="nearest" when one of the indexers is a slice.
        data = ds[self.variable].sel(
            time=slice(start, end),
        )

        # Sample the requested geographic point according to the source's
        # configured spatial method.
        if self.method == "nearest":
            data = data.sel(
                latitude=self.lat,
                longitude=self.lon,
                method="nearest",
            )
        elif self.method == "linear":
            data = data.interp(
                latitude=self.lat,
                longitude=self.lon,
                method="linear",
            )
        else:
            raise ValueError(
                f"Unsupported spatial sampling method: {self.method!r}. "
                "Expected 'nearest' or 'linear'."
            )

        series = data.to_series().astype("float64")
        series.index = pd.to_datetime(series.index, utc=True)

        # xarray label slices include both endpoints, whereas TideSurgeData uses
        # half-open [start, end) intervals. Enforce that contract here.
        start_utc = start.tz_localize("UTC")
        end_utc = end.tz_localize("UTC")

        series = series[
            (series.index >= start_utc)
            & (series.index < end_utc)
        ]

        series.name = self.variable

        request = {
            "dataset": self.dataset,
            "variable": self.variable,
            "lat": self.lat,
            "lon": self.lon,
        }

        return series, "unknown", request

    @classmethod
    def find_stations(
        cls, lat: float, lon: float, radius_km: float, variable: str | None = None
    ) -> list[SeriesMeta]:
        """Return the Dynamical grid points within the specified radius_km
        of the location specified by the lat long."""

        import icechunk
        import pystac
        import xarray as xr

        from tidesurgedata.sources.base import (
            haversine_km,
        )
        # use the great circle distance calculator that we have from sources functions.

        catalog = pystac.Catalog.from_file(
            "https://stac.dynamical.org/catalog.json"
        )  # access STAC catalog again

        stations = []  # initialise empty stations list

        for collection in catalog.get_children():
            variables = collection.extra_fields.get("cube:variables", {})

            asset = collection.assets["icechunk-https"]

            repo = icechunk.Repository.open(
                icechunk.http_storage(asset.href)
            )  # load the icechunk dataset
            session = repo.readonly_session("main")

            ds = xr.open_zarr(session.store, chunks=None)  # open up the grid
            grid_lat = float(
                ds.latitude.sel(latitude=lat, method="nearest")
            )  # Find the nearest grid cell to coordinates, lat
            grid_lon = float(
                ds.longitude.sel(longitude=lon, method="nearest")
            )  # Find the nearest grid cell to coordinates, lon

            # Only return the grid point if it is inside the search radius.
            # Using the nicely pre-defined great-circle calc
            if haversine_km(lat, lon, grid_lat, grid_lon) > radius_km:
                continue

            # Return one SeriesMeta for each requested variable.
            names = [variable] if variable else variables
            for name in names:
                source = cls(
                    collection.id, name, lat=grid_lat, lon=grid_lon
                )  # just packaging for the correct SeriesMeta output
            stations.append(source.metadata())

        return stations

    def fetch_fields(
        self,
        variables: Sequence[str],
        start: TimeLike,
        end: TimeLike,
        *,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
    ):
        """Fetch multiple variables over a regional space-time cube.

        This is an additive spatial API for applications such as maps and
        animations.  It deliberately does not use or modify the one-dimensional
        :meth:`fetch` / :meth:`_fetch` source contract.

        The remote Dynamical dataset is opened once, all requested variables are
        selected together, and only the requested time/latitude/longitude subset
        is loaded into memory.  The returned object is an ``xarray.Dataset``.
        """
        import icechunk
        import pystac
        import xarray as xr

        names = tuple(dict.fromkeys(variables))
        if not names:
            raise ValueError("variables must contain at least one variable name")
        if lat_min > lat_max:
            raise ValueError("lat_min must be <= lat_max")
        if lon_min > lon_max:
            raise ValueError("lon_min must be <= lon_max")

        start_utc = pd.Timestamp(start)
        end_utc = pd.Timestamp(end)
        if start_utc.tzinfo is None or end_utc.tzinfo is None:
            raise ValueError("start and end must be timezone-aware")
        start_utc = start_utc.tz_convert("UTC")
        end_utc = end_utc.tz_convert("UTC")
        if end_utc <= start_utc:
            raise ValueError("end must be after start")

        catalog = pystac.Catalog.from_file("https://stac.dynamical.org/catalog.json")
        collection = catalog.get_child(self.dataset)
        if collection is None:
            raise ValueError(f"Unknown dynamical.org dataset: {self.dataset}")

        available = collection.extra_fields.get("cube:variables", {})
        missing = [name for name in names if name not in available]
        if missing:
            raise ValueError(
                f"Variables {missing!r} are not available in {self.dataset!r}"
            )

        asset = collection.assets["icechunk-https"]
        repo = icechunk.Repository.open(icechunk.http_storage(asset.href))
        session = repo.readonly_session("main")
        ds = xr.open_zarr(session.store, chunks=None)

        for coord in ("time", "latitude", "longitude"):
            if coord not in ds.coords:
                raise ValueError(
                    f"Dataset {self.dataset!r} has no {coord!r} coordinate"
                )

        # xarray's datetime64 coordinates are timezone-naive.  Keep the public
        # API UTC-aware, then remove the timezone only for indexing the dataset.
        start_naive = start_utc.tz_localize(None)
        end_naive = end_utc.tz_localize(None)

        fields = ds[list(names)].sel(time=slice(start_naive, end_naive))
        # ``sel`` slices are inclusive at both ends; TideSurgeData ranges are
        # half-open, so explicitly remove a sample exactly at ``end``.
        fields = fields.where(fields.time < end_naive, drop=True)

        lat0 = float(ds.latitude.values[0])
        lat1 = float(ds.latitude.values[-1])
        lat_slice = slice(lat_min, lat_max) if lat0 <= lat1 else slice(lat_max, lat_min)

        lon0 = float(ds.longitude.values[0])
        lon1 = float(ds.longitude.values[-1])
        uses_360 = min(lon0, lon1) >= 0.0 and max(lon0, lon1) > 180.0
        if uses_360:
            req_lon_min = lon_min % 360.0
            req_lon_max = lon_max % 360.0
            if req_lon_min > req_lon_max:
                raise ValueError(
                    "longitude bounds cross the dataset's 0-degree seam; "
                    "split this request into two regions"
                )
        else:
            req_lon_min = lon_min
            req_lon_max = lon_max

        lon_slice = (
            slice(req_lon_min, req_lon_max)
            if lon0 <= lon1
            else slice(req_lon_max, req_lon_min)
        )

        fields = fields.sel(latitude=lat_slice, longitude=lon_slice)
        if fields.sizes.get("time", 0) == 0:
            raise ValueError("No data available in the requested time range")
        if fields.sizes.get("latitude", 0) == 0 or fields.sizes.get("longitude", 0) == 0:
            raise ValueError("No grid cells fall inside the requested spatial bounds")

        # Materialise only after all remote dimensions have been restricted.
        return fields.load()

    def init_times(self, start: TimeLike, end: TimeLike) -> pd.DatetimeIndex:
        """Initialisation times of the forecast dataset in ``[start, end)``."""
        import icechunk
        import pystac
        import xarray as xr

        catalog = pystac.Catalog.from_file("https://stac.dynamical.org/catalog.json")
        collection = catalog.get_child(self.dataset)

        if collection is None:
            raise ValueError(f"Unknown dynamical.org dataset: {self.dataset}")

        asset = collection.assets["icechunk-https"]
        repo = icechunk.Repository.open(icechunk.http_storage(asset.href))
        session = repo.readonly_session("main")
        ds = xr.open_zarr(session.store, chunks=None)

        times = pd.DatetimeIndex(pd.to_datetime(ds.init_time.values, utc=True))

        start = pd.Timestamp(start)
        end = pd.Timestamp(end)

        return times[(times >= start) & (times < end)]

    def fetch_forecast(
        self, issued: TimeLike, horizon: pd.Timedelta, members: Sequence[str] | None = None
    ) -> Forecast:
        """Fetch the latest forecast available at `issued`

        Use the latest initialisation time available by `issued` (respecting latency).
        Never use data initialised after that.

        Returns a :class:`~tidesurgedata.sources.base.Forecast` with valid times from the
        initialisation time to ``issued + horizon`` and one column per requested member
        (``"control"`` for deterministic datasets); validated with
        :func:`tidesurgedata.contract.validate_forecast`."""

        import icechunk
        import pystac
        import xarray as xr

        issued = pd.Timestamp(issued)
        issued = issued.tz_localize("UTC") if issued.tzinfo is None else issued.tz_convert("UTC")

        # Find the latest forecast that would have been available at `issued`.
        cutoff = issued - self.latency
        init_times = self.init_times(
            pd.Timestamp.min.tz_localize("UTC"), cutoff + pd.Timedelta("1ns")
        )

        if len(init_times) == 0:
            raise ValueError(f"No forecast available by {issued}")

        init_time = init_times[-1]

        # Open the Dynamical dataset.
        catalog = pystac.Catalog.from_file("https://stac.dynamical.org/catalog.json")
        collection = catalog.get_child(self.dataset)

        if collection is None:
            raise ValueError(f"Unknown dynamical.org dataset: {self.dataset}")

        asset = collection.assets["icechunk-https"]
        repo = icechunk.Repository.open(icechunk.http_storage(asset.href))
        session = repo.readonly_session("main")
        ds = xr.open_zarr(session.store, chunks=None)

        # xarray coordinates are timezone-naive.
        init_naive = init_time.tz_localize(None)

        data = ds[self.variable].sel(
            init_time=init_naive,
        )

        if self.method == "nearest":
            data = data.sel(
                latitude=self.lat,
                longitude=self.lon,
                method="nearest",
            )
        elif self.method == "linear":
            data = data.interp(
                latitude=self.lat,
                longitude=self.lon,
                method="linear",
            )
        else:
            raise ValueError(
                f"Unsupported spatial sampling method: {self.method!r}. "
                "Expected 'nearest' or 'linear'."
            )

        # Convert lead times into actual forecast valid times.
        valid_time = init_time + pd.to_timedelta(data.lead_time.values)

        # Keep forecasts only through issued + horizon.
        keep = valid_time <= issued + horizon
        data = data.isel(lead_time=keep)
        valid_time = valid_time[keep]

        # Convert to the DataFrame required by Forecast.
        # Find any dimension other than lead_time.
        member_dims = [dim for dim in data.dims if dim != "lead_time"]

        if member_dims:
            member_dim = member_dims[0]

            # Rows = lead times, columns = ensemble members.
            data = data.transpose("lead_time", member_dim)

            frame = pd.DataFrame(
                data.values,
                index=valid_time,
                columns=[str(m) for m in data[member_dim].values],
            )

            if members is not None:
                frame = frame[list(map(str, members))]

        else:
            frame = pd.DataFrame(
                {"control": data.values},
                index=valid_time,
            )

        frame = frame.astype("float64")  # convert the float32 output from the Dynamical data to f64
        frame.index.name = "valid_time"

        record = FetchRecord(
            meta=self.metadata(),
            start=init_time,
            end=issued + horizon,
            retrieved_at=pd.Timestamp.now(tz="UTC"),
            quality="unknown",
            n_values=int(frame.size),
            n_missing=int(frame.isna().sum().sum()),
            request={
                "dataset": self.dataset,
                "variable": self.variable,
                "lat": str(self.lat),
                "lon": str(self.lon),
            },
        )

        return Forecast(record=record, init_time=init_time, values=frame)
