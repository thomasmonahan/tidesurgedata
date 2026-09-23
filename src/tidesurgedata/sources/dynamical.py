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

        # Select only the requested variable, time range, and nearest grid point.
        #data = ds[self.variable].sel(time=slice(start, end), latitude=self.lat, longitude=self.lon, method=self.method,)

        # First select the requested time range.
        data = ds[self.variable].sel(
            time=slice(start, end),
        )

        # Then sample the spatial grid independently.
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

        # Convert the one-dimensional xarray result to the pandas Series expected
        series = data.to_series().astype("float64")
        series.index = pd.to_datetime(series.index, utc=True)
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
                f"Unsupported spatial sampling method: {self.method!r}"
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

# ---------------------------------------------------------------------------
# Spatial field helpers
# ---------------------------------------------------------------------------
# These functions are attached to Dynamical below rather than changing the
# existing BaseSource point-fetch contract. They intentionally return xarray
# DataArray objects, not pandas Series / Forecast objects.

def _normalise_field_bounds(
    lat_min: float, lat_max: float, lon_min: float, lon_max: float
) -> tuple[float, float, float, float]:
    """Validate and normalise a geographic bounding box."""
    values = (lat_min, lat_max, lon_min, lon_max)
    if not all(pd.notna(v) for v in values):
        raise ValueError("Field bounds must be finite latitude/longitude values.")
    lat_min, lat_max, lon_min, lon_max = map(float, values)
    if not (-90.0 <= lat_min < lat_max <= 90.0):
        raise ValueError("Require -90 <= lat_min < lat_max <= 90.")
    if not (-180.0 <= lon_min < lon_max <= 180.0):
        raise ValueError("Require -180 <= lon_min < lon_max <= 180.")
    return lat_min, lat_max, lon_min, lon_max


def _coordinate_slice(coord, low: float, high: float):
    """Return a slice that follows an xarray coordinate's stored direction."""
    first = float(coord.values[0])
    last = float(coord.values[-1])
    return slice(low, high) if first <= last else slice(high, low)


def _open_spatial_dataset(source: Dynamical):
    """Open a Dynamical collection as xarray without changing point-fetch code."""
    import icechunk
    import pystac
    import xarray as xr

    catalog = pystac.Catalog.from_file("https://stac.dynamical.org/catalog.json")
    collection = catalog.get_child(source.dataset)
    if collection is None:
        raise ValueError(f"Unknown dynamical.org dataset: {source.dataset}")

    asset = collection.assets["icechunk-https"]
    repo = icechunk.Repository.open(icechunk.http_storage(asset.href))
    session = repo.readonly_session("main")
    return xr.open_zarr(session.store, chunks=None), collection


def _fetch_field(
    self: Dynamical,
    time: TimeLike,
    *,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    variable: str | None = None,
):
    """Fetch one two-dimensional analysis field over a geographic box.

    This is additive to :meth:`fetch`: it does not participate in the
    BaseSource/Recipe one-dimensional Series contract. The nearest available
    analysis time is selected and recorded in ``selected_time`` in attrs.
    """
    lat_min, lat_max, lon_min, lon_max = _normalise_field_bounds(
        lat_min, lat_max, lon_min, lon_max
    )
    variable = self.variable if variable is None else variable
    ds, collection = _open_spatial_dataset(self)

    if variable not in ds:
        raise ValueError(f"Variable {variable!r} is not available in {self.dataset!r}")
    if "time" not in ds.coords:
        raise ValueError(f"Dataset {self.dataset!r} does not expose an analysis 'time' coordinate.")
    if "latitude" not in ds.coords or "longitude" not in ds.coords:
        raise ValueError(f"Dataset {self.dataset!r} does not expose latitude/longitude coordinates.")

    requested = pd.Timestamp(time)
    requested = requested.tz_localize("UTC") if requested.tzinfo is None else requested.tz_convert("UTC")
    requested_naive = requested.tz_localize(None)

    field = ds[variable].sel(time=requested_naive, method="nearest")
    field = field.sel(
        latitude=_coordinate_slice(ds.latitude, lat_min, lat_max),
        longitude=_coordinate_slice(ds.longitude, lon_min, lon_max),
    ).squeeze(drop=True)

    extra_dims = [d for d in field.dims if d not in ("latitude", "longitude")]
    if extra_dims:
        raise ValueError(
            f"Variable {variable!r} is not a 2D analysis field after time selection; "
            f"remaining dimensions: {extra_dims}."
        )
    if field.sizes.get("latitude", 0) == 0 or field.sizes.get("longitude", 0) == 0:
        raise ValueError("Requested bounds contain no Dynamical grid cells.")

    selected = pd.Timestamp(field["time"].values, tz="UTC") if "time" in field.coords else requested
    field = field.astype("float64").load()
    field.attrs = dict(field.attrs)
    field.attrs.update(
        {
            "dataset": self.dataset,
            "variable": variable,
            "requested_time": requested.isoformat(),
            "selected_time": selected.isoformat(),
            "licence": collection.extra_fields.get("license", "CC-BY-4.0"),
            "attribution": collection.extra_fields.get("attribution", ""),
        }
    )
    return field


def _fetch_wind_field(
    self: Dynamical,
    time: TimeLike,
    *,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    u_variable: str = "wind_u_10m",
    v_variable: str = "wind_v_10m",
):
    """Fetch matching 2D U/V wind-component analysis fields."""
    u = self.fetch_field(
        time,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        variable=u_variable,
    )
    v = self.fetch_field(
        time,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        variable=v_variable,
    )
    if not u.latitude.equals(v.latitude) or not u.longitude.equals(v.longitude):
        raise ValueError("Dynamical U/V wind fields do not share the same spatial grid.")
    return u, v


# Additive public methods. Existing Dynamical methods above are intentionally
# untouched so BaseSource.fetch(), Recipe and forecast behavior are unchanged.
Dynamical.fetch_field = _fetch_field
Dynamical.fetch_wind_field = _fetch_wind_field
