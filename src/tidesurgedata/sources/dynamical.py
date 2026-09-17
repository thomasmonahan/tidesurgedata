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
from tidesurgedata.sources.base import Forecast, GriddedSource
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
        #try: 
        #except: ImportError("import pystac required")

        if self.lat is None or self.lon is None: raise ValueError("lat and lon must be set before requesting metadata") #check lat-long req

        # Open the dynamical.org STAC catalogue. 
        catalog = pystac.Catalog.from_file("https://stac.dynamical.org/catalog.json") ### opens Dynamical's STAC catalogue and selects the dataset passed to the class (Dynamical)

        # Find the collection requested when Dynamical(...) was constructed. 
        collection = catalog.get_child(self.dataset)

        if collection is None: raise ValueError(f"Unknown dynamical.org dataset: {self.dataset}") #Check for valid dataset

        variables = collection.extra_fields.get("cube:variables", {})
        if self.variable not in variables: raise ValueError( f"Variable {self.variable!r} is not available in {self.dataset!r}" ) 

        variable = variables[self.variable]

        return SeriesMeta( source="dynamical", station_id=f"{self.dataset}:{self.lat},{self.lon}", variable=self.variable, lat=self.lat, lon=self.lon, units=variable["unit"], datum=None, sampling="instantaneous", window=None, label=None, licence=collection.extra_fields.get("license", "CC-BY-4.0"), attribution=collection.extra_fields.get("attribution", ""), url=collection.get_self_href() or "https://stac.dynamical.org/catalog.json", name=collection.title or self.dataset, extra={"dataset": self.dataset})

    #def _fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Series, Quality, dict]:
    #    raise NotImplementedError("BL-15")

    def _fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Series, Quality, dict]:
        """Fetch analysis data from dynamical.org for the user specified location and time range."""

        # Imports for this function
        import icechunk
        import pystac
        import xarray as xr

        if self.lat is None or self.lon is None: raise ValueError("lat and lon must be set before fetching data") #check lat-long req

        # Find the requested dataset in the dynamical.org STAC catalogue.
        catalog = pystac.Catalog.from_file("https://stac.dynamical.org/catalog.json")
        collection = catalog.get_child(self.dataset) #same synbtax here to access dynamics's STAC as in metadata()

        if collection is None: raise ValueError(f"Unknown dynamical.org dataset: {self.dataset}")

        # STAC tells us where the Icechunk repository is stored.
        asset = collection.assets["icechunk-https"]

        # Open the remote Icechunk repository read-only.
        repo = icechunk.Repository.open(icechunk.http_storage(asset.href))
        session = repo.readonly_session("main")

        # Open the repository as an xarray Dataset.
        ds = xr.open_zarr(session.store, chunks=None)

        # Have to be careful at htis point: xarray uses timezone-naive datetime64 coordinates, while tidesurgedata
        # supplies timezone-aware UTC timestamps.
        # so here I am converting to UTC to conform with the rest of tidesurge. This neeeds a global check
        start = start.tz_convert("UTC").tz_localize(None)
        end = end.tz_convert("UTC").tz_localize(None)

        # Select only the requested variable, time range, and nearest grid point.
        data = ds[self.variable].sel(time=slice(start, end), latitude=self.lat, longitude=self.lon)

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
    def find_stations(cls, lat: float, lon: float, radius_km: float, variable: str | None = None) -> list[SeriesMeta]:
        """Return the Dynamical grid points within the specified radius_km of the location specified by the lat long."""

        import icechunk
        import pystac
        import xarray as xr
        from tidesurgedata.sources.base import haversine_km #use the great circle distance calculator that we have from sources functions.

        catalog = pystac.Catalog.from_file("https://stac.dynamical.org/catalog.json") #access STAC catalog again

        stations = [] #initialise empty stations list

        for collection in catalog.get_children():
            variables = collection.extra_fields.get("cube:variables", {}) 

            asset = collection.assets["icechunk-https"]

            repo = icechunk.Repository.open(icechunk.http_storage(asset.href)) #load the icechunk dataset
            session = repo.readonly_session("main") 
            
            ds = xr.open_zarr(session.store, chunks=None) #open up the grid
            grid_lat = float(ds.latitude.sel(latitude=lat, method="nearest")) #Find the nearest grid cell to coordinates, lat
            grid_lon = float(ds.longitude.sel(longitude=lon, method="nearest")) #Find the nearest grid cell to coordinates, lon

            # Only return the grid point if it is inside the search radius. Using the nicely pre-defined great-circle calc
            if haversine_km(lat, lon, grid_lat, grid_lon) > radius_km: continue

            # Return one SeriesMeta for each requested variable. 
            names = [variable] if variable else variables 
            for name in names: source = cls(collection.id, name, lat=grid_lat, lon=grid_lon) #just packaging for the correct SeriesMeta output
            stations.append(source.metadata())

        return stations


    

    def init_times(self, start: TimeLike, end: TimeLike) -> pd.DatetimeIndex:
        """Initialisation times of the forecast dataset in ``[start, end)``."""
        raise NotImplementedError("BL-16")

    def fetch_forecast(
        self, issued: TimeLike, horizon: pd.Timedelta, members: Sequence[str] | None = None
    ) -> Forecast:
        """Use the latest initialisation time available by `issued` (respecting latency).
        Never use data initialised after that.

        Returns a :class:`~tidesurgedata.sources.base.Forecast` with valid times from the
        initialisation time to ``issued + horizon`` and one column per requested member
        (``"control"`` for deterministic datasets); validated with
        :func:`tidesurgedata.contract.validate_forecast`.
        """
        raise NotImplementedError("BL-16")
