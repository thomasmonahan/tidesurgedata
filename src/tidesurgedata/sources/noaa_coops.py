"""NOAA CO-OPS adapter (BL-05). STUB.

Provider facts
--------------
- NOAA Center for Operational Oceanographic Products and Services (CO-OPS) Data API
  (``https://api.tidesandcurrents.noaa.gov/api/prod/datagetter``).
- Per-request limits depend on the interval: 1-minute data 4 days, 6-minute data 1 month,
  hourly data 1 year. Chunk accordingly.
- A datum is mandatory for water-level products.
- Water level is preliminary until verified (monthly).
- Request times in GMT (``time_zone=gmt``).
- Many stations also serve wind, air pressure and air temperature. Products in scope: water
  level, predictions (reference only), wind, air pressure.
- US Government data; attribution to NOAA CO-OPS.

Implementation checklist
------------------------
- [ ] ``metadata()``: station name and location from the CO-OPS Metadata API; canonical units
      (request ``units=metric``; wind -> ``u``/``v`` components in ``m s-1``; pressure ``hPa`` ->
      ``Pa``); datum set for water level; sampling convention per product.
- [ ] ``max_request`` by ``interval`` (override as a property if it depends on the instance).
- [ ] ``_fetch()``: one request per chunk; parse flags; quality ``"verified"`` or
      ``"preliminary"`` from the product/data flags; missing values as NaN.
- [ ] ``find_stations()`` using the Metadata API, filtered by distance and product.
- [ ] Contract tests in ``tests/sources/test_noaa_coops.py`` with a recorded cassette; remove the
      ``xfail``/``skip`` markers.
- [ ] Live smoke test passes; licence and attribution recorded; CHANGELOG entry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests

from tidesurgedata.meta import Quality, SeriesMeta
from tidesurgedata.sources.base import BaseSource, haversine_km
from tidesurgedata.sources.registry import register_source
from tidesurgedata.units import convert

__all__ = ["NOAACoops"]


@register_source("noaa_coops")
@dataclass(frozen=True)
class NOAACoops(BaseSource):
    """NOAA CO-OPS station data.

    Parameters
    ----------
    station_id : str
        CO-OPS station identifier, e.g. ``"8518750"`` (The Battery, NY).
    product : str
        CO-OPS product, e.g. ``"water_level"``, ``"predictions"``, ``"wind"``, ``"air_pressure"``.
    datum : str
        Vertical datum for water-level products, e.g. ``"MSL"``, ``"MLLW"``, ``"NAVD"``.
    interval : str, optional
        Sampling interval (e.g. ``"6"``, ``"h"``); ``None`` uses the product default.
    """

    station_id: str
    product: str = "water_level"
    datum: str = "MSL"
    interval: str | None = None

    def __post_init__(self) -> None:
        valid_products = {
            "water_level",
            "predictions",
            "wind",
            "wind_u_10m",
            "wind_v_10m",
            "air_pressure",
        }

        if self.product not in valid_products:
            raise ValueError(f"Unsupported NOAA product: {self.product!r}")

        if self.interval not in {None, "1", "6", "h"}:
            raise ValueError(f"Unsupported NOAA CO-OPS interval: {self.interval!r}")

    def _units(self) -> str:
        """Determine canonical units for the selected NOAA product"""
        units = {
            "water_level": "m",
            "predictions": "m",
            "wind_u_10m": "m s-1",
            "wind_v_10m": "m s-1",
            "air_pressure": "Pa",
        }

        try:
            return units[self.product]
        except KeyError:
            raise ValueError(f"Unsupported NOAA CO-OPS product: {self.product!r}") from None

    def _sampling(self) -> tuple[str, pd.Timedelta | None, str | None]:
        if self.product == "water_level":
            return "window_mean", pd.Timedelta("3min"), "centre"

        if self.product in {"air_pressure", "wind_u_10m", "wind_v_10m"}:
            return "instantaneous", None, None

        raise ValueError(f"Sampling metadata not defined for NOAA CO-OPS product: {self.product!r}")

    def _noaa_product(self) -> str:
        """Returns the appropriate NOAA API product name, according to interval and source"""
        if self.product in {"wind_u_10m", "wind_v_10m"}:
            return "wind"

        if self.product != "water_level":
            return self.product

        product = {
            "1": "one_minute_water_level",
            "6": "water_level",
            "h": "hourly_height",
            None: "water_level",
        }

        try:
            return product[self.interval]
        except KeyError:
            raise ValueError(f"Unsupported water-level interval: {self.interval!r}") from None

    def _supports_one_minute(self) -> bool:
        """Return whether NOAA lists this station as having 1-minute water-level data."""
        url = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json?type=1minute"

        response = requests.get(url)
        response.raise_for_status()
        stations = response.json()["stations"]

        return any(station["id"] == self.station_id for station in stations)

    @property
    def max_request(self) -> pd.Timedelta:
        """Return NOAA's maximum request duration for 6-minute water-level data."""
        limits = {
            "1": pd.Timedelta("4D"),
            "6": pd.Timedelta("30D"),
            "h": pd.Timedelta("365D"),
            None: pd.Timedelta("30D"),
        }

        return limits[self.interval]

    # NOAA Metadata API --> What/Where is this station?
    # API: https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/<station>.json
    # Purpose: Tell me about <station>
    # Returns: name, latitude, logitude, available metadata, etc
    def metadata(self) -> SeriesMeta:
        # Somehow obtain station info
        metadata_url = (
            "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/"
            f"stations/{self.station_id}.json"
        )

        response = requests.get(metadata_url)
        response.raise_for_status()

        raw = response.json()
        station = raw["stations"][0]

        sampling, window, label = self._sampling()

        return SeriesMeta(
            source=self.registry_name,
            station_id=self.station_id,
            variable=self.product,
            lat=station["lat"],
            lon=station["lng"],
            units=self._units(),
            datum=self.datum if self.product in {"water_level", "predictions"} else None,
            sampling=sampling,
            window=window,
            label=label,
            licence="US Government public domain",  # TODO(BL-05): confirm canonical wording
            attribution="NOAA CO-OPS",
            url=station["self"],
            name=station["name"],
        )

    # Retrieve the data from the NOAA API
    # API: https://api.tidesandcurrents.noaa.gov/api/prod/datagetter
    # Purpose: Give me OBSERVATIONS from <station>
    # Inputs: station name, water level, datum etc (params taken from self)
    def _fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Series, Quality, dict]:

        NOAAProduct = self._noaa_product()

        params = {
            "product": NOAAProduct,
            "application": "tidesurgedata",
            "begin_date": start.strftime("%Y%m%d %H:%M"),
            "end_date": end.strftime("%Y%m%d %H:%M"),
            "station": self.station_id,
            "time_zone": "gmt",
            "units": "metric",
            "format": "json",
        }

        if self.product in {"water_level", "predictions"}:
            params["datum"] = self.datum

        data_url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

        response = requests.get(data_url, params=params)
        response.raise_for_status()

        raw = response.json()

        if "error" in raw:
            message = raw["error"].get("message", "Unknown NOAA CO-OPS error")
            raise ValueError(f"NOAA CO-OPS: {message}")

        df = pd.DataFrame(raw["data"])

        times = pd.to_datetime(df["t"], utc=True)

        if self.product in {"wind_u_10m", "wind_v_10m"}:
            speed = pd.to_numeric(df["s"], errors="coerce")
            direction = np.deg2rad(pd.to_numeric(df["d"], errors="coerce"))

            if self.product == "wind_u_10m":
                values = -speed * np.sin(direction)
            else:
                values = -speed * np.cos(direction)

        else:
            values = pd.to_numeric(df["v"], errors="coerce")

            if self.product == "air_pressure":
                values = convert(values, "hPa", "Pa")

        series = pd.Series(
            values.to_numpy(),
            index=pd.DatetimeIndex(times),
            dtype="float64",
        )

        if NOAAProduct == "one_minute_water_level":
            quality: Quality = "preliminary"
        elif NOAAProduct == "hourly_height":
            quality = "verified"
        elif self.product in {"air_pressure", "wind_u_10m", "wind_v_10m"}:
            quality = "unknown"
        else:
            qualities = set(df["q"])

            if qualities == {"v"}:
                quality = "verified"
            elif qualities == {"p"}:
                quality = "preliminary"
            else:
                quality = "mixed"

        return series, quality, params

    # Discover stations
    @classmethod
    def find_stations(
        cls, lat: float, lon: float, radius_km: float, variable: str | None = None
    ) -> list[SeriesMeta]:

        if variable is None:
            variable = "water_level"
        if variable == "water_level":
            station_type = "waterlevels"
        elif variable in {"air_pressure", "wind_u_10m", "wind_v_10m"}:
            station_type = "met"
        else:
            raise ValueError(
                f"Station discovery not yet implemented for NOAA variable: {variable!r}"
            )

        url = f"https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json?type={station_type}"
        response = requests.get(url)
        response.raise_for_status()
        stations = response.json()["stations"]

        nearby = [
            station
            for station in stations
            if haversine_km(lat, lon, station["lat"], station["lng"]) <= radius_km
        ]
        if variable == "water_level":
            units = "m"
            datum = "MSL"
            sampling = "window_mean"
            window = pd.Timedelta("3min")
            label = "centre"

        if variable == "air_pressure":
            pressure_stations = []

            for station in nearby:
                sensors_url = station["sensors"]["self"]
                sensors = requests.get(sensors_url).json()["sensors"]

                has_pressure = any(
                    sensor["name"] == "Barometric Pressure" and sensor["status"] == 1
                    for sensor in sensors
                )
                if has_pressure:
                    pressure_stations.append(station)

            nearby = pressure_stations
            units = "Pa"
            datum = None
            sampling = "instantaneous"
            window = None
            label = None

        if variable in {"wind_u_10m", "wind_v_10m"}:
            wind_stations = []

            for station in nearby:
                sensors_url = station["sensors"]["self"]
                sensors = requests.get(sensors_url).json()["sensors"]

                has_wind = any(
                    sensor["name"] == "Wind" and sensor["status"] == 1 for sensor in sensors
                )
                if has_wind:
                    wind_stations.append(station)

            nearby = wind_stations
            units = "m s-1"
            datum = None
            sampling = "instantaneous"
            window = None
            label = None

        return [
            SeriesMeta(
                source=cls.registry_name,
                station_id=station["id"],
                variable=variable,
                lat=station["lat"],
                lon=station["lng"],
                units=units,
                datum=datum,
                sampling=sampling,
                window=window,
                label=label,
                licence="US Government public domain",  # TODO(BL-05): confirm canonical wording
                attribution="NOAA CO-OPS",
                url=station["self"],
                name=station["name"],
            )
            for station in nearby
        ]
