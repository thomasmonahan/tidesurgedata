"""USGS adapter (BL-06). STUB.

Provider facts
--------------
- USGS Water Data APIs, accessed through ``dataretrieval.waterdata`` (install the ``usgs`` extra:
  ``pip install "tidesurgedata[usgs]"``; ``dataretrieval>=1.1.0``).
- **Do not use the legacy ``nwis`` module / WaterServices**: decommissioning is scheduled for
  Q1 2027.
- API key via the environment variable ``API_USGS_PAT`` (never stored in code, specs, recipes,
  records or cassettes).
- Convert ft³/s -> m³/s (discharge) and ft -> m (stage).

Implementation checklist
------------------------
- [ ] Import ``dataretrieval`` inside functions; raise ``ImportError`` naming the ``usgs`` extra.
- [ ] ``metadata()``: site name, location, parameter mapping (``discharge`` -> ``m3 s-1``,
      ``stage`` -> ``m`` with datum).
- [ ] ``_fetch()``: time-series values for ``[start, end)``; approval status -> quality
      (approved -> ``"verified"``, provisional -> ``"preliminary"``); units converted.
- [ ] Confirm per-request limits (``max_request``) and typical latency.
- [ ] ``find_stations()`` via ``waterdata`` monitoring-location queries.
- [ ] Cassette with API key filtered; contract tests; remove markers; CHANGELOG entry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import pandas as pd

from tidesurgedata.meta import Quality, SeriesMeta
from tidesurgedata.sources.base import BaseSource, haversine_km
from tidesurgedata.sources.registry import register_source
from tidesurgedata.units import convert

__all__ = ["USGS"]


@register_source("usgs")
@dataclass(frozen=True)
class USGS(BaseSource):
    """USGS monitoring-location time series.

    Parameters
    ----------
    site_id : str
        USGS site number, e.g. ``"01376500"``.
    parameter : {"discharge", "stage"}
        Variable to retrieve.
    """

    site_id: str
    parameter: str = "discharge"

    # TODO(BL-06): confirm max_request and latency.

    # USGS continuous requests are limited to 3 years.
    max_request: ClassVar[pd.Timedelta | None] = pd.Timedelta("1095D")

    def metadata(self) -> SeriesMeta:
        try:
            from dataretrieval import waterdata
        except ImportError as exc:
            raise ImportError(
                'USGS support requires the "usgs" extra: pip install "tidesurgedata[usgs]"'
            ) from exc

        if self.parameter == "discharge":
            units = "m3 s-1"
            datum = None
        elif self.parameter == "stage":
            units = "m"
            datum = "local gage datum"
        else:
            raise ValueError("USGS parameter must be 'discharge' or 'stage'.")

        monitoring_location_id = (
            self.site_id if self.site_id.startswith("USGS-") else f"USGS-{self.site_id}"
        )

        locations, _ = waterdata.get_monitoring_locations(
            monitoring_location_id=monitoring_location_id,
        )

        if locations.empty:
            raise ValueError(f"USGS site {self.site_id!r} was not found.")

        row = locations.iloc[0]
        geometry = row["geometry"]

        if hasattr(geometry, "x"):
            lon = float(geometry.x)
            lat = float(geometry.y)
        else:
            lon = float(geometry[0])
            lat = float(geometry[1])

        return SeriesMeta(
            source=self.registry_name,
            station_id=self.site_id,
            variable=self.parameter,
            lat=lat,
            lon=lon,
            units=units,
            datum=datum,
            sampling="instantaneous",
            window=None,
            label=None,
            licence="US public domain",
            attribution="U.S. Geological Survey",
            url=f"https://waterdata.usgs.gov/monitoring-location/{monitoring_location_id}",
            name=str(row["monitoring_location_name"]),
        )

    def _fetch(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> tuple[pd.Series, Quality, dict]:

        if self.parameter == "discharge":
            parameter_code = "00060"
            expected_unit = "ft^3/s"
            input_unit = "ft3 s-1"
            output_unit = "m3 s-1"

        elif self.parameter == "stage":
            parameter_code = "00065"
            expected_unit = "ft"
            input_unit = "ft"
            output_unit = "m"

        else:
            raise ValueError("USGS parameter must be 'discharge' or 'stage'.")

        try:
            from dataretrieval import waterdata
        except ImportError as exc:
            raise ImportError(
                'USGS support requires the "usgs" extra: pip install "tidesurgedata[usgs]"'
            ) from exc

        monitoring_location_id = (
            self.site_id if self.site_id.startswith("USGS-") else f"USGS-{self.site_id}"
        )

        request = {
            "monitoring_location_id": monitoring_location_id,
            "parameter_code": parameter_code,
            "time": f"{start.isoformat()}/{end.isoformat()}",
        }

        raw, _ = waterdata.get_continuous(**request)

        if raw.empty:
            series = pd.Series(
                [],
                index=pd.DatetimeIndex(
                    [],
                    tz="UTC",
                    name="time",
                ),
                dtype="float64",
                name=self.parameter,
            )
            return series, "unknown", request

        raw = raw.copy()

        raw["time"] = pd.to_datetime(
            raw["time"],
            utc=True,
        )

        raw = raw.loc[(raw["time"] >= start) & (raw["time"] < end)]

        units = set(raw["unit_of_measure"].dropna())

        if units != {expected_unit}:
            raise ValueError(f"Unexpected USGS {self.parameter} units: {units}")

        values = pd.to_numeric(
            raw["value"],
            errors="coerce",
        )

        values = convert(
            values,
            input_unit,
            output_unit,
        )

        series = pd.Series(
            values.to_numpy(),
            index=pd.DatetimeIndex(raw["time"]),
            dtype="float64",
            name=self.parameter,
        )

        series.index.name = "time"

        statuses = set(raw["approval_status"].dropna().astype(str).str.lower())

        if statuses == {"approved"}:
            quality: Quality = "verified"
        elif statuses == {"provisional"}:
            quality = "preliminary"
        elif len(statuses) > 1:
            quality = "mixed"
        else:
            quality = "unknown"

        return series, quality, request

    @classmethod
    def find_stations(
        cls,
        lat: float,
        lon: float,
        radius_km: float,
        variable: str | None = None,
    ) -> list[SeriesMeta]:
        try:
            from dataretrieval import waterdata
        except ImportError as exc:
            raise ImportError(
                'USGS support requires the "usgs" extra: pip install "tidesurgedata[usgs]"'
            ) from exc

        parameter_codes = {
            "discharge": "00060",
            "stage": "00065",
        }

        if variable is not None and variable not in parameter_codes:
            raise ValueError("USGS variable must be 'discharge', 'stage', or None.")

        codes = (
            [parameter_codes[variable]] if variable is not None else list(parameter_codes.values())
        )

        lat_delta = radius_km / 111.0
        lon_delta = radius_km / (111.0 * max(math.cos(math.radians(lat)), 0.01))

        inventory, _ = waterdata.get_time_series_metadata(
            bbox=[
                lon - lon_delta,
                lat - lat_delta,
                lon + lon_delta,
                lat + lat_delta,
            ],
            parameter_code=codes,
            statistic_id="00011",
            computation_identifier="Instantaneous",
        )

        if inventory.empty:
            return []

        pairs = inventory[["monitoring_location_id", "parameter_code", "geometry"]].drop_duplicates(
            subset=["monitoring_location_id", "parameter_code"]
        )

        # Calculate the distance of every candidate station from the requested
        # location before making any further provider requests.
        candidates = []

        for _, row in pairs.iterrows():
            geometry = row["geometry"]

            if hasattr(geometry, "x"):
                station_lon = float(geometry.x)
                station_lat = float(geometry.y)
            else:
                station_lon = float(geometry[0])
                station_lat = float(geometry[1])

            distance_km = haversine_km(
                lat,
                lon,
                station_lat,
                station_lon,
            )

            if distance_km <= radius_km:
                candidates.append((distance_km, row))

        # No stations within the requested radius.
        if not candidates:
            return []

        # Sort geographically, without making any additional USGS requests.
        candidates.sort(key=lambda candidate: candidate[0])

        # Only resolve metadata for the closest station.
        _, row = candidates[0]

        station_variable = "discharge" if row["parameter_code"] == "00060" else "stage"

        site_id = str(row["monitoring_location_id"]).removeprefix("USGS-")

        return [
            cls(
                site_id,
                parameter=station_variable,
            ).metadata()
        ]
