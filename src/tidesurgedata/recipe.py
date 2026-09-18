"""Recipes: declarative, serialisable definitions of training and forecast datasets.

A :class:`Recipe` names a target source and a set of :class:`Driver` s with lags. It knows the
frame's column layout and the maximum forecast lead time it supports (ADR 0003), and it builds
training and forecast frames (BL-12, BL-13).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import pandas as pd

from tidesurgedata.align import lag_column_name, materialise_lags, to_grid
from tidesurgedata.contract import validate_frame
from tidesurgedata.meta import FetchRecord
from tidesurgedata.sources.base import BaseSource, ForecastSource, GriddedSource
from tidesurgedata.sources.registry import source_from_spec
from tidesurgedata.timeutil import TimeLike

__all__ = ["SCHEMA_VERSION", "Driver", "Recipe"]

SCHEMA_VERSION = 1
_NAME_RE = re.compile(r"[a-z][a-z0-9_]*")
_HOW = ("instant", "mean")


@dataclass(frozen=True)
class Driver:
    """An input variable of a recipe, with the lags at which it enters the frame.

    Parameters
    ----------
    name : str
        Column prefix, matching ``[a-z][a-z0-9_]*``; unique within a recipe.
    source : BaseSource
        Observed or analysis data for the driver.
    lags_hours : tuple of float
        Non-empty lags in hours; negative means the past. Each must be a multiple of the recipe
        ``freq``.
    forecast : BaseSource, optional
        Forecast source (must also implement ``ForecastSource``) used for valid times after the
        issue time in forecast frames.
    how : {"instant", "mean"}
        How the driver is put on the grid (see :func:`tidesurgedata.align.to_grid`).
    max_gap : str or None
        Largest gap filled by interpolation, e.g. ``"2h"``; ``None`` disables filling.
    """

    name: str
    source: BaseSource
    lags_hours: tuple[float, ...]
    forecast: BaseSource | None = None
    how: Literal["instant", "mean"] = "mean"
    max_gap: str | None = "2h"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _NAME_RE.fullmatch(self.name):
            raise ValueError(f"Driver name {self.name!r} must match [a-z][a-z0-9_]*.")
        if not isinstance(self.source, BaseSource):
            raise TypeError(f"Driver {self.name!r}: source must be a BaseSource.")
        if self.forecast is not None and not (
            isinstance(self.forecast, BaseSource) and isinstance(self.forecast, ForecastSource)
        ):
            raise TypeError(
                f"Driver {self.name!r}: forecast must be a BaseSource implementing ForecastSource."
            )
        lags = tuple(float(lag) for lag in self.lags_hours)
        if not lags:
            raise ValueError(f"Driver {self.name!r}: lags_hours must not be empty.")
        if not all(math.isfinite(lag) for lag in lags):
            raise ValueError(f"Driver {self.name!r}: lags_hours must be finite.")
        if len(set(lags)) != len(lags):
            raise ValueError(f"Driver {self.name!r}: lags_hours must be unique.")
        object.__setattr__(self, "lags_hours", lags)
        if self.how not in _HOW:
            raise ValueError(f"Driver {self.name!r}: how must be one of {_HOW}.")
        if self.max_gap is not None:
            _positive_timedelta(self.max_gap, f"Driver {self.name!r}: max_gap")

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation."""
        return {
            "name": self.name,
            "source": self.source.to_spec(),
            "lags_hours": list(self.lags_hours),
            "forecast": None if self.forecast is None else self.forecast.to_spec(),
            "how": self.how,
            "max_gap": self.max_gap,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Driver:
        """Inverse of :meth:`to_dict`."""
        return cls(
            name=d["name"],
            source=source_from_spec(d["source"]),
            lags_hours=tuple(d["lags_hours"]),
            forecast=None if d.get("forecast") is None else source_from_spec(d["forecast"]),
            how=d.get("how", "mean"),
            max_gap=d.get("max_gap", "2h"),
        )


def _positive_timedelta(value: Any, what: str) -> pd.Timedelta:
    try:
        td = pd.Timedelta(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} {value!r} is not a valid duration.") from exc
    if td is pd.NaT or td <= pd.Timedelta(0):
        raise ValueError(f"{what} must be positive, got {value!r}.")
    return td


@dataclass(frozen=True)
class Recipe:
    """Definition of a dataset: a target series and lagged drivers on a regular grid.

    Parameters
    ----------
    target : BaseSource
        The series to predict.
    drivers : tuple of Driver
        Input variables, in column order.
    freq : str
        Grid spacing, e.g. ``"1h"``.
    target_how : {"instant", "mean"}
        How the target is put on the grid.
    target_column : str, optional
        Name of the target column; ``None`` uses the target's variable name. RTide, for example,
        expects ``"observations"``.
    schema_version : int
        Serialisation schema version.
    """

    target: BaseSource
    drivers: tuple[Driver, ...] = ()
    freq: str = "1h"
    target_how: Literal["instant", "mean"] = "instant"
    target_column: str | None = None
    schema_version: int = SCHEMA_VERSION
    _last_provenance: tuple[FetchRecord, ...] = field(
        default=(), init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.target, BaseSource):
            raise TypeError("Recipe target must be a BaseSource.")
        drivers = tuple(self.drivers)
        object.__setattr__(self, "drivers", drivers)
        if not all(isinstance(d, Driver) for d in drivers):
            raise TypeError("Recipe drivers must be Driver instances.")
        names = [d.name for d in drivers]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"Driver names must be unique; duplicated: {duplicates}.")
        step = _positive_timedelta(self.freq, "Recipe freq")
        for d in drivers:
            for lag in d.lags_hours:
                if pd.Timedelta(hours=lag) % step != pd.Timedelta(0):
                    raise ValueError(
                        f"Driver {d.name!r}: lag {lag} h is not a multiple of freq {self.freq!r}."
                    )
        if self.target_how not in _HOW:
            raise ValueError(f"target_how must be one of {_HOW}.")
        if self.target_column is not None:
            if not isinstance(self.target_column, str) or not self.target_column:
                raise ValueError("target_column must be a non-empty string or None.")
            if self.target_column in self.feature_columns:
                raise ValueError(
                    f"target_column {self.target_column!r} clashes with a feature column."
                )
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported recipe schema_version {self.schema_version}; "
                f"this version of tidesurgedata supports {SCHEMA_VERSION}."
            )

    # --- column layout and lead time ----------------------------------------------------------

    @property
    def target_latlon(self) -> tuple[float, float]:
        """Latitude and longitude of the target series."""
        meta = self.target.metadata()
        return meta.lat, meta.lon

    @property
    def target_column_name(self) -> str:
        """Name of the target column: ``target_column`` or the target's variable name."""
        if self.target_column is not None:
            return self.target_column
        return self.target.metadata().variable

    @property
    def feature_columns(self) -> list[str]:
        """Feature column names, in driver order and then lag order."""
        return [lag_column_name(d.name, lag) for d in self.drivers for lag in d.lags_hours]

    @property
    def max_lead_time(self) -> pd.Timedelta | None:
        """Largest forecast horizon supported without forecast data (ADR 0003).

        For each driver without a forecast source,
        ``lead_d = min(-lag for lag in lags_hours) - latency_d``. The result is the minimum over
        those drivers; ``None`` means unlimited (no drivers, or all drivers have forecast
        sources). A result ``<= 0`` means the recipe supports hindcasts only.
        """
        leads = [
            pd.Timedelta(hours=min(-lag for lag in d.lags_hours)) - d.source.latency
            for d in self.drivers
            if d.forecast is None
        ]
        return min(leads) if leads else None

    def resolved(self) -> Recipe:
        """Copy in which gridded driver sources without a location sample the target location."""
        latlon: tuple[float, float] | None = None

        def locate(source: BaseSource | None) -> BaseSource | None:
            nonlocal latlon
            if isinstance(source, GriddedSource) and (
                getattr(source, "lat", None) is None or getattr(source, "lon", None) is None
            ):
                if latlon is None:
                    latlon = self.target_latlon
                return source.with_location(*latlon)
            return source

        drivers = tuple(
            replace(d, source=locate(d.source), forecast=locate(d.forecast)) for d in self.drivers
        )
        return replace(self, drivers=drivers)

    # --- serialisation ------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation."""
        return {
            "schema_version": self.schema_version,
            "target": self.target.to_spec(),
            "drivers": [d.to_dict() for d in self.drivers],
            "freq": self.freq,
            "target_how": self.target_how,
            "target_column": self.target_column,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Recipe:
        """Inverse of :meth:`to_dict`."""
        return cls(
            target=source_from_spec(d["target"]),
            drivers=tuple(Driver.from_dict(x) for x in d.get("drivers", ())),
            freq=d.get("freq", "1h"),
            target_how=d.get("target_how", "instant"),
            target_column=d.get("target_column"),
            schema_version=d.get("schema_version", SCHEMA_VERSION),
        )

    def to_json(self, **kwargs: Any) -> str:
        """Serialise to JSON; keyword arguments are passed to :func:`json.dumps`."""
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_json(cls, s: str) -> Recipe:
        """Inverse of :meth:`to_json`."""
        return cls.from_dict(json.loads(s))

    # --- frames (to be implemented) -----------------------------------------------------------

    def training_frame(self, start: TimeLike, end: TimeLike) -> pd.DataFrame:
        """Build a training frame over ``[start, end)``.

        Parameters
        ----------
        start, end : time-like
            Timezone-aware bounds of the half-open range.

        Returns
        -------
        pandas.DataFrame
            Passes :func:`tidesurgedata.contract.validate_frame` with
            ``target_column_name`` and ``feature_columns``:

            - index: :func:`tidesurgedata.timeutil.regular_grid` over ``[start, end)`` at
              ``freq``;
            - first column: the target on the grid using ``target_how`` (never interpolated,
              i.e. ``max_gap=None``);
            - then the feature columns in :attr:`feature_columns` order, built with
              :func:`tidesurgedata.align.to_grid` (driver ``how`` and ``max_gap``) and
              :func:`tidesurgedata.align.materialise_lags`;
            - each driver is fetched over the window widened by its lags, so that lagged values
              at the edges of the range are populated where data exist;
            - rows containing NaN are **kept**; consumers decide how to drop them.

        Notes
        -----
        Gridded drivers are resolved first (:meth:`resolved`). The fetch records used are
        available from :meth:`provenance` afterwards.

        Raises
        ------
        ValueError
            If ``start``/``end`` are naive or ``end <= start``.
        """
        from tidesurgedata.timeutil import regular_grid, to_utc

        start_utc = to_utc(start)
        end_utc = to_utc(end)

        if end_utc <= start_utc:
            raise ValueError("end must be after start")

        # Generate index
        index = regular_grid(start_utc, end_utc, self.freq)
        columns = {}
        records: list[FetchRecord] = []

        recipe = self.resolved()

        # Target
        target_series, target_record = recipe.target.fetch_with_record(start_utc, end_utc)
        records.append(target_record)
        target_grid = to_grid(
            target_series,
            recipe.target.metadata(),
            recipe.freq,
            how=recipe.target_how,
            max_gap=None,
        ) # align target to grid
        columns[recipe.target_column_name] = target_grid.reindex(index) # add target to column dict

        # Drivers 
        for driver in recipe.drivers:
            lag_start = start_utc + pd.Timedelta(hours=min(0.0, min(driver.lags_hours)))
            lag_end = end_utc + pd.Timedelta(hours=max(0.0, max(driver.lags_hours)))
            driver_series, driver_record = driver.source.fetch_with_record(lag_start, lag_end)
            records.append(driver_record)

            # align driver to grid
            driver_grid = to_grid(
                driver_series,
                driver.source.metadata(),
                recipe.freq,
                how=driver.how,
                max_gap=driver.max_gap
            )

            # Lags
            lagged = materialise_lags(
                driver_grid,
                driver.name,
                driver.lags_hours,
            )

            for col in lagged:
                columns[col] = lagged[col].reindex(index) # add lag to column dict

        # Assemble target and lags
        df = pd.DataFrame(columns, index=index, dtype="float64")
        df = df[[recipe.target_column_name, *recipe.feature_columns]]
        validate_frame(df, recipe.target_column_name, recipe.feature_columns)
        object.__setattr__(self, "_last_provenance", tuple(records))
        return df

    def forecast_frame(
        self, issued: TimeLike, horizon_hours: float, member: str | None = None
    ) -> pd.DataFrame:
        """Build the frame a model needs to forecast from issue time ``issued``.

        Parameters
        ----------
        issued : time-like
            Timezone-aware issue time. Only information available at this time is used.
        horizon_hours : float
            Forecast horizon in hours.
        member : str, optional
            Forecast ensemble member to use for drivers with forecast sources; default
            ``"control"``.

        Returns
        -------
        pandas.DataFrame
            Passes :func:`tidesurgedata.contract.validate_frame`:

            - index: the regular grid over ``(issued, issued + horizon]``;
            - target column: all NaN;
            - feature columns: observed or analysis data **only** where available by
              ``issued - latency`` of that driver's source (values at later valid times are
              NaN unless provided by a forecast);
            - for drivers with a forecast source, values at valid times after ``issued`` come
              from ``forecast.fetch_forecast(issued, ...)``.

        Notes
        -----
        :func:`~tidesurgedata.align.materialise_lags` keeps its input index, so extend each
        gridded driver series to the end of the forecast grid (with NaN) before lagging;
        otherwise lagged values at future valid times are lost.

        Raises
        ------
        ValueError
            If ``horizon_hours`` exceeds :attr:`max_lead_time`, or ``issued`` is naive.
        """
        raise NotImplementedError("BL-13: Recipe.forecast_frame")

    def forecast_frames(self, issued: TimeLike, horizon_hours: float) -> dict[str, pd.DataFrame]:
        """Build one forecast frame per ensemble member.

        Parameters
        ----------
        issued : time-like
            Timezone-aware issue time.
        horizon_hours : float
            Forecast horizon in hours.

        Returns
        -------
        dict of str to pandas.DataFrame
            Keyed by member label; each frame as returned by
            ``forecast_frame(issued, horizon_hours, member=label)``. Drivers without ensemble
            forecasts contribute identical columns to every member. Members are those common to
            all forecast sources in the recipe.
        """
        raise NotImplementedError("BL-22: Recipe.forecast_frames for ensembles")

    def provenance(self) -> list[FetchRecord]:
        """Fetch records used to build the most recent frame.

        Returns
        -------
        list of FetchRecord
            Target first, then drivers in order (forecast records after their driver's
            observation record). Empty if no frame has been built.
        """
        return list(self._last_provenance)
