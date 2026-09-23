"""Standalone Dynamical point-data demo.

Exercises only the existing Dynamical/BaseSource point-query API. It does not
use NOAA CO-OPS, USGS, Recipe, or BL-05.

Run from the repository root after installing the meteorology extra:

    pip install -e ".[met]"
    python examples/dynamical_demo.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from tidesurgedata.sources.dynamical import Dynamical

# Example point near New York Harbor. Change these to any location of interest.
LAT = 40.71
LON = -74.01
DATASET = "noaa-gfs-analysis"
PRESSURE_VARIABLE = "pressure_surface"
LOOKBACK_HOURS = 24


def main() -> None:
    # Stay comfortably behind real time so an analysis is likely to exist.
    end = pd.Timestamp.now(tz="UTC").floor("1h") - pd.Timedelta("1D")
    start = end - pd.Timedelta(hours=LOOKBACK_HOURS)

    source = Dynamical(
        dataset=DATASET,
        variable=PRESSURE_VARIABLE,
        lat=LAT,
        lon=LON,
    )

    print(f"Fetching {PRESSURE_VARIABLE} from {DATASET}")
    print(f"Location: ({LAT:.3f}, {LON:.3f})")
    print(f"Range: [{start}, {end})")

    pressure = source.fetch(start, end)
    if pressure.empty:
        raise RuntimeError("Dynamical returned no pressure observations for the requested range.")

    print("\nFetched series:")
    print(pressure)
    print("\nMetadata:")
    print(source.metadata())

    # Plot 1: conventional point time series.
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(pressure.index, pressure.values)
    ax.set_title(f"Dynamical surface pressure at {LAT:.2f}, {LON:.2f}")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel(PRESSURE_VARIABLE)
    fig.autofmt_xdate()
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
