# TideSurgeData Explorer

A repository-local interactive application built on the public `tidesurgedata` API.
It does not add UI dependencies to the core package.

## Install

From the repository root:

```bash
pip install -e ".[all]"
pip install -r apps/explorer/requirements.txt
```

## Run

```bash
streamlit run apps/explorer/app.py
```

## Workflow

1. Click a location on the map.
2. Click **Discover nearby data sources**.
3. The app uses `NOAACoops.find_stations()` and `USGS.find_stations()` and selects the nearest matching station.
4. Dynamical historical and forecast pressure sources use the clicked latitude/longitude directly.
5. Choose dates and click **Load and plot observations** to build `Recipe.training_frame()`.
6. Optionally load a forecast frame, bounded by `recipe.max_lead_time`.
7. Inspect provenance, licence and attribution alongside the plotted data.

The UI intentionally delegates station discovery, source retrieval, frame alignment, lead-time rules and provenance to `tidesurgedata`; it does not reimplement those contracts.
