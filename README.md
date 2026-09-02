# Coffin Bay Groundwater Intelligence

## Synthetic temporal training dataset

The app now starts in **Use demo data** mode so the spatial workspace works immediately without an upload.

The built-in demo contains:

- **1,200 fixed well points** inside the Coffin Bay/Rizin AOI.
- **5 complete years** of monthly groundwater observations (2021–2025).
- **60 observations per well**, giving **72,000 synthetic records**.
- Seasonal recharge/ET behaviour, recurring inter-annual variability, well-specific effects, spatial controls, and a modest long-term scenario trend.
- Synthetic data are explicitly labelled **scenario data**, not observed measurements.

The 1/3/10-year synthetic toggle has been removed. Five years is now the fixed demo modelling horizon so temporal models have a meaningful recurring series to learn from.

## Real-data training

Switch the sidebar to **Upload CSV** to use real/read groundwater observations. The upload remains authoritative and is never replaced by synthetic data.

For temporal model training, a CSV should ideally include:

- `well_id`
- `date` (preferred) or `year` + `month`
- `longitude`
- `latitude`
- `groundwater_level_mAHD` (or an accepted alias such as `water_level` / `gw_level`)

The app derives month, season, time index and cyclical month features where possible. Uploaded repeated well observations are evaluated with a **temporal holdout**: the latest 20% of each well's observations are held out for testing rather than randomly mixing future observations into training.

## Model workflow

1. Start with the synthetic five-year dataset to inspect the map and time-series workflow.
2. Open **Model Lab** and explicitly click **Train selected models on current data**.
3. Compare RMSE, MAE and R².
4. Load a model as the active model to populate prediction layers and diagnostics.
5. Switch to **Upload CSV** when real observations are available and train on those observations.

Training is intentionally explicit so the app does not spend a long time fitting models every time the dashboard opens.

## Piezometric surface fix

The Piezometric map preview now uses a valid Plotly `Choroplethmap` marker configuration. The previous `marker_line_width` argument was invalid in current Plotly versions and caused the `ValueError` shown in the dashboard screenshot.

## AOI

`AOI.geojson` is bundled as a complete, readable representation of the supplied AOI geometry. The uploaded `AOI.shp` had no companion `.dbf/.shx/.prj` files, so the geometry was converted using the Coffin Bay MGA zone 53 coordinate domain (EPSG:28353) and exported as GeoJSON.

## Repository hygiene

`__pycache__/`, Python bytecode and local Streamlit/upload working directories are excluded through `.gitignore` and should not be committed to GitHub.
