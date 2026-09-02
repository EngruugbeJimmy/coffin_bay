# Coffin Bay Groundwater Intelligence — Multi-Model Interactive Edition

A Streamlit research prototype designed as a light coastal / physical-geography map interface. It supports spatial selection and CSV extraction, plus model training and comparison across **Random Forest, Generalized Additive Model (GAM), XGBoost and Long Short-Term Memory (LSTM)**.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Main capabilities
- Green-blue coastal UI instead of a black AI-style dashboard.
- **Piezometric map** as the primary interactive workspace.
- Lasso, box and point selection of map observations.
- Export selected wells / observations to CSV.
- Upload a groundwater observation CSV and normalise common column names.
- Train and compare **Random Forest, GAM, XGBoost and LSTM** from the sidebar.
- Select an **Active prediction layer** and use it throughout the map, diagnostics, drivers and well explorer.
- **Model Lab** ranks successful models by holdout RMSE, then reports the leading predictive feature for the champion model.
- Feature diagnostics use native tree importance, permutation importance for GAM and a sequence-permutation diagnostic for LSTM.
- Synthetic demonstration data contain repeated annual observations for each synthetic well so the LSTM has a temporal sequence to learn.

## Model-selection logic
All selected models are trained against the same groundwater target. The leaderboard prioritises **lowest holdout RMSE**, with MAE and R² shown alongside it. The top feature is the highest model-specific importance score.

This is an exploratory benchmark on synthetic data. The champion model and feature should not be presented as a validated real-world Coffin Bay result.

## Notes for uploaded observational data
For GAM / RF / XGBoost, a single observation per well can still be used. LSTM requires repeated observations with `well_id` and `year` (or an equivalent temporal field). If the uploaded dataset has too few repeated wells, the LSTM entry reports a clear error rather than silently producing an invalid result.

Recommended columns:
- `well_id` (optional for non-LSTM models; required for temporal LSTM sequences)
- `longitude` / `lon`
- `latitude` / `lat`
- `groundwater_level_mAHD` / `water_level`
- `year` / `date_year`
- `dem_m` / `dem` / `elevation`
- `distance_coast_m` / `coast_distance_m`
- `geology` / `formation`
- optional rainfall, ET, NDVI, pressure and surface-water-distance predictors

## Scientific caution
The included data are synthetic and intended for interface and workflow prototyping only. Replace them with validated Coffin Bay observations and an appropriately documented modelling / interpolation workflow before scientific inference.
