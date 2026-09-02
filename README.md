# Coffin Bay Groundwater Intelligence — Rizin / DEA Coastline Edition

This build uses the supplied **Rizin/AOI boundary** as the study-area frame and integrates the official **Digital Earth Australia (DEA) Coastlines annual shoreline** service into the hydrogeographic workflow.

## Coastline workflow
- The app requests the DEA `shorelines_annual` WFS using the Rizin AOI bounding box.
- The returned annual shoreline is clipped to Rizin before it is shown on the map or used for measurements.
- Default analysis year is **2024**. **2025** is available as an option but is explicitly treated as interim DEA Coastlines data.
- Well-to-coast distance is calculated as the shortest/perpendicular distance from every well point to the clipped shoreline in **EPSG:28353 (GDA94 / MGA zone 53)**, giving metres.
- The exported well data includes `coastline_year` and `distance_coast_method` so the distance provenance travels with the CSV.

DEA Coastlines annual shorelines represent the median / most representative shoreline position at approximately mean sea level for each year. The current DEA release covers 1988–2025. Geoscience Australia documents the WFS layer as `dea:shorelines_annual` and provides Python examples using the same WFS endpoint.

## Hydrogeographic controls
- **Rizin** is the study-area boundary and is bundled as `rizin.geojson` plus a complete `rizin.shp` set.
- **Coastal analysis datum = 0.0 m AHD** is shown as the conceptual hydraulic datum anchor.
- **Lake Wangary = 3.0 m AHD** is shown as a surface-water anchor.
- The synthetic well generator places wells inside Rizin in an inland-to-coast corridor. When the DEA shoreline is reachable, the proxy distance is replaced by the measured DEA shoreline distance and the synthetic groundwater target is regenerated from that actual distance.

## Model workflow
- Random Forest
- Generalized Additive Model (GAM)
- XGBoost
- Long Short-Term Memory (LSTM)

Model Lab lets the user train/compare candidates and explicitly **Load as active model**. Downstream prediction layers follow the loaded model.

## Data and map interaction
- Upload a real well CSV as the primary data source.
- Optional DEMO mode is available for interface testing.
- Piezometric map supports Lasso, Box and Point selection.
- Selected wells can be exported directly to CSV.
- The map displays the Rizin boundary, DEA coastline, Lake Wangary and the 0 m AHD coastal datum in a blue-green hydrogeographic interface.

## Coordinate note
The bundled Rizin geometry is represented in **EPSG:28353** before conversion to WGS84 for web mapping. DEA Coastlines arrives through the official WFS workflow and is transformed into EPSG:28353 for metric clipping and distance calculations.

## Scientific note
The coastal 0 m AHD and Lake Wangary 3 m AHD values are implemented as explicit conceptual controls for this research interface. They should be validated against authoritative surveyed / hydrometric observations before being used as calibrated model boundary conditions.

## Provenance
Digital Earth Australia Coastlines is a Geoscience Australia vector product. Current documentation identifies version 3.1.0, coverage through 2025, and a Creative Commons Attribution 4.0 licence.
