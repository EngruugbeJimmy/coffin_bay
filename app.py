import io
import os
import zipfile
from pathlib import Path

import requests
import re

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
import geopandas as gpd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

try:
    from xgboost import XGBRegressor
    XGB_OK = True
except Exception:
    XGB_OK = False

try:
    from pykrige.ok import OrdinaryKriging
    PYKRIGE_OK = True
except Exception:
    OrdinaryKriging = None
    PYKRIGE_OK = False

try:
    from scipy.interpolate import Rbf
    SCIPY_OK = True
except Exception:
    Rbf = None
    SCIPY_OK = False

try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except Exception:
    TORCH_OK = False

st.set_page_config(
    page_title="Coffin Bay Groundwater Intelligence",
    page_icon="GB",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# FUTURISTIC BLUE-GREEN HYDROLOGY UI
# ============================================================
st.markdown(
    """
    <style>
    :root{
      --ink:#123b42;--deep:#075f69;--teal:#0f8f91;--aqua:#35c4b5;
      --pale:#edf9f6;--mint:#f8fcfb;--sand:#f3efe2;--line:#cbe7e1;
      --muted:#6a8587;--amber:#c7a24a;--navy:#0a3d46
    }
    html,body,[class*="css"]{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
    .stApp{background:
      radial-gradient(circle at 88% 0%,rgba(52,196,181,.09),transparent 26%),
      radial-gradient(circle at 10% 10%,rgba(8,95,105,.08),transparent 30%),
      linear-gradient(135deg,#f1faf8 0%,#f9f8f2 55%,#edf8f6 100%);
      color:var(--ink)}
    [data-testid="stHeader"]{background:rgba(248,253,252,.82);backdrop-filter:blur(12px)}
    [data-testid="stSidebar"]{background:linear-gradient(180deg,#083f48 0%,#075965 48%,#0a4951 100%);border-right:1px solid rgba(255,255,255,.08)}
    [data-testid="stSidebar"] *{color:#eefcf9!important}
    [data-testid="stSidebar"] .stCaption{color:#b8d7d2!important}
    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"]{background:rgba(53,196,181,.86)!important;border:0!important}
    [data-testid="stSidebar"] .stRadio > div{gap:.2rem}
    .hero{background:linear-gradient(135deg,rgba(255,255,255,.96),rgba(236,250,247,.92));border:1px solid var(--line);border-radius:24px;padding:22px 26px;box-shadow:0 18px 40px rgba(11,73,77,.08);position:relative;overflow:hidden}
    .hero:after{content:"";position:absolute;right:-30px;top:-55px;width:190px;height:190px;border-radius:50%;background:radial-gradient(circle,rgba(52,196,181,.24),rgba(52,196,181,0) 66%)}
    .hero-title{font-size:31px;font-weight:850;letter-spacing:-.8px;margin:0;color:var(--ink)}
    .hero-sub{margin-top:4px;color:var(--muted);font-size:13px}
    .chip{display:inline-block;margin-top:12px;padding:6px 11px;border-radius:999px;background:#e3f5ef;color:#0a6a5f;font-size:11px;font-weight:800;letter-spacing:.2px}
    .section{font-size:18px;font-weight:850;color:var(--ink);margin:20px 0 10px}
    .panel{background:rgba(255,255,255,.93);border:1px solid var(--line);border-radius:18px;padding:16px 18px;box-shadow:0 9px 26px rgba(15,76,76,.055)}
    .panel-accent{background:linear-gradient(135deg,#f7fffd,#eef9f4);border:1px solid #bfe1d9}
    .hydro-note{background:#fffdf0;border:1px solid #e8dfb1;border-left:5px solid var(--amber);border-radius:14px;padding:12px 14px;color:#5c542f;font-size:12px}
    .hydro-good{background:#ebfaf4;border:1px solid #bce1d5;border-left:5px solid var(--teal);border-radius:14px;padding:12px 14px;color:#1f5f54;font-size:12px}
    .small{font-size:12px;color:var(--muted)}
    .anchor-card{padding:10px 12px;border-radius:14px;background:#f4fbf8;border:1px solid #cfe8e1;margin-top:7px}
    div[data-testid="stMetric"]{background:rgba(255,255,255,.95);border:1px solid var(--line);padding:12px 14px;border-radius:16px;box-shadow:0 7px 19px rgba(16,73,73,.05)}
    .stButton>button,.stDownloadButton>button{border-radius:11px!important;border:1px solid #9ed9ce!important;background:#fff!important;color:#0b6664!important;font-weight:700!important}
    .stButton>button:hover,.stDownloadButton>button:hover{border-color:#63c3b4!important;background:#effaf7!important}
    .map-caption{display:flex;justify-content:space-between;gap:14px;flex-wrap:wrap;color:#668587;font-size:11px;margin-top:-8px;margin-bottom:4px}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# STUDY-AREA / HYDROLOGIC CONTEXT
# ============================================================
DATA_DIR = Path(__file__).resolve().parent
RIZIN_GEOJSON = DATA_DIR / "rizin.geojson"
RIZIN_SHP = DATA_DIR / "rizin.shp"

LAKE_WANGARY = {"name":"Lake Wangary", "latitude":-34.54259, "longitude":135.49462, "level_mAHD":2.5}
COAST_ANCHOR = {"name":"Coastal boundary", "level_mAHD":0.0}

@st.cache_data

def load_rizin_boundary():
    if RIZIN_GEOJSON.exists():
        gdf = gpd.read_file(RIZIN_GEOJSON)
    elif RIZIN_SHP.exists():
        os.environ["SHAPE_RESTORE_SHX"] = "YES"
        gdf = gpd.read_file(RIZIN_SHP)
    else:
        return None
    if gdf.empty:
        return None
    if gdf.crs is None:
        # The uploaded Rizin geometry is in the Coffin Bay MGA 53 coordinate domain.
        gdf = gdf.set_crs(28353)
    return gdf.to_crs(4326)

BOUNDARY = load_rizin_boundary()


def boundary_polygon():
    if BOUNDARY is None:
        return None
    geom = unary_union(BOUNDARY.geometry)
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    return None

AOI_POLY = boundary_polygon()


# ============================================================
# DEA COASTLINE / HYDROGEOGRAPHIC REFERENCE
# ============================================================
DEA_WFS_URL = "https://geoserver.dea.ga.gov.au/geoserver/dea/wfs"
DEA_COAST_LAYER = "dea:shorelines_annual"

@st.cache_data(ttl=24*60*60, show_spinner=False)
def load_dea_coastline(aoi_geojson_text, year=2024):
    """Fetch the DEA annual shoreline by AOI bbox and clip it to Rizin."""
    try:
        aoi = gpd.read_file(io.StringIO(aoi_geojson_text)).to_crs(4326)
        if aoi.empty:
            return None, "Rizin AOI is empty."
        xmin, ymin, xmax, ymax = aoi.total_bounds
        params = {
            "service":"WFS", "version":"1.1.0", "request":"GetFeature",
            "typeName":DEA_COAST_LAYER, "maxFeatures":"2000",
            "bbox":f"{ymin},{xmin},{ymax},{xmax},urn:ogc:def:crs:EPSG:4326",
            "outputFormat":"application/json",
            "CQL_FILTER":f"year={int(year)}",
        }
        r = requests.get(DEA_WFS_URL, params=params, timeout=45)
        r.raise_for_status()
        try:
            payload = r.json()
        except Exception:
            payload = {}
        features = payload.get("features", [])
        if not features:
            # Some WFS deployments reject CQL filters. Retry the AOI-bounded query
            # without the server-side year filter, then filter locally.
            params.pop("CQL_FILTER", None)
            r = requests.get(DEA_WFS_URL, params=params, timeout=45)
            r.raise_for_status()
            features = r.json().get("features", [])
        if not features:
            return None, f"DEA Coastlines returned no features for {year}."
        coast = gpd.GeoDataFrame.from_features(features, crs=3577)
        if "year" in coast.columns:
            yy=pd.to_numeric(coast["year"],errors="coerce")
            selected=coast[yy.eq(int(year))].copy()
            if not selected.empty:
                coast=selected
        coast = coast[coast.geometry.notna() & ~coast.geometry.is_empty].copy()
        if coast.empty:
            return None, "DEA Coastlines returned empty geometries."
        clipped = gpd.clip(coast.to_crs(28353), aoi.to_crs(28353))
        clipped = clipped[clipped.geometry.notna() & ~clipped.geometry.is_empty].copy()
        if clipped.empty:
            return None, "The DEA shoreline does not intersect the Rizin AOI."
        return clipped.to_crs(4326), f"DEA Coastlines · {year}"
    except Exception as exc:
        return None, f"DEA Coastlines unavailable: {exc}"


def attach_coast_distance(data, coastline_gdf):
    """Shortest perpendicular/normal distance from every well to the coastline, in metres."""
    d = data.copy()
    if coastline_gdf is None or coastline_gdf.empty:
        return d
    try:
        pts = gpd.GeoSeries(gpd.points_from_xy(d["longitude"], d["latitude"]), crs=4326).to_crs(28353)
        coast_union = unary_union(coastline_gdf.to_crs(28353).geometry)
        d["distance_coast_m"] = pts.distance(coast_union).to_numpy(float)
    except Exception:
        pass
    return d


def coast_label_point(coastline_gdf, center_lon, center_lat):
    if coastline_gdf is None or coastline_gdf.empty:
        return None
    try:
        coast_union = unary_union(coastline_gdf.to_crs(28353).geometry)
        center = gpd.GeoSeries([Point(center_lon, center_lat)], crs=4326).to_crs(28353).iloc[0]
        q = coast_union.interpolate(coast_union.project(center))
        return gpd.GeoSeries([q], crs=28353).to_crs(4326).iloc[0]
    except Exception:
        return None

def point_inside_boundary(lon, lat):
    if AOI_POLY is None:
        return True
    return AOI_POLY.contains(Point(float(lon), float(lat))) or AOI_POLY.touches(Point(float(lon), float(lat)))


# ============================================================
# SYNTHETIC SPATIO-TEMPORAL DATA
# Wells follow an inland-to-coast corridor inside Rizin, not the
# Rizin boundary itself. DEM + coast distance are location-driven.
# ============================================================
@st.cache_data(show_spinner=False)
def _fetch_dem_open_meteo_cached(coords, batch_size=100):
    """Cached Open-Meteo DEM lookup keyed only by rounded (lat, lon) pairs."""
    elevations=np.full(len(coords),np.nan,dtype=float)
    batch_size=min(int(batch_size),100)
    for start in range(0,len(coords),batch_size):
        chunk=coords[start:start+batch_size]
        lat_list=",".join(f"{lat:.5f}" for lat,lon in chunk)
        lon_list=",".join(f"{lon:.5f}" for lat,lon in chunk)
        url="https://api.open-meteo.com/v1/elevation"
        params={"latitude":lat_list,"longitude":lon_list}
        try:
            resp=requests.get(url,params=params,timeout=30)
            resp.raise_for_status()
            values=resp.json().get("elevation",[])
            if len(values)!=len(chunk):
                raise ValueError(f"Expected {len(chunk)} elevations, received {len(values)}")
            elevations[start:start+len(chunk)]=pd.to_numeric(values,errors="coerce")
        except Exception as e:
            print(f"Elevation fetch failed for rows {start}-{start+len(chunk)}: {e}")
    return elevations


def fetch_dem_open_meteo(df, lat_col="latitude", lon_col="longitude", batch_size=100):
    """
    Fetch terrain elevation from the Open-Meteo Elevation API.

    The API endpoint accepts at most 100 coordinate pairs per request, so
    requests are always chunked to <=100 points. Results are cached by the
    rounded latitude/longitude coordinates so Streamlit reruns do not repeat
    lookups for the same static well locations. Returned elevations preserve
    the input DataFrame row order.
    """
    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(f"DataFrame must contain {lat_col!r} and {lon_col!r}")

    coords_df=pd.DataFrame({
        "latitude":pd.to_numeric(df[lat_col],errors="coerce"),
        "longitude":pd.to_numeric(df[lon_col],errors="coerce")
    })
    coords_df["latitude"]=coords_df["latitude"].round(5)
    coords_df["longitude"]=coords_df["longitude"].round(5)

    unique_coords=list(dict.fromkeys(
        (float(row.latitude),float(row.longitude))
        for row in coords_df.itertuples(index=False)
        if np.isfinite(row.latitude) and np.isfinite(row.longitude)
    ))

    if not unique_coords:
        return np.full(len(df),np.nan,dtype=float)

    unique_elevations=_fetch_dem_open_meteo_cached(
        tuple(unique_coords),
        batch_size=min(int(batch_size),100)
    )
    lookup=dict(zip(unique_coords,unique_elevations))

    return np.array([
        lookup.get((float(lat),float(lon)),np.nan)
        if np.isfinite(lat) and np.isfinite(lon) else np.nan
        for lat,lon in zip(coords_df["latitude"],coords_df["longitude"])
    ],dtype=float)


def apply_dem_open_meteo_fallback(df):
    """Fill only missing DEM values; never overwrite user-supplied dem_m."""
    d=df.copy()
    if "dem_m" not in d.columns:
        d["dem_m"]=np.nan

    d["dem_m"]=pd.to_numeric(d["dem_m"],errors="coerce")
    missing=d["dem_m"].isna()
    if not missing.any():
        return d

    # Query unique spatial well locations, not every temporal observation.
    targets=d.loc[missing,["well_id","latitude","longitude"]].drop_duplicates("well_id")
    if targets.empty:
        return d

    elevations=fetch_dem_open_meteo(targets,lat_col="latitude",lon_col="longitude",batch_size=100)
    lookup=dict(zip(targets["well_id"].astype(str),elevations))
    fetched=d["well_id"].astype(str).map(lookup)
    d.loc[missing,"dem_m"]=fetched.loc[missing]
    return d


@st.cache_data
def make_data(n_wells=1200, years=5, seed=42, well_points=None):
    """Create a five-year monthly synthetic groundwater time series.

    Demo mode represents 1,200 fixed well locations inside the Rizin/Coffin Bay AOI
    and repeats each location for 60 monthly observations (5 years = 72,000 rows).
    The synthetic target is explicitly scenario data: useful for exercising spatial,
    seasonal, recurring-variability and trend-modelling workflows, not observations.
    """
    rng = np.random.default_rng(seed)
    months_per_year = 12
    total_months = int(years * months_per_year)

    if well_points is not None and len(well_points) > 0:
        wp = well_points.copy()
        if wp.crs is not None:
            wp = wp.to_crs(4326)
        wp = wp[wp.geometry.notna() & ~wp.geometry.is_empty].copy()
        wp = wp[wp.geometry.geom_type.eq("Point")].copy()
        if AOI_POLY is not None:
            wp = wp[wp.geometry.apply(lambda q: AOI_POLY.contains(q) or AOI_POLY.touches(q))].copy()
        wp = wp.drop_duplicates(subset=["geometry"]).reset_index(drop=True)
        if len(wp) > n_wells:
            wp = wp.iloc[:n_wells].copy()
        n_wells = len(wp)
        lon0 = wp.geometry.x.to_numpy(float)
        lat0 = wp.geometry.y.to_numpy(float)
        minx,miny,maxx,maxy = AOI_POLY.bounds if AOI_POLY is not None else (lon0.min(),lat0.min(),lon0.max(),lat0.max())
        transect_t = np.clip((lon0-minx)/max(1e-9,maxx-minx),0,1)
    elif AOI_POLY is not None:
        # Deterministic spatial sample: exactly 1,200 points inside the AOI by default.
        minx,miny,maxx,maxy=AOI_POLY.bounds
        pts=[]; tries=0
        while len(pts)<n_wells and tries<400000:
            tries += 1
            lon=rng.uniform(minx,maxx); lat=rng.uniform(miny,maxy)
            if point_inside_boundary(lon,lat):
                t=(lon-minx)/max(1e-9,maxx-minx)
                pts.append((lon,lat,t))
        coords=np.asarray(pts)
        lon0,lat0,transect_t=coords[:,0],coords[:,1],coords[:,2]
    else:
        lon0=135.08+rng.uniform(0,.63,n_wells)
        lat0=-34.75+rng.uniform(0,.42,n_wells)
        transect_t=(lon0-lon0.min())/(lon0.max()-lon0.min())

    # Real terrain elevation from Open-Meteo. Coordinates are static per well,
    # so this lookup happens once per unique coordinate set and is cached.
    well_coords=pd.DataFrame({"latitude":lat0,"longitude":lon0})
    dem0=fetch_dem_open_meteo(well_coords,batch_size=100)

    # Keep the demo usable if the external API is temporarily unavailable.
    # This fallback is only for failed API responses; successful Open-Meteo
    # elevations always replace the old synthetic DEM generation.
    if np.isnan(dem0).any():
        coast_distance=np.clip(140 + 14500*transect_t + rng.normal(0,420,n_wells),25,16000)
        inlandness=np.clip(coast_distance/np.nanmax(coast_distance),0,1)
        fallback_dem=np.clip(0.25 + 34*(inlandness**0.72) + rng.normal(0,0.9,n_wells),0.05,40)
        dem0=np.where(np.isfinite(dem0),dem0,fallback_dem)
    else:
        coast_distance=np.clip(140 + 14500*transect_t + rng.normal(0,420,n_wells),25,16000)
        inlandness=np.clip(coast_distance/np.nanmax(coast_distance),0,1)
    gs=np.array(["Bridgewater Formation","Uley Formation","Wanilla Formation",
                 "Sleaford Complex","Hutchison Supergroup","Kiana Granite"])
    geo0=rng.choice(gs,n_wells,p=[.38,.14,.12,.15,.09,.12])
    gf0=pd.Series(geo0).map({
        "Bridgewater Formation":1.25,"Uley Formation":.8,"Wanilla Formation":.4,
        "Sleaford Complex":1.55,"Hutchison Supergroup":-.35,"Kiana Granite":1.0
    }).to_numpy()
    dist_lake=np.sqrt(
        ((lon0-LAKE_WANGARY["longitude"])/0.0058)**2 +
        ((lat0-LAKE_WANGARY["latitude"])/0.0048)**2
    )*1000

    # Static per-well hydrogeological structure. These demo values are explicitly
    # synthetic and are repeated unchanged across all 60 temporal observations.
    depth_to_basement_m=np.clip(18 + 115*(1-inlandness) + rng.normal(0,7,n_wells),8,160)
    aquifer_thickness_m=np.clip(4 + 32*inlandness + rng.normal(0,2.5,n_wells),2,45)
    clay_layer_total_m=np.clip(1.5 + 15*(1-inlandness) + rng.normal(0,1.2,n_wells),0.2,20)
    hydraulic_conductivity_K=np.clip(
        pd.Series(geo0).map({
            "Bridgewater Formation":18.0,"Uley Formation":9.0,"Wanilla Formation":5.0,
            "Sleaford Complex":2.5,"Hutchison Supergroup":0.8,"Kiana Granite":0.15
        }).to_numpy() * np.exp(rng.normal(0,0.28,n_wells)), 0.05, 35
    )
    specific_yield_Sy=np.clip(
        pd.Series(geo0).map({
            "Bridgewater Formation":0.28,"Uley Formation":0.20,"Wanilla Formation":0.16,
            "Sleaford Complex":0.08,"Hutchison Supergroup":0.05,"Kiana Granite":0.015
        }).to_numpy() + rng.normal(0,0.018,n_wells), 0.005, 0.35
    )
    is_confined=((clay_layer_total_m > 9) & (aquifer_thickness_m > 15)).astype(int)

    # Five complete calendar years: 2021-01 through 2025-12.
    dates=pd.date_range("2021-01-01",periods=total_months,freq="MS")
    rows=[]
    well_effect=rng.normal(0,0.34,n_wells)
    season_map={12:"Summer",1:"Summer",2:"Summer",3:"Autumn",4:"Autumn",5:"Autumn",
                6:"Winter",7:"Winter",8:"Winter",9:"Spring",10:"Spring",11:"Spring"}
    season_factor={"Summer":-.25,"Autumn":.03,"Winter":.46,"Spring":.20}

    for i in range(n_wells):
        for j,dt in enumerate(dates):
            yr=int(dt.year); month=int(dt.month); year_idx=yr-dates[0].year
            season=season_map[month]
            # Smooth annual cycle plus inter-annual variability and a modest trend.
            phase=2*np.pi*(month-1)/12
            wet_cycle=95*np.sin(2*np.pi*year_idx/3.2)
            rain=np.clip(rng.normal(520 + wet_cycle + 85*np.sin(phase-np.pi/2),55),220,900)
            et=np.clip(rng.normal(1050 - 0.30*wet_cycle + 105*np.sin(phase),65),700,1400)
            nd=np.clip(rng.normal(.50 + .055*np.sin(phase+0.7),.07),.18,.84)
            nda=np.clip(rng.normal(.06*np.sin(phase),.055),-.25,.25)
            pressure=rng.normal(1013,7)
            sw=np.clip(dist_lake[i]+rng.normal(0,260),30,11000)
            dem=float(np.clip(dem0[i]+rng.normal(0,.08),.05,40))
            coast=float(coast_distance[i])
            coastal_lift=0.55*(coast/10000.0)
            lake_influence=1.15*np.exp(-dist_lake[i]/4200.0)
            trend=0.028*year_idx
            cyclical=0.22*np.sin(2*np.pi*year_idx/3.0)
            seasonal_wave=0.20*np.sin(phase)
            gw=(0.15 + 0.54*dem + coastal_lift
                + lake_influence*(LAKE_WANGARY["level_mAHD"]-0.8)
                + 0.72*gf0[i] + .006*(rain-500) - .0022*(et-1000)
                + 1.20*nd + .62*nda + .025*(pressure-1013)
                + season_factor[season] + seasonal_wave + trend + cyclical
                + well_effect[i] + rng.normal(0,.30))
            rows.append([
                f"CB_{i+1:05d}",lon0[i],lat0[i],dem,coast,geo0[i],gf0[i],
                depth_to_basement_m[i],aquifer_thickness_m[i],clay_layer_total_m[i],
                hydraulic_conductivity_K[i],specific_yield_Sy[i],is_confined[i],
                nd,nda,rain,et,sw,pressure,yr,month,dt,season,
                max(-.15,gw),dist_lake[i],j
            ])

    return pd.DataFrame(rows,columns=[
        "well_id","longitude","latitude","dem_m","distance_coast_m","geology_formation",
        "geology_factor","depth_to_basement_m","aquifer_thickness_m","clay_layer_total_m",
        "hydraulic_conductivity_K","specific_yield_Sy","is_confined",
        "ndvi_mean","ndvi_anomaly","rainfall_mm","et_mm",
        "surface_water_distance_m","pressure_hpa","year","month","date","season",
        "groundwater_level_mAHD","distance_lake_wangary_m","time_index"
    ])


def normalise_columns(df):
    d=df.copy()
    COLUMN_ALIASES={
        "well_id":["well_id","well","site","id","bore_id"],
        "longitude":["longitude","lon","x_lon"],
        "latitude":["latitude","lat","y_lat"],
        "groundwater_level_mAHD":["groundwater_level_mAHD","groundwater_level","water_level","gw_level","head_mAHD"],
        "dem_m":["dem_m","dem","elevation","elev_m"],
        "distance_coast_m":["distance_coast_m","coast_distance_m","distance_to_coast_m"],
        "geology_formation":["geology_formation","formation","lithology"],
        "depth_to_basement_m":["depth_to_basement_m","basement_depth_m","depth_basement_m"],
        "aquifer_thickness_m":["aquifer_thickness_m","aquifer_thickness","aquifer_thick_m"],
        "clay_layer_total_m":["clay_layer_total_m","total_clay_m","clay_thickness_m"],
        "hydraulic_conductivity_K":["hydraulic_conductivity_K","hydraulic_conductivity","k","conductivity_K"],
        "specific_yield_Sy":["specific_yield_Sy","specific_yield","sy","specific_yield_fraction"],
        "is_confined":["is_confined","confined","confined_aquifer"],
        "season":["season"],"year":["year","date_year"],"month":["month","month_num"],
        "date":["date","datetime","timestamp","observation_date"]
    }
    lower={str(c).strip().lower():c for c in d.columns}
    for target,names in COLUMN_ALIASES.items():
        if target not in d.columns:
            found=next((lower.get(n.lower()) for n in names if n.lower() in lower),None)
            if found is not None:d[target]=d[found]

    if "well_id" not in d.columns:d["well_id"]=[f"CB_{i:05d}" for i in range(1,len(d)+1)]
    for c,default in [
        ("dem_m",np.nan),("distance_coast_m",np.nan),("geology_formation","Unknown"),
        ("depth_to_basement_m",np.nan),("aquifer_thickness_m",np.nan),
        ("clay_layer_total_m",np.nan),("hydraulic_conductivity_K",np.nan),
        ("specific_yield_Sy",np.nan),("is_confined",np.nan),
        ("season","Unknown"),("year",2025),("month",1)
    ]:
        if c not in d.columns:d[c]=default

    numeric_cols=[
        "longitude","latitude","groundwater_level_mAHD","dem_m","distance_coast_m",
        "depth_to_basement_m","aquifer_thickness_m","clay_layer_total_m",
        "hydraulic_conductivity_K","specific_yield_Sy","is_confined","year","month"
    ]
    for c in numeric_cols:
        if c in d:d[c]=pd.to_numeric(d[c],errors="coerce")
    if "is_confined" in d.columns:
        d["is_confined"]=d["is_confined"].where(d["is_confined"].isna(),d["is_confined"].round().clip(0,1)).astype("Int64")

    d=d.dropna(subset=["longitude","latitude","groundwater_level_mAHD"]).reset_index(drop=True)
    d["geology_formation"]=d["geology_formation"].fillna("Unknown").astype(str)
    if "date" in d.columns:
        d["date"]=pd.to_datetime(d["date"],errors="coerce")
        d.loc[d["date"].notna(),"year"]=d.loc[d["date"].notna(),"date"].dt.year
        d.loc[d["date"].notna(),"month"]=d.loc[d["date"].notna(),"date"].dt.month
    d["month"]=pd.to_numeric(d["month"],errors="coerce").fillna(1).clip(1,12).astype(int)
    season_lookup={12:"Summer",1:"Summer",2:"Summer",3:"Autumn",4:"Autumn",5:"Autumn",6:"Winter",7:"Winter",8:"Winter",9:"Spring",10:"Spring",11:"Spring"}
    d["season"]=d["season"].replace("Unknown",np.nan).fillna(d["month"].map(season_lookup)).fillna("Unknown").astype(str)
    if "date" not in d.columns or d["date"].isna().all():
        d["date"]=pd.to_datetime(d["year"].astype(int).astype(str)+"-"+d["month"].astype(int).astype(str)+"-01",errors="coerce")
    else:
        missing=d["date"].isna()
        d.loc[missing,"date"]=pd.to_datetime(d.loc[missing,"year"].astype(int).astype(str)+"-"+d.loc[missing,"month"].astype(int).astype(str)+"-01",errors="coerce")
    d=d.sort_values(["well_id","date"]).reset_index(drop=True)
    if len(d):
        origin=d["date"].min()
        d["time_index"]=((d["date"].dt.year-origin.year)*12+d["date"].dt.month-origin.month).astype(int)
    else:d["time_index"]=pd.Series(dtype=int)
    d["month_sin"]=np.sin(2*np.pi*(d["month"]-1)/12)
    d["month_cos"]=np.cos(2*np.pi*(d["month"]-1)/12)
    return d


FEATURE_CANDIDATES=[
    "longitude","latitude","dem_m","distance_coast_m","geology_factor",
    "depth_to_basement_m","aquifer_thickness_m","clay_layer_total_m",
    "hydraulic_conductivity_K","specific_yield_Sy","is_confined",
    "ndvi_mean","ndvi_anomaly","rainfall_mm","et_mm","surface_water_distance_m",
    "distance_lake_wangary_m","pressure_hpa","year","month","time_index","month_sin","month_cos"
]


SUBSURFACE_COLUMNS=[
    "depth_to_basement_m","aquifer_thickness_m","clay_layer_total_m",
    "hydraulic_conductivity_K","specific_yield_Sy","is_confined"
]


def _project_lonlat_to_local_m(x, y):
    """Convert longitude/latitude to a local metric approximation around the dataset."""
    lon0=float(np.nanmean(x)); lat0=float(np.nanmean(y))
    x_m=(np.asarray(x,dtype=float)-lon0)*111320.0*np.cos(np.deg2rad(lat0))
    y_m=(np.asarray(y,dtype=float)-lat0)*110540.0
    return x_m,y_m


def interpolate_subsurface(known_boreholes, target_wells, value_columns=None,
                           search_radius_m=5000.0, min_points=5, max_local_points=30):
    """
    Interpolate static subsurface properties from REAL bore-log wells only.

    known_boreholes must contain only wells with genuine bore-log measurements.
    target_wells is the full unique well grid to which the static properties are
    attached. No temporal columns are used here.

    Each `{column}_source` is one of: observed, interpolated, insufficient_data.
    A target with < min_points known values inside search_radius_m is never
    extrapolated.
    """
    value_columns=list(value_columns or SUBSURFACE_COLUMNS)
    required=["well_id","longitude","latitude"]
    missing=[c for c in required if c not in known_boreholes.columns or c not in target_wells.columns]
    if missing: raise ValueError(f"Subsurface interpolation requires columns: {missing}")
    if not (PYKRIGE_OK or SCIPY_OK):
        raise ImportError("Neither pykrige nor scipy is available for subsurface interpolation.")

    known=known_boreholes.copy()
    targets=target_wells[["well_id","longitude","latitude"]].drop_duplicates("well_id").copy()

    # Collapse repeated temporal records to one static value per borehole.
    for c in value_columns:
        if c not in known.columns: known[c]=np.nan
        known[c]=pd.to_numeric(known[c],errors="coerce")
    known["longitude"]=pd.to_numeric(known["longitude"],errors="coerce")
    known["latitude"]=pd.to_numeric(known["latitude"],errors="coerce")
    known=known.dropna(subset=["well_id","longitude","latitude"])
    known=known.groupby("well_id",as_index=False).first()

    out=targets.copy()
    kx,ky=_project_lonlat_to_local_m(known["longitude"].to_numpy(),known["latitude"].to_numpy())
    tx,ty=_project_lonlat_to_local_m(out["longitude"].to_numpy(),out["latitude"].to_numpy())

    # A well is considered observed independently for each property.
    for col in value_columns:
        out[col]=np.nan
        out[f"{col}_source"]="insufficient_data"
        valid=known[col].notna().to_numpy()
        if not valid.any():
            continue
        vx,vy=kx[valid],ky[valid]
        vv=known.loc[valid,col].to_numpy(float)
        vwell=known.loc[valid,"well_id"].astype(str).to_numpy()

        for i,(wid,x0,y0) in enumerate(zip(out["well_id"].astype(str),tx,ty)):
            same=np.where(vwell==wid)[0]
            if len(same):
                out.at[i,col]=float(vv[same[0]])
                out.at[i,f"{col}_source"]="observed"
                continue

            dist=np.hypot(vx-x0,vy-y0)
            local_idx=np.flatnonzero(dist<=float(search_radius_m))
            if len(local_idx)<int(min_points):
                out.at[i,f"{col}_source"]="insufficient_data"
                continue
            if len(local_idx)>int(max_local_points):
                local_idx=local_idx[np.argsort(dist[local_idx])[:int(max_local_points)]]

            try:
                if PYKRIGE_OK:
                    ok=OrdinaryKriging(
                        vx[local_idx],vy[local_idx],vv[local_idx],
                        variogram_model="linear",verbose=False,enable_plotting=False
                    )
                    pred,_=ok.execute("points",np.array([x0]),np.array([y0]))
                    value=float(np.asarray(pred).ravel()[0])
                else:
                    rbf=Rbf(vx[local_idx],vy[local_idx],vv[local_idx],
                            function="linear",smooth=0.0)
                    value=float(rbf(x0,y0))
                if np.isfinite(value):
                    out.at[i,col]=value
                    out.at[i,f"{col}_source"]="interpolated"
            except Exception:
                out.at[i,f"{col}_source"]="insufficient_data"

    return out


def merge_subsurface_features(data, known_boreholes=None, search_radius_m=5000.0):
    """
    Merge one static subsurface record per well into the temporal groundwater data.
    Only `known_boreholes` is used as an interpolation source; synthetic/demo rows
    are never used as kriging inputs.
    """
    d=data.copy()
    for c in SUBSURFACE_COLUMNS:
        if c not in d.columns:d[c]=np.nan

    if known_boreholes is None or known_boreholes.empty:
        for c in SUBSURFACE_COLUMNS:
            d[f"{c}_source"]=np.where(d[c].notna(),"observed","insufficient_data")
        return d

    target_wells=d[["well_id","longitude","latitude"]].drop_duplicates("well_id")
    sub=interpolate_subsurface(
        known_boreholes=known_boreholes,
        target_wells=target_wells,
        value_columns=SUBSURFACE_COLUMNS,
        search_radius_m=search_radius_m
    )
    source_cols=["well_id"]+[c for c in sub.columns if c in SUBSURFACE_COLUMNS or c.endswith("_source")]
    return d.drop(columns=[c for c in source_cols if c != "well_id" and c in d.columns]).merge(
        sub[source_cols],on="well_id",how="left",validate="many_to_one"
    )


def prepare_features(data):
    d=data.copy(); y=pd.to_numeric(d["groundwater_level_mAHD"],errors="coerce"); d=d[y.notna()].copy(); y=y[y.notna()]
    f=[c for c in FEATURE_CANDIDATES if c in d.columns and (pd.to_numeric(d[c],errors="coerce").notna().any())]
    for required in ["longitude","latitude"]:
        if required not in f:f.append(required)
    X=d[f].copy()
    for c in f:
        X[c]=pd.to_numeric(X[c],errors="coerce"); X[c]=X[c].fillna(X[c].median() if X[c].notna().any() else 0.0)
    return d.reset_index(drop=True),X.reset_index(drop=True),y.reset_index(drop=True),f


def metrics_dict(y,p):
    return {"R²":r2_score(y,p) if len(y)>1 else np.nan,"MAE (m)":mean_absolute_error(y,p),"RMSE (m)":mean_squared_error(y,p)**.5}


def fit_rf(d,X,y,f,idx_train,idx_test):
    m=RandomForestRegressor(n_estimators=350,max_depth=16,min_samples_leaf=3,random_state=42,n_jobs=-1)
    m.fit(X.iloc[idx_train],y.iloc[idx_train]); pred=m.predict(X)
    return m,pred,metrics_dict(y.iloc[idx_test],pred[idx_test]),pd.Series(m.feature_importances_,index=f).sort_values(ascending=False)


def fit_xgb(d,X,y,f,idx_train,idx_test):
    if not XGB_OK: raise RuntimeError("xgboost is not installed")
    m=XGBRegressor(n_estimators=500,max_depth=6,learning_rate=.045,subsample=.85,colsample_bytree=.9,objective="reg:squarederror",random_state=42,n_jobs=4)
    m.fit(X.iloc[idx_train],y.iloc[idx_train],verbose=False); pred=m.predict(X)
    return m,pred,metrics_dict(y.iloc[idx_test],pred[idx_test]),pd.Series(m.feature_importances_,index=f).sort_values(ascending=False)


def fit_gam(d,X,y,f,idx_train,idx_test):
    m=Pipeline([("spline",SplineTransformer(n_knots=min(7,max(3,len(X)//120)),degree=3,include_bias=False)),("ridge",Ridge(alpha=1.0))])
    m.fit(X.iloc[idx_train],y.iloc[idx_train]); pred=m.predict(X)
    pi=permutation_importance(m,X.iloc[idx_test],y.iloc[idx_test],n_repeats=5,random_state=42,scoring="neg_root_mean_squared_error")
    imp=pd.Series(np.maximum(pi.importances_mean,0),index=f).sort_values(ascending=False)
    return m,pred,metrics_dict(y.iloc[idx_test],pred[idx_test]),imp


if TORCH_OK:
    class LSTMRegressor(nn.Module):
        def __init__(self,input_size,hidden=48):
            super().__init__(); self.lstm=nn.LSTM(input_size,hidden,batch_first=True); self.head=nn.Sequential(nn.Linear(hidden,24),nn.ReLU(),nn.Linear(24,1))
        def forward(self,x):
            out,_=self.lstm(x); return self.head(out[:,-1,:]).squeeze(-1)


def make_sequences(d,X,y,f,seq_len=12):
    if "well_id" not in d or "year" not in d: raise RuntimeError("LSTM requires well_id and year columns")
    order=d.copy(); order["_row"]=np.arange(len(order)); order=order.sort_values(["well_id","date","_row"])
    xs=[]; ys=[]; rows=[]; groups=[]
    for wid,g in order.groupby("well_id"):
        inds=g["_row"].to_numpy(int); vals=X.iloc[inds].to_numpy(float); tgt=y.iloc[inds].to_numpy(float)
        if len(vals)<2: continue
        for j in range(len(vals)):
            a=max(0,j-seq_len+1); seq=vals[a:j+1]
            if len(seq)<seq_len: seq=np.vstack([np.repeat(seq[[0]],seq_len-len(seq),axis=0),seq])
            xs.append(seq); ys.append(tgt[j]); rows.append(inds[j]); groups.append(wid)
    return np.asarray(xs,dtype=np.float32),np.asarray(ys,dtype=np.float32),np.asarray(rows,int),np.asarray(groups)


def fit_lstm(d,X,y,f,seq_len=12):
    if not TORCH_OK: raise RuntimeError("PyTorch is not installed")
    xs,ys,rows,groups=make_sequences(d,X,y,f,seq_len)
    unique=np.unique(groups)
    if len(unique)<10: raise RuntimeError("LSTM needs repeated well observations (at least ~10 wells with time series)")
    rng=np.random.default_rng(42); rng.shuffle(unique); cut=max(1,int(.2*len(unique))); test_wells=set(unique[:cut]); test_mask=np.array([g in test_wells for g in groups]); train_mask=~test_mask
    scaler=StandardScaler().fit(xs[train_mask].reshape(-1,len(f)))
    xtr=torch.tensor(scaler.transform(xs[train_mask].reshape(-1,len(f))).reshape(-1,seq_len,len(f)),dtype=torch.float32); ytr=torch.tensor(ys[train_mask],dtype=torch.float32)
    xall=torch.tensor(scaler.transform(xs.reshape(-1,len(f))).reshape(-1,seq_len,len(f)),dtype=torch.float32)
    model=LSTMRegressor(len(f)); opt=torch.optim.Adam(model.parameters(),lr=.003); loss_fn=nn.MSELoss(); model.train(); batch=128
    for _ in range(45):
        perm=torch.randperm(len(xtr))
        for s in range(0,len(xtr),batch):
            ix=perm[s:s+batch]; opt.zero_grad(); loss=loss_fn(model(xtr[ix]),ytr[ix]); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad(): pred=model(xall).numpy()
    test_pred=pred[test_mask]; test_y=ys[test_mask]; imp=[]; base=mean_squared_error(test_y,test_pred)**.5; rng=np.random.default_rng(42)
    for k,feat in enumerate(f):
        xp=xall.clone(); perm=rng.permutation(len(xp)); xp[:,:,k]=xp[perm,:,k]
        with torch.no_grad(): pp=model(xp).numpy()
        imp.append(max(0,mean_squared_error(test_y,pp[test_mask])**.5-base))
    importance=pd.Series(imp,index=f).sort_values(ascending=False); full_pred=pd.Series(np.nan,index=np.arange(len(d))); full_pred.iloc[rows]=pred
    return (model,scaler,seq_len),full_pred.to_numpy(),metrics_dict(test_y,test_pred),importance


@st.cache_data(show_spinner=False)
def train_models(data,selected_models):
    d,X,y,f=prepare_features(data)
    if d["well_id"].nunique()<len(d):
        # Temporal holdout: last 20% of each well's observations are test data.
        order=d.copy(); order["_i"]=np.arange(len(d)); order=order.sort_values(["well_id","date"])
        test_rows=[]; train_rows=[]
        for _,g in order.groupby("well_id",sort=False):
            inds=g["_i"].to_numpy(int); cut=max(1,int(np.ceil(.2*len(inds))))
            if len(inds)-cut < 1: cut=1
            train_rows.extend(inds[:-cut]); test_rows.extend(inds[-cut:])
        idx_train=np.asarray(train_rows,dtype=int); idx_test=np.asarray(test_rows,dtype=int)
    else: idx_train,idx_test=train_test_split(np.arange(len(d)),test_size=.2,random_state=42)
    results={}; preds={}; imps={}; models={}
    for name in selected_models:
        try:
            if name=="Random Forest": out=fit_rf(d,X,y,f,idx_train,idx_test)
            elif name=="XGBoost": out=fit_xgb(d,X,y,f,idx_train,idx_test)
            elif name=="GAM": out=fit_gam(d,X,y,f,idx_train,idx_test)
            else: out=fit_lstm(d,X,y,f)
            models[name],preds[name],results[name],imps[name]=out
        except Exception as exc: results[name]={"Error":str(exc)}
    table=[]
    for name,m in results.items():
        if "Error" not in m: table.append({"Model":name,**m,"Top feature":imps[name].index[0] if len(imps[name]) else "—"})
    comparison=pd.DataFrame(table).sort_values(["RMSE (m)","MAE (m)"]) if table else pd.DataFrame()
    return d,X,y,f,results,preds,imps,models,comparison

def refresh_synthetic_target(data):
    """Rebuild the demo groundwater target using the actual clipped DEA coast distance."""
    d=data.copy()
    if d.empty or "distance_coast_m" not in d.columns:
        return d
    coast=pd.to_numeric(d["distance_coast_m"],errors="coerce").fillna(d["distance_coast_m"].median()).to_numpy(float)
    dem=pd.to_numeric(d.get("dem_m",0),errors="coerce").fillna(0).to_numpy(float)
    gf=pd.to_numeric(d.get("geology_factor",0),errors="coerce").fillna(0).to_numpy(float)
    rain=pd.to_numeric(d.get("rainfall_mm",500),errors="coerce").fillna(500).to_numpy(float)
    et=pd.to_numeric(d.get("et_mm",1000),errors="coerce").fillna(1000).to_numpy(float)
    nd=pd.to_numeric(d.get("ndvi_mean",.5),errors="coerce").fillna(.5).to_numpy(float)
    nda=pd.to_numeric(d.get("ndvi_anomaly",0),errors="coerce").fillna(0).to_numpy(float)
    pressure=pd.to_numeric(d.get("pressure_hpa",1013),errors="coerce").fillna(1013).to_numpy(float)
    sw=pd.to_numeric(d.get("surface_water_distance_m",np.nan),errors="coerce")
    if sw.isna().all():
        sw=pd.Series(d["distance_lake_wangary_m"])
    sw=sw.fillna(6000).to_numpy(float)
    season=d.get("season",pd.Series(["Unknown"]*len(d))).astype(str).to_numpy()
    yr=pd.to_numeric(d.get("year",2025),errors="coerce").fillna(2025).to_numpy(float)
    sf=np.array([{"Summer":-.25,"Autumn":.03,"Winter":.46,"Spring":.20}.get(x,0.0) for x in season])
    lake=1.15*np.exp(-pd.to_numeric(d["distance_lake_wangary_m"],errors="coerce").fillna(5000).to_numpy(float)/4200.0)
    j=np.maximum(0,yr-yr.min())
    # Deterministic small perturbation preserves repeatability without claiming observations.
    noise=0.42*np.sin(np.arange(len(d))*2.13)
    gw=(0.15 + .54*dem + .000055*coast + lake*(LAKE_WANGARY["level_mAHD"]-.8)
        + .72*gf + .006*(rain-500) - .0022*(et-1000) + 1.20*nd + .62*nda
        + .025*(pressure-1013) + sf + .025*j + noise)
    d["groundwater_level_mAHD"]=np.maximum(-.15,gw)
    return d



def calculate_sgd(df_predictions, grid_cell_width_m=50):
    """
    Calculate coastal submarine groundwater discharge (SGD) using Darcy's Law.

    Uses the app's authoritative ML prediction column, ``active_prediction_mAHD``,
    together with static/interpolated aquifer thickness and hydraulic conductivity.
    Only coastal wells/cells within 1 km of the DEA coastline are included.

    Missing subsurface properties are retained as NaN rather than being converted
    to zero. The returned total therefore sums only calculable coastal cells.
    """
    required=[
        "active_prediction_mAHD",
        "distance_coast_m",
        "aquifer_thickness_m",
        "hydraulic_conductivity_K",
    ]
    missing_cols=[c for c in required if c not in df_predictions.columns]
    if missing_cols:
        raise ValueError(
            "SGD calculation requires columns that are missing: "
            + ", ".join(missing_cols)
        )

    coastal_df=df_predictions.copy()
    coastal_df["distance_coast_m"]=pd.to_numeric(
        coastal_df["distance_coast_m"],errors="coerce"
    )
    coastal_df=coastal_df[
        coastal_df["distance_coast_m"].notna()
        & (coastal_df["distance_coast_m"] <= 1000)
    ].copy()

    for c in [
        "active_prediction_mAHD",
        "aquifer_thickness_m",
        "hydraulic_conductivity_K",
    ]:
        coastal_df[c]=pd.to_numeric(coastal_df[c],errors="coerce")

    missing_mask=coastal_df[[
        "aquifer_thickness_m", "hydraulic_conductivity_K"
    ]].isna().any(axis=1)
    missing=int(missing_mask.sum())
    if missing > 0:
        message=(
            f"Warning: {missing} coastal wells missing subsurface data — "
            "SGD will be NaN for these, not zero."
        )
        print(message)
        try:
            st.warning(message)
        except Exception:
            pass

    # Avoid a zero-distance division while preserving the supplied Darcy-law
    # formulation. A minimum 1 m denominator prevents numerical blow-up at the
    # coastline; this does not fill missing predictions or subsurface data.
    coastal_df["hydraulic_gradient"]=(
        coastal_df["active_prediction_mAHD"]
        / coastal_df["distance_coast_m"].clip(lower=1)
    )

    coastal_df["cross_section_area_m2"]=(
        coastal_df["aquifer_thickness_m"] * float(grid_cell_width_m)
    )

    coastal_df["SGD_m3_per_day"]=(
        coastal_df["hydraulic_conductivity_K"]
        * coastal_df["cross_section_area_m2"]
        * coastal_df["hydraulic_gradient"]
    )

    total_coastal_sgd=float(coastal_df["SGD_m3_per_day"].sum(skipna=True))
    return coastal_df, total_coastal_sgd


def grid_surface(df,col,grid_n=34):
    """Create a cell-based IDW surface for a solid geographic map overlay."""
    q=df[["longitude","latitude",col]].dropna().copy()
    if len(q)<6 or q[col].nunique()<2:return None
    minx,miny,maxx,maxy=AOI_POLY.bounds if AOI_POLY is not None else (q.longitude.min(),q.latitude.min(),q.longitude.max(),q.latitude.max())
    xs=np.linspace(minx,maxx,grid_n+1); ys=np.linspace(miny,maxy,grid_n+1)
    centers=[]; values=[]
    pxv=q.longitude.to_numpy(float); pyv=q.latitude.to_numpy(float); zv=q[col].to_numpy(float)
    for iy in range(grid_n):
        for ix in range(grid_n):
            a=(xs[ix]+xs[ix+1])/2; b=(ys[iy]+ys[iy+1])/2
            dd=np.sqrt(((pxv-a)*np.cos(np.deg2rad((pyv+b)/2)))**2+(pyv-b)**2)+1e-9
            w=1/(dd**2); values.append(float(np.sum(w*zv)/np.sum(w))); centers.append((ix,iy))
    return xs,ys,np.asarray(values,float),centers

def draw_surface(fig,df,col):
    surf=grid_surface(df,col)
    if surf is None:return
    xs,ys,vals,cells=surf
    features=[]; z=[]
    # Clip cells visually by using only cells whose center falls in the AOI.
    for v,(ix,iy) in zip(vals,cells):
        cx=(xs[ix]+xs[ix+1])/2; cy=(ys[iy]+ys[iy+1])/2
        if AOI_POLY is not None and not point_inside_boundary(cx,cy): continue
        cell=Polygon([[xs[ix],ys[iy]],[xs[ix+1],ys[iy]],[xs[ix+1],ys[iy+1]],[xs[ix],ys[iy+1]]])
        if AOI_POLY is not None:
            cell=cell.intersection(AOI_POLY)
        if cell.is_empty: continue
        geoms=list(cell.geoms) if cell.geom_type=="MultiPolygon" else [cell]
        for gg in geoms:
            ring=[list(xy) for xy in gg.exterior.coords]
            features.append({"id":str(len(features)),"type":"Feature","properties":{"z":float(v)},"geometry":{"type":"Polygon","coordinates":[ring]}})
            z.append(float(v))
    if not features:return
    gj={"type":"FeatureCollection","features":features}
    locations=[f["id"] for f in features]
    if hasattr(go,"Choroplethmap"):
        fig.add_trace(go.Choroplethmap(geojson=gj,locations=locations,z=z,featureidkey="id",colorscale=[[0,"#0b5963"],[.45,"#35b7aa"],[.72,"#9cd8c8"],[1,"#d0b45e"]],marker=dict(line=dict(width=0),opacity=.34),showscale=False,hoverinfo="skip",name="Piezometric surface"))
    else:
        fig.add_trace(go.Choroplethmapbox(geojson=gj,locations=locations,z=z,featureidkey="id",colorscale=[[0,"#0b5963"],[.45,"#35b7aa"],[.72,"#9cd8c8"],[1,"#d0b45e"]],marker=dict(line=dict(width=0),opacity=.34),showscale=False,hoverinfo="skip",name="Piezometric surface"))


# ============================================================
# MAP HELPERS
# ============================================================

def teal_template(fig):
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#ffffff",font=dict(color="#17363a",family="Inter,Arial"),margin=dict(l=8,r=8,t=48,b=8),legend=dict(orientation="h",y=1.08,x=0))
    fig.update_xaxes(showgrid=True,gridcolor="#e7f0ee"); fig.update_yaxes(showgrid=True,gridcolor="#e7f0ee"); return fig


def render_sgd_heatmap(coastal_df, total_sgd):
    """Render coastal submarine groundwater discharge hotspots on a MapLibre map."""
    st.subheader("Coastal Submarine Groundwater Discharge (SGD) Hotspots")

    if coastal_df is None or coastal_df.empty:
        st.info("No coastal cells/wells within 1,000 m of the coastline have calculable SGD values.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Estimated Coastal SGD Discharge", f"{total_sgd:,.2f} m³/day")
    with col2:
        max_flux = pd.to_numeric(coastal_df["SGD_m3_per_day"], errors="coerce").max()
        st.metric("Peak Discharge Hotspot Rate", f"{max_flux:,.2f} m³/day/cell" if pd.notna(max_flux) else "—")

    plot_df = coastal_df.dropna(subset=["latitude", "longitude", "SGD_m3_per_day"]).copy()
    if plot_df.empty:
        st.warning("Coastal SGD was calculated, but there are no finite values available to map.")
        return

    color_scale=[[0,"#075a67"],[.48,"#37b9ae"],[1,"#d3b355"]]
    hover_data={
        "latitude": False, "longitude": False,
        "active_prediction_mAHD": ":.2f",
        "aquifer_thickness_m": ":.1f",
        "hydraulic_gradient": ":.4f",
        "SGD_m3_per_day": ":.2f",
    }

    # Plotly >=5.24 provides the forward-compatible MapLibre density_map trace.
    # Fall back only when the installed Plotly version does not expose it.
    try:
        fig = px.density_map(
            plot_df, lat="latitude", lon="longitude", z="SGD_m3_per_day",
            radius=18, center=dict(lat=-34.62, lon=135.47), zoom=11,
            map_style="open-street-map",
            color_continuous_scale=color_scale,
            labels={"SGD_m3_per_day": "Discharge Flux (m³/day)"},
            hover_data=hover_data,
        )
    except (AttributeError, TypeError):
        # Compatibility fallback for older Plotly installations.
        fig = px.density_mapbox(
            plot_df, lat="latitude", lon="longitude", z="SGD_m3_per_day",
            radius=18, center=dict(lat=-34.62, lon=135.47), zoom=11,
            mapbox_style="open-street-map",
            color_continuous_scale=color_scale,
            labels={"SGD_m3_per_day": "Discharge Flux (m³/day)"},
            hover_data=hover_data,
        )

    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    st.plotly_chart(teal_template(fig), use_container_width=True)

    if st.session_state.dataset == "Use demo data":
        st.info(
            "This is a synthetic-data prototype unless running on uploaded/live observations — "
            "do not present SGD figures from demo data as real Coffin Bay discharge estimates.",
            icon="⚠️"
        )
    else:
        st.info(
            "SGD is estimated from the active ML hydraulic-head prediction and the available "
            "observed/interpolated subsurface properties. Treat these values as model-derived estimates, not direct discharge measurements.",
            icon="ℹ️"
        )


def add_boundary_trace(fig):
    if BOUNDARY is None:return
    for geom in BOUNDARY.geometry:
        geoms = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
        for g in geoms:
            x,y = g.exterior.xy
            fig.add_trace(go.Scattermap(lat=np.asarray(y),lon=np.asarray(x),mode="lines",line=dict(width=3,color="#0b767b"),name="Rizin study boundary",hoverinfo="skip"))


def add_coastline_trace(fig, coastline_gdf):
    if coastline_gdf is None or coastline_gdf.empty:return
    for geom in coastline_gdf.geometry:
        geoms = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
        for g in geoms:
            x,y = g.xy
            fig.add_trace(go.Scattermap(lat=np.asarray(y),lon=np.asarray(x),mode="lines",line=dict(width=3,color="#ef8354"),name="DEA coastline",hoverinfo="skip",showlegend=False))


def make_map(df,value_col,title,height=710,center=None,zoom=9.2,show_anchors=True,coastline_gdf=None,show_coastline=True):
    if df.empty:return go.Figure()
    center=center or {"lat":float(df.latitude.median()),"lon":float(df.longitude.median())}
    hover_cols=[c for c in ["dem_m","geology_formation","distance_coast_m","distance_lake_wangary_m","groundwater_level_mAHD",value_col,"year","season"] if c in df.columns]
    if hasattr(px,"scatter_map"):
        fig=px.scatter_map(df,lat="latitude",lon="longitude",color=value_col,hover_name="well_id",hover_data=hover_cols,color_continuous_scale=[[0,"#075a67"],[.48,"#37b9ae"],[1,"#d3b355"]],zoom=zoom,height=height,center=center,opacity=.90,size_max=11)
        fig.update_layout(map_style="open-street-map")
        fig.update_traces(customdata=np.arange(len(df)))
        if show_coastline:add_coastline_trace(fig,coastline_gdf)
        add_boundary_trace(fig)
        if show_anchors:
            fig.add_trace(go.Scattermap(lat=[LAKE_WANGARY["latitude"]],lon=[LAKE_WANGARY["longitude"]],mode="markers+text",marker=dict(size=15,color="#19786f",symbol="diamond"),text=[f"LAKE WANGARY · {LAKE_WANGARY['level_mAHD']:.1f} m AHD"],textposition="top center",textfont=dict(size=11,color="#124d52"),name="Lake Wangary anchor",hovertext=[f"Surface-water anchor · {LAKE_WANGARY['level_mAHD']:.2f} m AHD"],hoverinfo="text"))
            cp=coast_label_point(coastline_gdf,center["lon"],center["lat"])
            if cp is not None:
                fig.add_trace(go.Scattermap(lat=[float(cp.y)],lon=[float(cp.x)],mode="markers+text",marker=dict(size=10,color="#ef8354",symbol="circle"),text=[f"DEA COAST · {COAST_ANCHOR['level_mAHD']:.1f} m AHD"],textposition="bottom center",textfont=dict(size=10,color="#9a5b3d"),name="Coastal datum anchor",hovertext=[f"DEA Coastlines shoreline · approximately 0 m mean sea level · analysis datum {COAST_ANCHOR['level_mAHD']:.2f} m AHD"],hoverinfo="text"))
    else:
        fig=px.scatter_mapbox(df,lat="latitude",lon="longitude",color=value_col,hover_name="well_id",hover_data=hover_cols,color_continuous_scale=[[0,"#075a67"],[.48,"#37b9ae"],[1,"#d3b355"]],zoom=zoom,height=height,center=center,opacity=.90,size_max=11)
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_traces(customdata=np.arange(len(df)))
        if show_coastline:add_coastline_trace(fig,coastline_gdf)
        add_boundary_trace(fig)
    fig.update_layout(margin=dict(l=0,r=0,t=28,b=0),paper_bgcolor="rgba(0,0,0,0)",legend=dict(orientation="h",y=1.02,x=.01),coloraxis_colorbar=dict(title="m AHD"))
    return fig


def load_well_points_upload(uploaded_file):
    """Read a point shapefile ZIP / GeoJSON / GeoPackage and return point geometries."""
    if uploaded_file is None:
        return None, None
    name=str(getattr(uploaded_file,"name","uploaded" )).lower()
    try:
        if name.endswith(".zip"):
            tmp=DATA_DIR/"_well_upload"
            tmp.mkdir(exist_ok=True)
            zpath=tmp/"wells.zip"
            zpath.write_bytes(uploaded_file.getvalue())
            with zipfile.ZipFile(zpath) as z:
                z.extractall(tmp)
            shp=next(tmp.rglob("*.shp"),None)
            if shp is None:
                return None,"ZIP does not contain a .shp file."
            gdf=gpd.read_file(shp)
        else:
            tmp=DATA_DIR/"_well_upload_source"
            tmp.mkdir(exist_ok=True)
            ext=Path(name).suffix or ".geojson"
            path=tmp/('wells'+ext)
            path.write_bytes(uploaded_file.getvalue())
            gdf=gpd.read_file(path)
        if gdf.empty:
            return None,"Well-point layer is empty."
        gdf=gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
        gdf=gdf[gdf.geometry.geom_type.isin(["Point"])].copy()
        if gdf.empty:
            return None,"The uploaded layer contains no Point geometries."
        if gdf.crs is None:
            # Only accept missing CRS when coordinates clearly look geographic.
            if not gdf.geometry.x.between(-180,180).all() or not gdf.geometry.y.between(-90,90).all():
                return None,"Well-point layer has no CRS and the coordinates are not clearly geographic. Define the CRS before uploading."
            gdf=gdf.set_crs(4326)
        else:
            gdf=gdf.to_crs(4326)
        if AOI_POLY is not None:
            inside=gdf.geometry.apply(lambda q: AOI_POLY.contains(q) or AOI_POLY.touches(q))
            gdf=gdf[inside].copy()
        gdf=gdf.drop_duplicates(subset=["geometry"]).reset_index(drop=True)
        if gdf.empty:
            return None,"No uploaded well points fall inside the Rizin AOI."
        return gdf, f"Loaded {len(gdf):,} well points from {getattr(uploaded_file,'name','upload')}."
    except Exception as exc:
        return None,f"Well-point layer could not be read: {exc}"

def normalise_subsurface_input(df):
    """Normalise a bore-log-only table without requiring groundwater observations."""
    d=df.copy()
    aliases={
        "well_id":["well_id","well","site","id","bore_id"],
        "longitude":["longitude","lon","x_lon"],"latitude":["latitude","lat","y_lat"],
        "depth_to_basement_m":["depth_to_basement_m","basement_depth_m","depth_basement_m"],
        "aquifer_thickness_m":["aquifer_thickness_m","aquifer_thickness","aquifer_thick_m"],
        "clay_layer_total_m":["clay_layer_total_m","total_clay_m","clay_thickness_m"],
        "hydraulic_conductivity_K":["hydraulic_conductivity_K","hydraulic_conductivity","k","conductivity_K"],
        "specific_yield_Sy":["specific_yield_Sy","specific_yield","sy","specific_yield_fraction"],
        "is_confined":["is_confined","confined","confined_aquifer"],
    }
    lower={str(c).strip().lower():c for c in d.columns}
    for target,names in aliases.items():
        if target not in d.columns:
            found=next((lower.get(n.lower()) for n in names if n.lower() in lower),None)
            if found is not None:d[target]=d[found]
    for c in ["longitude","latitude"]+SUBSURFACE_COLUMNS:
        if c not in d.columns:d[c]=np.nan
        d[c]=pd.to_numeric(d[c],errors="coerce")
    if "is_confined" in d:
        d["is_confined"]=d["is_confined"].where(d["is_confined"].isna(),d["is_confined"].round().clip(0,1))
    d["well_id"]=d.get("well_id",pd.Series([f"BORE_{i:05d}" for i in range(1,len(d)+1)],index=d.index))
    return d.dropna(subset=["well_id","longitude","latitude"]).copy()

def sheet_url_to_csv(share_url):
    """Convert a Google Sheets share URL into a public CSV/Google Visualization URL."""
    if not share_url or not isinstance(share_url, str):
        return None
    match = re.search(r"/spreadsheets/d/([a-zA-Z0-9_-]+)", share_url.strip())
    if not match:
        return None
    sheet_id = match.group(1)
    gid_match = re.search(r"(?:[#?&]gid=)(\d+)", share_url)
    gid = gid_match.group(1) if gid_match else "0"
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&gid={gid}"


@st.cache_data(ttl=300, show_spinner=False)
def fetch_public_google_sheet(csv_url):
    """Fetch a Google Sheet shared for public viewing and return a DataFrame.

    The request deliberately uses Google's public CSV/Visualization endpoint, so
    no service-account credentials are required. Clear errors are raised for
    private/unpublished sheets and malformed responses.
    """
    import requests
    from io import StringIO

    headers = {"User-Agent": "CB-HYDRO-Streamlit/1.0"}
    response = requests.get(csv_url, headers=headers, timeout=20, allow_redirects=True)
    response.raise_for_status()

    content_type = (response.headers.get("content-type") or "").lower()
    text = response.text.lstrip("\ufeff").strip()

    # A private Google Sheet commonly redirects to an HTML login/error page.
    if "<html" in text[:500].lower() or "<!doctype" in text[:500].lower():
        raise PermissionError(
            "Google returned an HTML page instead of sheet data. "
            "Set Share → General access → Anyone with the link → Viewer."
        )

    if not text:
        raise ValueError("The Google Sheet returned an empty response.")

    # CSV parsing is more reliable here than trying to use gspread without credentials.
    frame = pd.read_csv(StringIO(text))
    if frame.empty:
        raise ValueError("The Google Sheet contains no data rows.")

    return frame


# ============================================================
# STATE / SIDEBAR
# ============================================================
if "dataset" not in st.session_state: st.session_state.dataset="Use demo data"
if "uploaded_df" not in st.session_state: st.session_state.uploaded_df=None
if "live_sheet_df" not in st.session_state: st.session_state.live_sheet_df=None
if "live_sheet_url" not in st.session_state: st.session_state.live_sheet_url=""
if "uploaded_borelog_df" not in st.session_state: st.session_state.uploaded_borelog_df=None
if "active_model" not in st.session_state: st.session_state.active_model=None
if "models_loaded" not in st.session_state: st.session_state.models_loaded=False
if "coastline" not in st.session_state: st.session_state.coastline=None
if "coastline_status" not in st.session_state: st.session_state.coastline_status="Not loaded"
if "coast_year" not in st.session_state: st.session_state.coast_year=2024
if "loaded_coast_year" not in st.session_state: st.session_state.loaded_coast_year=None
if "uploaded_well_points" not in st.session_state: st.session_state.uploaded_well_points=None
if "well_points_status" not in st.session_state: st.session_state.well_points_status="No well-point layer uploaded."
if "spatial_selection_ids" not in st.session_state: st.session_state.spatial_selection_ids=[]

with st.sidebar:
    st.markdown("# CB / HYDRO")
    st.caption("Coffin Bay physical-geography intelligence workspace")
    view=st.radio("Workspace",["Overview","Piezometric map","SGD hotspots","Model lab","Model drivers","Well explorer","Diagnostics","Scenario lab","Data & export"],label_visibility="collapsed")
    st.markdown("### Data source")
    source_options=["Upload CSV","Use demo data","Live Google Sheet Sync"]
    current_index=source_options.index(st.session_state.dataset) if st.session_state.dataset in source_options else 1
    mode=st.radio("Source",source_options,index=current_index)
    if mode=="Upload CSV":
        up=st.file_uploader("Upload well observations",type=["csv"],help="CSV should contain longitude, latitude and groundwater level; year/well_id are recommended for temporal modelling.")
        if up is not None:
            try:
                incoming=normalise_columns(pd.read_csv(up))
                incoming=apply_dem_open_meteo_fallback(incoming)
                st.session_state.uploaded_df=incoming
                st.session_state.dataset="Upload CSV"
                st.session_state.models_loaded=False
                st.session_state.active_model=None
            except Exception as exc: st.error(f"CSV could not be read: {exc}")
        borelog_up=st.file_uploader(
            "Upload bore-log / subsurface CSV (optional)",
            type=["csv"],
            help="Use only genuine bore-log measurements. Required: well_id, longitude, latitude and any of the six static subsurface fields."
        )
        if borelog_up is not None:
            try:
                bore=normalise_subsurface_input(pd.read_csv(borelog_up))
                st.session_state.uploaded_borelog_df=bore
                st.session_state.models_loaded=False
                st.session_state.active_model=None
                st.caption(f"Loaded {len(bore):,} bore-log rows; only these rows can seed subsurface interpolation.")
            except Exception as exc: st.error(f"Bore-log CSV could not be read: {exc}")
    elif mode=="Live Google Sheet Sync":
        sheet_url=st.text_input(
            "Google Sheets share link",
            value=st.session_state.get("live_sheet_url", ""),
            help="Open your sheet → Share → General access → Anyone with the link → Viewer. Paste the full share URL here."
        ).strip()

        previous_url=st.session_state.get("live_sheet_url", "")
        st.session_state.live_sheet_url=sheet_url

        # Fetch only when the URL changes or no dataset has been loaded yet.
        # This prevents unnecessary requests on unrelated Streamlit reruns.
        should_fetch = bool(sheet_url) and (
            sheet_url != previous_url or st.session_state.get("live_sheet_df") is None
        )

        if not sheet_url:
            st.session_state.live_sheet_df=None
            st.session_state.live_sheet_error=None
            st.info("Paste a Google Sheets share link to load the live dataset.")

        elif should_fetch:
            csv_url=sheet_url_to_csv(sheet_url)
            if csv_url is None:
                st.session_state.live_sheet_df=None
                st.session_state.live_sheet_error="Couldn't parse that as a Google Sheets link. Paste the full URL from the browser."
                st.error(st.session_state.live_sheet_error)
            else:
                try:
                    incoming=normalise_columns(fetch_public_google_sheet(csv_url))
                    incoming=apply_dem_open_meteo_fallback(incoming)
                    if incoming.empty:
                        raise ValueError("The sheet contains no data rows.")

                    # Persist the authoritative fetched dataset across reruns.
                    st.session_state.live_sheet_df=incoming
                    st.session_state.live_sheet_error=None
                    st.session_state.live_sheet_loaded_url=sheet_url
                    st.session_state.models_loaded=False
                    st.session_state.active_model=None
                    st.success(f"DATA · LOADED — {len(incoming):,} rows from Google Sheets.")
                except PermissionError as exc:
                    st.session_state.live_sheet_df=None
                    st.session_state.live_sheet_error=str(exc)
                    st.error("Google Sheet access denied. Set the sheet to “Anyone with the link → Viewer”.")
                    st.caption(str(exc))
                except requests.RequestException as exc:
                    st.session_state.live_sheet_df=None
                    st.session_state.live_sheet_error=f"Google Sheets request failed: {exc}"
                    st.error("Could not reach Google Sheets. Check the link and your internet connection.")
                except Exception as exc:
                    st.session_state.live_sheet_df=None
                    st.session_state.live_sheet_error=str(exc)
                    st.error(f"Could not load the Google Sheet: {exc}")
        else:
            # The data is already in session_state, so show a persistent loaded state.
            if st.session_state.get("live_sheet_df") is not None:
                st.success(f"DATA · LOADED — {len(st.session_state.live_sheet_df):,} rows from Google Sheets.")
            elif st.session_state.get("live_sheet_error"):
                st.error(st.session_state.live_sheet_error)
    else:
        st.session_state.dataset="Use demo data"
        st.markdown("### Synthetic training dataset")
        st.session_state["synthetic_years"]=5
        st.session_state["synthetic_horizon_label"]="5 years · monthly time series"
        st.caption("Fixed demo dataset: 1,200 well points × 60 monthly observations = 72,000 synthetic records (2021–2025). Scenario data only; not observed measurements.")
    st.markdown("### Well-point geometry")
    wp_upload=st.file_uploader("Upload well-point layer (ZIP SHP / GeoJSON)",type=["zip","geojson","json","gpkg"],help="For the demo, supplied points anchor the synthetic wells. The selected horizon repeats each well through time so temporal models can learn seasonal, recurring, or long-term patterns.")
    if wp_upload is not None and st.session_state.get("well_points_filename") != getattr(wp_upload,"name",""):
        wp,status=load_well_points_upload(wp_upload)
        st.session_state.uploaded_well_points=wp
        st.session_state.well_points_status=status
        st.session_state.well_points_filename=getattr(wp_upload,"name","")
        st.session_state.models_loaded=False
        st.session_state.active_model=None
    st.session_state.dataset=mode
    if st.session_state.uploaded_well_points is not None:
        st.caption(st.session_state.well_points_status)
        if len(st.session_state.uploaded_well_points) != 1200:
            st.warning(f"Well-point layer contains {len(st.session_state.uploaded_well_points):,} points. The demo will use that point count.")

well_points = st.session_state.uploaded_well_points if st.session_state.dataset=="Use demo data" else None
if st.session_state.dataset=="Upload CSV" and st.session_state.uploaded_df is None:
    st.markdown('<div class="hero"><div class="hero-title">CB / Groundwater Intelligence</div><div class="hero-sub">Upload the groundwater observation CSV to activate the analysis workspace.</div><span class="chip">DATA · WAITING FOR UPLOAD</span></div>',unsafe_allow_html=True)
    st.info("No groundwater observations are loaded yet. Use the sidebar upload control, or switch the data source to another source.")
    st.stop()
if st.session_state.dataset=="Live Google Sheet Sync" and st.session_state.live_sheet_df is None:
    st.markdown('<div class="hero"><div class="hero-title">CB / Groundwater Intelligence</div><div class="hero-sub">Connect a public Google Sheet to activate the analysis workspace.</div><span class="chip">DATA · WAITING FOR GOOGLE SHEET</span></div>',unsafe_allow_html=True)
    st.info("No live Google Sheet dataset is loaded yet. Paste a full share link in the sidebar and ensure the sheet is shared as 'Anyone with the link → Viewer'.")
    st.stop()
if st.session_state.dataset in ["Upload CSV","Live Google Sheet Sync"]:
    # Real external observations remain authoritative. Apply the same DEM fallback
    # used by Upload CSV, filling only missing DEM values and never overwriting supplied values.
    if st.session_state.dataset=="Upload CSV":
        base=apply_dem_open_meteo_fallback(st.session_state.uploaded_df)
        st.session_state.uploaded_df=base
    else:
        base=apply_dem_open_meteo_fallback(st.session_state.live_sheet_df)
        st.session_state.live_sheet_df=base
else:
    # Always load the complete five-year synthetic monthly time series on startup.
    # 1,200 fixed spatial well locations are repeated through 60 months.
    base=make_data(n_wells=1200,years=5,well_points=well_points)
if "geology_factor" not in base.columns: base=base.copy(); base["geology_factor"]=0.0
for c in FEATURE_CANDIDATES:
    if c not in base.columns: base[c]=np.nan
if "distance_lake_wangary_m" not in base.columns:
    base["distance_lake_wangary_m"]=np.sqrt(((base["longitude"]-LAKE_WANGARY["longitude"])/0.0058)**2 + ((base["latitude"]-LAKE_WANGARY["latitude"])/0.0048)**2)*1000

available_models=["Random Forest","GAM"]
if XGB_OK: available_models.append("XGBoost")
if TORCH_OK: available_models.append("LSTM")
with st.sidebar:
    st.markdown("### Modelling")
    if view=="Model lab":
        st.caption("Choose models on this page, train them, then load one as the active layer.")
    default_models=[m for m in available_models if m!="LSTM"] if st.session_state.dataset=="Use demo data" else available_models
    selected=st.multiselect("Models to train / compare",available_models,default=default_models)
    st.markdown("### Coastline reference")
    coast_year=st.selectbox("DEA annual shoreline",list(range(1988,2026)),index=list(range(1988,2026)).index(st.session_state.get("coast_year",2024)))
    st.session_state["coast_year"]=int(coast_year)
    show_coastline=st.checkbox("Show DEA coastline",value=True)
    st.caption("Well-to-coast distance is calculated from the selected annual DEA shoreline clipped to Rizin. 2025 is interim.")
    st.markdown("### Hydrologic controls")
    coastal_anchor=st.number_input("Coastal boundary · m AHD",value=0.0,step=.1,format="%.1f")
    lake_selected=st.checkbox("Use Lake Wangary anchor",value=True)
    lake_level=st.number_input("Lake Wangary · m AHD",value=2.5,step=.1,format="%.1f",disabled=not lake_selected)
    LAKE_WANGARY["level_mAHD"] = float(lake_level)
    COAST_ANCHOR["level_mAHD"] = float(coastal_anchor)
    st.caption("Datum anchors are used by the conceptual surface preview; validate surveyed levels before scientific use.")
    st.markdown("### Filters")
    geos=st.multiselect("Geology",sorted(base.geology_formation.dropna().astype(str).unique()),sorted(base.geology_formation.dropna().astype(str).unique()))
    seasons=st.multiselect("Season",sorted(base.season.dropna().astype(str).unique()),sorted(base.season.dropna().astype(str).unique()))
    yr_numeric=pd.to_numeric(base.year,errors="coerce")
    ymin,ymax=int(yr_numeric.min()),int(yr_numeric.max()); years=st.slider("Observation year",ymin,ymax,(ymin,ymax))

# Fetch and clip the authoritative DEA annual shoreline to Rizin.
if BOUNDARY is not None:
    if st.session_state.get("loaded_coast_year") != st.session_state.get("coast_year",2024):
        coast, status = load_dea_coastline(BOUNDARY.to_json(), st.session_state.get("coast_year",2024))
        st.session_state["coastline"] = coast
        st.session_state["coastline_status"] = status
        st.session_state["loaded_coast_year"] = st.session_state.get("coast_year",2024)
    coastline_gdf=st.session_state.get("coastline")
else:
    coastline_gdf=None
base=attach_coast_distance(base, coastline_gdf)
if coastline_gdf is not None:
    base["coastline_year"]=st.session_state.get("coast_year",2024)
    base["distance_coast_method"]=f"DEA Coastlines {st.session_state.get('coast_year',2024)} · shortest distance in EPSG:28353"
if st.session_state.dataset=="Use demo data" and coastline_gdf is not None:
    base=refresh_synthetic_target(base)

# Static subsurface structure is merged once per well_id, before model training.
# In real-data mode, ONLY the dedicated bore-log upload is allowed to seed
# interpolation; demo/synthetic values are never used as interpolation inputs.
known_borelogs=st.session_state.get("uploaded_borelog_df")
if known_borelogs is not None and not known_borelogs.empty:
    base=merge_subsurface_features(base,known_boreholes=known_borelogs,search_radius_m=5000.0)
elif st.session_state.dataset=="Use demo data":
    for c in SUBSURFACE_COLUMNS:
        if c not in base.columns: base[c]=np.nan
        base[f"{c}_source"]="synthetic"
else:
    base=merge_subsurface_features(base,known_boreholes=None)

if not selected:
    st.warning("Select at least one model. Open Model lab to train and load an active model."); st.stop()

# Train only after the user explicitly asks for model training on Model lab, or on first page load.
# Training is explicit in Model Lab. This keeps the spatial/temporal demo responsive
# on startup while still allowing real uploaded observations to train the models.
if "trained_signature" not in st.session_state: st.session_state.trained_signature=None
trained_sig=st.session_state.get("trained_signature")
if trained_sig == (st.session_state.dataset, tuple(selected), len(base), int(base["well_id"].nunique()), int(len(known_borelogs) if known_borelogs is not None else 0)) and st.session_state.get("trained_bundle") is not None:
    d0,X,y,f,results,preds,imps,models,comparison=st.session_state.trained_bundle
else:
    d0,X,y,f=prepare_features(base)
    results={}; preds={}; imps={}; models={}; comparison=pd.DataFrame()
valid_models=[name for name in selected if name in comparison.Model.tolist()] if not comparison.empty else []
if st.session_state.active_model not in valid_models:
    st.session_state.active_model = None
active=st.session_state.active_model

res=d0.copy()
for name,p in preds.items(): res[f"{name}_predicted_mAHD"]=p
# Keep one authoritative prediction column for downstream analyses. For the
# sklearn-style active models, refresh it directly from model.predict(X) so the
# SGD workflow uses exactly the same prediction convention as the active model.
model=models.get(active) if active else None
if active and active != "LSTM" and model is not None:
    res["active_prediction_mAHD"]=model.predict(X)
else:
    res["active_prediction_mAHD"]=res.get(f"{active}_predicted_mAHD",np.nan)
res["active_residual_m"]=res["groundwater_level_mAHD"]-res["active_prediction_mAHD"]
res["active_abs_error_m"]=res["active_residual_m"].abs()
d=res[res.geology_formation.astype(str).isin(geos) & res.season.astype(str).isin(seasons) & res.year.between(years[0],years[1])].copy()

data_label = "UPLOAD" if st.session_state.dataset=="Upload CSV" else ("LIVE SHEET" if st.session_state.dataset=="Live Google Sheet Sync" else "DEMO")
active_label = active.upper().replace(" ", " · ") if active else "NOT LOADED"
status_chip = f"DATA · {data_label} &nbsp;|&nbsp; ACTIVE MODEL · {active.upper()}" if active else f"DATA · {data_label} &nbsp;|&nbsp; MODEL · NOT LOADED"
st.markdown(f'<div class="hero"><div class="hero-title">CB / Groundwater Intelligence</div><div class="hero-sub">Physical-geography workspace · Rizin AOI · DEA shoreline distance · surface-water anchors · spatial extraction</div><span class="chip">{status_chip}</span></div>',unsafe_allow_html=True)

# ============================================================
# PAGES
# ============================================================
if view=="Overview":
    st.markdown('<div class="section">Hydrologic context</div>',unsafe_allow_html=True)
    a,b,c,e=st.columns(4); a.metric("Wells",f"{d.well_id.nunique():,}"); b.metric("Observations",f"{len(d):,}"); c.metric("DEA coastline",str(st.session_state.get("coast_year")) if coastline_gdf is not None else "Unavailable"); e.metric("Lake Wangary",f"{lake_level:.1f} m AHD" if lake_selected else "Off")
    if st.session_state.dataset=="Use demo data":
        st.caption(f"Synthetic training series: 1,200 fixed wells × 60 monthly observations · {len(d):,} records · 2021–2025. This is scenario data for modelling demonstrations, not observed groundwater measurements.")
    elif d.well_id.nunique()!=len(d):
        label="Live Google Sheet" if st.session_state.dataset=="Live Google Sheet Sync" else "Uploaded"
        st.caption(f"{label} temporal dataset: {d.well_id.nunique():,} unique wells across {len(d):,} observations.")
    if coastline_gdf is not None:
        st.markdown(f'<div class="hydro-good"><b>Distance-to-coast engine:</b> shortest perpendicular distance from each well to the clipped DEA annual shoreline for <b>{st.session_state.get("coast_year")}</b>, calculated in EPSG:28353 metres.</div>',unsafe_allow_html=True)
    else:
        st.warning(st.session_state.get("coastline_status","DEA coastline not loaded"))
    st.markdown('<div class="panel panel-accent"><b>Conceptual control points</b><div class="small">The piezometric surface is interpreted relative to two explicit surface-water / datum anchors rather than as an unconstrained black-box prediction.</div><div class="anchor-card">COASTAL BOUNDARY · <b>0.0 m AHD</b> — near-coast hydraulic datum</div><div class="anchor-card">LAKE WANGARY · <b>2.5 m AHD</b> — selected surface-water level anchor</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="section">Hydrogeographic response</div>',unsafe_allow_html=True)
    if active:
        fig=px.scatter(d,x="dem_m",y="groundwater_level_mAHD",color="active_prediction_mAHD",hover_name="well_id",hover_data=["geology_formation","distance_coast_m","year","season"],color_continuous_scale=[[0,"#075a67"],[.48,"#37b9ae"],[1,"#d3b355"]],labels={"dem_m":"DEM elevation (m)","groundwater_level_mAHD":"Observed groundwater (m AHD)","active_prediction_mAHD":f"{active} prediction (m AHD)"})
        st.plotly_chart(teal_template(fig),use_container_width=True)
        st.plotly_chart(make_map(d,"active_prediction_mAHD",f"{active} · spatial prediction layer",coastline_gdf=coastline_gdf,show_coastline=show_coastline),use_container_width=True)
    else:
        st.info("No model is loaded. Train models in Model Lab and choose **Load as active model** to populate prediction layers.")
        st.plotly_chart(make_map(d,"groundwater_level_mAHD","Observed groundwater · hydrogeographic reference",coastline_gdf=coastline_gdf,show_coastline=show_coastline),use_container_width=True)
    st.markdown('<div class="map-caption"><span>Rizin boundary shown as the study-area frame.</span><span>Lake Wangary = 2.5 m AHD anchor · coastal datum = 0 m AHD.</span></div>',unsafe_allow_html=True)

elif view=="Piezometric map":
    st.markdown('<div class="section">Piezometric surface explorer</div>',unsafe_allow_html=True)
    map_options=["Observed groundwater"] if not active else [f"{active} predicted groundwater","Observed groundwater","Absolute prediction error"]
    c1,c2,c3,c4=st.columns([1.4,1,1,1]); variable=c1.selectbox("Map variable",map_options); col={f"{active} predicted groundwater":"active_prediction_mAHD","Observed groundwater":"groundwater_level_mAHD","Absolute prediction error":"active_abs_error_m"}.get(variable,"groundwater_level_mAHD"); selection_mode=c2.selectbox("Selection",["Lasso","Box","Point"]); layer=c3.checkbox("Piezometric preview",True); anchors=c4.checkbox("Hydrologic controls",True)
    fig=make_map(d,col,f"{variable} · select wells",show_anchors=anchors,coastline_gdf=coastline_gdf,show_coastline=show_coastline).to_plotly_json()
    # Convert JSON back to object to conditionally add the preview without duplicating base map logic.
    fig=go.Figure(fig)
    if layer: draw_surface(fig,d,col)
    event=st.plotly_chart(fig,use_container_width=True,key="piezometric_map",on_select="rerun",selection_mode={"Lasso":"lasso","Box":"box","Point":"points"}[selection_mode])
    ids=[]
    try:
        for pt in event.selection.points:
            cd=pt.get("customdata")
            if isinstance(cd,(list,tuple,np.ndarray)) and len(cd):
                ids.append(int(cd[0]))
            elif cd is not None and np.isscalar(cd):
                ids.append(int(cd))
            else:
                idx=pt.get("point_index",pt.get("pointNumber",None))
                curve=pt.get("curve_number",pt.get("curveNumber",None))
                if idx is not None and curve in (0,None): ids.append(int(idx))
    except Exception: pass
    ids=sorted(set(i for i in ids if 0<=i<len(d)))
    if ids: st.session_state.spatial_selection_ids=ids
    elif event.selection is not None and len(event.selection.points)==0: st.session_state.spatial_selection_ids=[]
    ids=st.session_state.get("spatial_selection_ids",[])
    selected_rows=d.iloc[ids].copy() if ids else d.iloc[0:0].copy()
    st.markdown('<div class="panel"><b>Spatial extraction</b><div class="small">Draw a lasso or box over the map to extract the well points contained by the selected area. The preview surface is a continuous cell overlay; blue dots are not synthetic wells.</div></div>',unsafe_allow_html=True)
    a,b=st.columns([1,2]); a.metric("Selected wells",f"{len(selected_rows):,}")
    if len(selected_rows):
        b.download_button("Export selected wells · CSV",selected_rows.to_csv(index=False).encode(),"coffin_bay_selected_wells.csv","text/csv",use_container_width=True); st.dataframe(selected_rows,use_container_width=True,hide_index=True)
    else: st.caption("No spatial selection yet. Use Lasso, Box or Point above the map.")

elif view=="SGD hotspots":
    st.markdown('<div class="section">Coastal submarine groundwater discharge</div>',unsafe_allow_html=True)
    if not active:
        st.info("Load an active model from Model Lab first. SGD requires ML-predicted hydraulic heads.")
        st.stop()

    st.markdown(
        '<div class="hydro-note"><b>Darcy-law SGD estimate:</b> coastal cells within 1 km of the DEA coastline are evaluated using the active ML-predicted hydraulic head, aquifer thickness, and hydraulic conductivity.</div>',
        unsafe_allow_html=True
    )
    sgd_width=st.number_input(
        "Grid-cell width (m)", min_value=1.0, value=50.0, step=5.0,
        help="Cross-shore cell width used to convert aquifer thickness into cross-sectional area."
    )
    df_predictions=res.copy()
    df_predictions["active_prediction_mAHD"]=res["active_prediction_mAHD"]
    try:
        coastal_df,total_sgd=calculate_sgd(df_predictions,grid_cell_width_m=sgd_width)
        render_sgd_heatmap(coastal_df,total_sgd)
        missing_flux=int(coastal_df["SGD_m3_per_day"].isna().sum()) if not coastal_df.empty else 0
        if missing_flux:
            st.warning(f"{missing_flux:,} coastal wells have NaN SGD because required subsurface inputs or predictions are missing.")
        if not coastal_df.empty:
            st.dataframe(
                coastal_df[[c for c in [
                    "well_id","latitude","longitude","distance_coast_m",
                    "active_prediction_mAHD","aquifer_thickness_m",
                    "hydraulic_conductivity_K","hydraulic_gradient",
                    "cross_section_area_m2","SGD_m3_per_day",
                    "aquifer_thickness_m_source","hydraulic_conductivity_K_source"
                ] if c in coastal_df.columns]],
                use_container_width=True, hide_index=True
            )
    except ValueError as exc:
        st.error(f"SGD calculation could not run: {exc}")

elif view=="Model lab":
    st.markdown('<div class="section">Model training & active-model selection</div>',unsafe_allow_html=True)
    st.markdown('<div class="panel panel-accent"><b>Train → compare → load</b><div class="small">Select candidate models, train them on the same holdout structure, compare RMSE / MAE / R², then explicitly load one model as the active prediction layer. All downstream pages follow the active model selected here.</div></div>',unsafe_allow_html=True)
    available_results=sorted(comparison.Model.tolist()) if not comparison.empty else []
    if st.button("Train selected models on current data",use_container_width=True,type="primary"):
        with st.spinner("Training models on the current groundwater time series…"):
            bundle=train_models(base,tuple(selected))
        st.session_state.trained_bundle=bundle
        st.session_state.trained_signature=(st.session_state.dataset, tuple(selected), len(base), int(base["well_id"].nunique()), int(len(known_borelogs) if known_borelogs is not None else 0))
        st.session_state.models_loaded=True
        st.rerun()
    if st.session_state.dataset=="Use demo data":
        st.info("Demo training set: 1,200 fixed wells × 60 monthly observations (5 years). For a real study, switch to Upload CSV or Live Google Sheet Sync and train on the external observations.")
    c1,c2=st.columns([1.2,1.8])
    with c1:
        st.markdown('#### Active model')
        desired=st.selectbox("Load model",available_results or selected,index=(available_results.index(active) if active in available_results else 0),key="model_loader")
        if st.button("Load as active model",use_container_width=True,type="primary"):
            st.session_state.active_model=desired
            active=desired
            st.rerun()
        st.info(f"Currently loaded: {st.session_state.active_model or 'none'}")
    with c2:
        if not comparison.empty:
            champion=comparison.iloc[0]
            st.markdown('#### Benchmark leader')
            a,b,c,d4=st.columns(4); a.metric("Best model",champion["Model"]); b.metric("Best RMSE",f"{champion['RMSE (m)']:.3f} m"); c.metric("Best R²",f"{champion['R²']:.3f}"); d4.metric("Top feature",champion["Top feature"])
            wording='uploaded dataset' if st.session_state.dataset=="Upload CSV" else 'demonstration dataset'
            st.markdown(f'<div class="hydro-good">Benchmark leader on the current {wording}: <b>{champion["Model"]}</b>. Leading predictive feature: <b>{champion["Top feature"]}</b>. The ranking is a model-comparison result, not causal evidence.</div>',unsafe_allow_html=True)
    if not comparison.empty:
        st.dataframe(comparison,use_container_width=True,hide_index=True)
        melted=comparison.melt(id_vars=["Model"],value_vars=["RMSE (m)","MAE (m)"],var_name="Metric",value_name="Value"); fig=px.bar(melted,x="Model",y="Value",color="Metric",barmode="group",title="Holdout error comparison"); st.plotly_chart(teal_template(fig),use_container_width=True)
        fig2=px.bar(comparison.sort_values("R²"),x="R²",y="Model",orientation="h",title="Holdout R² comparison"); st.plotly_chart(teal_template(fig2),use_container_width=True)
    for name,rr in results.items():
        if "Error" in rr: st.error(f"{name}: {rr['Error']}")

elif view=="Model drivers":
    st.markdown('<div class="section">Prediction drivers</div>',unsafe_allow_html=True)
    if not active:
        st.info("Load an active model from Model Lab first.")
        st.stop()
    imp=imps.get(active,pd.Series(dtype=float)); imp_df=imp.rename("Importance").reset_index().rename(columns={"index":"Feature"}); st.caption("Feature importance is a predictive diagnostic, not causal evidence.")
    fig=px.bar(imp_df.sort_values("Importance"),x="Importance",y="Feature",orientation="h",title=f"{active} feature importance"); st.plotly_chart(teal_template(fig),use_container_width=True)
    feat=st.selectbox("Inspect predictor",imp_df.Feature.tolist() if not imp_df.empty else f)
    scatter=px.scatter(res,x=feat,y="active_prediction_mAHD",color="geology_formation",hover_name="well_id",labels={"active_prediction_mAHD":f"{active} predicted groundwater (m AHD)"}); tmp=res[[feat,"active_prediction_mAHD"]].dropna()
    if len(tmp)>2 and tmp[feat].nunique()>1:
        slope,intercept=np.polyfit(tmp[feat].to_numpy(float),tmp["active_prediction_mAHD"].to_numpy(float),1); xl=np.linspace(tmp[feat].min(),tmp[feat].max(),50); scatter.add_trace(go.Scatter(x=xl,y=slope*xl+intercept,mode="lines",name="Linear guide"))
    st.plotly_chart(teal_template(scatter),use_container_width=True)

elif view=="Well explorer":
    st.markdown('<div class="section">Well-by-well inspection</div>',unsafe_allow_html=True)
    if not active:
        st.info("Load an active model from Model Lab first.")
        st.stop()
    w=st.selectbox("Well",d.well_id.astype(str).unique().tolist()); rd=d[d.well_id.astype(str)==w].sort_values("date"); r=rd.iloc[-1]
    a,b,c,e=st.columns(4); a.metric("Observed",f"{r.groundwater_level_mAHD:.2f} m AHD"); b.metric(f"{active} prediction",f"{r.active_prediction_mAHD:.2f} m AHD" if pd.notna(r.active_prediction_mAHD) else "—"); c.metric("Residual",f"{r.active_residual_m:.2f} m" if pd.notna(r.active_residual_m) else "—"); e.metric("DEM",f"{r.dem_m:.2f} m")
    st.dataframe(rd[["well_id","date","year","month","season","groundwater_level_mAHD","active_prediction_mAHD","active_residual_m","dem_m","distance_coast_m","distance_lake_wangary_m","geology_formation"]],use_container_width=True,hide_index=True)
    fig=px.line(rd,x="date",y=["groundwater_level_mAHD","active_prediction_mAHD"],markers=True,title=f"Observed vs {active} prediction — {w}"); st.plotly_chart(teal_template(fig),use_container_width=True)

elif view=="Diagnostics":
    st.markdown('<div class="section">Active model diagnostics</div>',unsafe_allow_html=True)
    if not active:
        st.info("Load an active model from Model Lab first.")
        st.stop()
    fig=px.scatter(res,x="groundwater_level_mAHD",y="active_prediction_mAHD",color="active_abs_error_m",hover_name="well_id",color_continuous_scale=[[0,"#075a67"],[.48,"#37b9ae"],[1,"#d3b355"]],labels={"groundwater_level_mAHD":"Observed (m AHD)","active_prediction_mAHD":f"{active} prediction (m AHD)"})
    lo=min(res.groundwater_level_mAHD.min(),res.active_prediction_mAHD.min()); hi=max(res.groundwater_level_mAHD.max(),res.active_prediction_mAHD.max()); fig.add_shape(type="line",x0=lo,y0=lo,x1=hi,y1=hi,line=dict(dash="dash",color="#557b7f")); st.plotly_chart(teal_template(fig),use_container_width=True)
    hist=px.histogram(res,x="active_residual_m",nbins=35,title="Residual distribution"); st.plotly_chart(teal_template(hist),use_container_width=True); st.dataframe(res.sort_values("active_abs_error_m",ascending=False)[["well_id","geology_formation","dem_m","distance_coast_m","groundwater_level_mAHD","active_prediction_mAHD","active_residual_m","active_abs_error_m"]].head(80),use_container_width=True,hide_index=True)

elif view=="Scenario lab":
    st.markdown('<div class="section">Hydrologic scenario lab</div>',unsafe_allow_html=True)
    if not active:
        st.info("Load an active model from Model Lab first.")
        st.stop()
    st.markdown('<div class="hydro-note">Scenarios are conceptual. The controls make the coastal and Lake Wangary anchors explicit so scenario changes are visible relative to a physical datum framework.</div>',unsafe_allow_html=True)
    rr=res.iloc[0].copy()
    def sval(obj,key,default):
        try:
            v=pd.to_numeric(obj.get(key,default),errors="coerce")
            return float(v) if pd.notna(v) else float(default)
        except Exception:
            return float(default)
    c1,c2,c3=st.columns(3)
    dem=c1.slider("DEM (m)",0.,40.,float(np.clip(sval(rr,"dem_m",10),0,40)),.5)
    rain=c2.slider("Rainfall (mm)",250,820,int(np.clip(sval(rr,"rainfall_mm",520),250,820)),10)
    nd=c3.slider("NDVI",.1,.95,float(np.clip(sval(rr,"ndvi_mean",.5),.1,.95)),.01)
    coast=c1.slider("Distance to coast (m)",20,16000,int(np.clip(sval(rr,"distance_coast_m",1000),20,16000)),100)
    et=c2.slider("ET (mm)",700,1400,int(np.clip(sval(rr,"et_mm",1050),700,1400)),10)
    sw=c3.slider("Surface-water distance (m)",20,11000,int(np.clip(sval(rr,"surface_water_distance_m",6000),20,11000)),100)
    row=rr.copy(); row.update({"dem_m":dem,"rainfall_mm":rain,"ndvi_mean":nd,"distance_coast_m":coast,"et_mm":et,"surface_water_distance_m":sw}); Xrow=pd.DataFrame([{q:(row[q] if pd.notna(row[q]) else res[q].median()) for q in f}]); mdl=models.get(active)
    if active!="LSTM": pred=float(mdl.predict(Xrow)[0]); st.metric("Scenario prediction",f"{pred:.2f} m AHD")
    else: st.info("The LSTM is sequence-based; use Well explorer or Model lab for temporal predictions and benchmark results.")

else:
    st.markdown('<div class="section">Data, spatial selection & export</div>',unsafe_allow_html=True)
    st.markdown('<div class="panel">The recommended extraction workflow is <b>Piezometric map → Lasso / Box → Export selected wells</b>. The current filters can also be exported here.</div>',unsafe_allow_html=True)
    a,b,c=st.columns(3); a.metric("Wells",f"{d.well_id.nunique():,}"); b.metric("Observations",f"{len(d):,}"); c.metric("Coast distance",("DEA shoreline" if coastline_gdf is not None else "Unavailable"))
    st.download_button("Export current filtered dataset · CSV",d.to_csv(index=False).encode(),"coffin_bay_filtered_wells.csv","text/csv",use_container_width=True); st.dataframe(d.head(700),use_container_width=True,hide_index=True)
    st.markdown('<div class="hydro-good">Study-area boundary: <b>Rizin</b>. The supplied geometry is bundled into this build as <code>rizin.geojson</code> plus a complete <code>rizin.shp</code> set so the app can display and validate the boundary reliably.</div>',unsafe_allow_html=True)

st.markdown("---")
if st.session_state.dataset=="Use demo data":
    st.caption("Demo synthetic mode · 5 years monthly · 1,200 fixed wells · 72,000 scenario observations. Switch to Upload CSV or Live Google Sheet Sync to train on real/read observations.")
else:
    st.caption(f"Research workspace · Rizin AOI + DEA Coastlines {st.session_state.get('coast_year','—')} + hydrologic anchors + spatial well extraction + RF / GAM / XGBoost / LSTM model comparison on the uploaded dataset.")
