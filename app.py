import io
import os
import zipfile
from pathlib import Path

import requests

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

LAKE_WANGARY = {"name":"Lake Wangary", "latitude":-34.54259, "longitude":135.49462, "level_mAHD":3.0}
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
@st.cache_data

def make_data(n_wells=1200, years=4, seed=42):
    rng = np.random.default_rng(seed)

    if AOI_POLY is not None:
        minx,miny,maxx,maxy=AOI_POLY.bounds
        # Sample from a broad west-east transect, then keep points inside Rizin.
        pts=[]
        tries=0
        while len(pts)<n_wells and tries<200000:
            tries+=1
            # Prefer inland-to-coast progression with more points in the interior.
            t=rng.uniform(0,1)
            lon=minx+(maxx-minx)*t + rng.normal(0,(maxx-minx)*0.02)
            lat=miny+(maxy-miny)*(0.18+0.66*(1-t)) + rng.normal(0,(maxy-miny)*0.055)
            if point_inside_boundary(lon,lat): pts.append((lon,lat,t))
        if len(pts)<n_wells:
            # fallback rejection sampling across the AOI
            minx,miny,maxx,maxy=AOI_POLY.bounds
            while len(pts)<n_wells:
                lon=rng.uniform(minx,maxx); lat=rng.uniform(miny,maxy)
                if point_inside_boundary(lon,lat):
                    t=(lon-minx)/max(1e-9,(maxx-minx)); pts.append((lon,lat,t))
        coords=np.array([(p[0],p[1],p[2]) for p in pts])
        lon0,lat0,transect_t=coords[:,0],coords[:,1],coords[:,2]
    else:
        lon0=135.08+rng.uniform(0,.63,n_wells); lat0=-34.75+rng.uniform(0,.42,n_wells); transect_t=(lon0-lon0.min())/(lon0.max()-lon0.min())

    # A location-based hydrogeologic ordering: larger t = more inland.
    coast_distance=np.clip(140 + 14500*transect_t + rng.normal(0,420,n_wells),25,16000)
    inlandness=np.clip(coast_distance/np.nanmax(coast_distance),0,1)
    dem0=np.clip(0.25 + 34*(inlandness**0.72) + rng.normal(0,0.9,n_wells),0.05,40)

    gs=np.array(["Bridgewater Formation","Uley Formation","Wanilla Formation","Sleaford Complex","Hutchison Supergroup","Kiana Granite"])
    geo0=rng.choice(gs,n_wells,p=[.38,.14,.12,.15,.09,.12])
    gf0=pd.Series(geo0).map({"Bridgewater Formation":1.25,"Uley Formation":.8,"Wanilla Formation":.4,"Sleaford Complex":1.55,"Hutchison Supergroup":-.35,"Kiana Granite":1.0}).to_numpy()

    # Location relative to Lake Wangary anchors a surface-water connection signal.
    dist_lake=np.sqrt(((lon0-LAKE_WANGARY["longitude"])/0.0058)**2 + ((lat0-LAKE_WANGARY["latitude"])/0.0048)**2)*1000
    rows=[]
    start_year=2026-years
    for i in range(n_wells):
        for j,yr in enumerate(range(start_year,2026)):
            seasonal=["Summer","Autumn","Winter","Spring"][(j+i)%4]
            sf={"Summer":-.25,"Autumn":.03,"Winter":.46,"Spring":.20}[seasonal]
            rain=np.clip(rng.normal(520,85)+18*np.sin(j),260,820)
            et=np.clip(rng.normal(1050,120),700,1400)
            nd=np.clip(rng.normal(.50,.11),.18,.84)
            nda=np.clip(rng.normal(0,.08),-.25,.25)
            sw=np.clip(dist_lake+rng.normal(0,260),30,11000)
            pressure=rng.normal(1013,8)
            dem=float(np.clip(dem0[i]+rng.normal(0,.08),.05,40))
            coast=float(coast_distance[i])
            # Conceptual synthetic groundwater surface anchored to 0 m AHD near coast
            # and 3 m AHD at Lake Wangary, with inland head rise controlled by relief.
            coastal_lift=0.55*(coast/10000.0)
            lake_influence=1.15*np.exp(-dist_lake[i]/4200.0)
            trend=.08*j+.22*np.sin((j+i)*.7)
            gw=(0.15 + 0.54*dem + coastal_lift + lake_influence*(LAKE_WANGARY["level_mAHD"]-0.8)
                + 0.72*gf0[i] + .006*(rain-500) - .0022*(et-1000) + 1.20*nd + .62*nda
                + .025*(pressure-1013) + sf + trend + rng.normal(0,.52))
            # Keep groundwater conceptually above the 0m coastal datum; clamp only for demo stability.
            gw=max(-0.15,gw)
            rows.append([f"CB_{i+1:05d}",lon0[i],lat0[i],dem,coast,geo0[i],gf0[i],nd,nda,rain,et,sw,pressure,yr,seasonal,gw,dist_lake[i]])
    return pd.DataFrame(rows,columns=["well_id","longitude","latitude","dem_m","distance_coast_m","geology","geology_factor","ndvi_mean","ndvi_anomaly","rainfall_mm","et_mm","surface_water_distance_m","pressure_hpa","year","season","groundwater_level_mAHD","distance_lake_wangary_m"])


def normalise_columns(df):
    d=df.copy()
    aliases={
        "well_id":["well_id","well","site","id","bore_id"],"longitude":["longitude","lon","x_lon"],"latitude":["latitude","lat","y_lat"],
        "groundwater_level_mAHD":["groundwater_level_mAHD","groundwater_level","water_level","gw_level","head_mAHD"],
        "dem_m":["dem_m","dem","elevation","elev_m"],"distance_coast_m":["distance_coast_m","coast_distance_m","distance_to_coast_m"],
        "geology":["geology","formation","lithology"],"season":["season"],"year":["year","date_year"]}
    lower={str(c).strip().lower():c for c in d.columns}
    for target,names in aliases.items():
        if target not in d.columns:
            found=next((lower.get(n) for n in names if n in lower),None)
            if found is not None:d[target]=d[found]
    if "well_id" not in d.columns:d["well_id"]= [f"CB_{i:05d}" for i in range(1,len(d)+1)]
    for c,default in [("dem_m",np.nan),("distance_coast_m",np.nan),("geology","Unknown"),("season","Unknown"),("year",2025)]:
        if c not in d.columns:d[c]=default
    for c in ["longitude","latitude","groundwater_level_mAHD","dem_m","distance_coast_m","year"]:
        if c in d:d[c]=pd.to_numeric(d[c],errors="coerce")
    d=d.dropna(subset=["longitude","latitude","groundwater_level_mAHD"]).reset_index(drop=True)
    d["geology"]=d["geology"].fillna("Unknown").astype(str); d["season"]=d["season"].fillna("Unknown").astype(str)
    return d

FEATURE_CANDIDATES=["longitude","latitude","dem_m","distance_coast_m","geology_factor","ndvi_mean","ndvi_anomaly","rainfall_mm","et_mm","surface_water_distance_m","distance_lake_wangary_m","pressure_hpa","year"]


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


def make_sequences(d,X,y,f,seq_len=4):
    if "well_id" not in d or "year" not in d: raise RuntimeError("LSTM requires well_id and year columns")
    order=d.copy(); order["_row"]=np.arange(len(order)); order=order.sort_values(["well_id","year","_row"])
    xs=[]; ys=[]; rows=[]; groups=[]
    for wid,g in order.groupby("well_id"):
        inds=g["_row"].to_numpy(int); vals=X.iloc[inds].to_numpy(float); tgt=y.iloc[inds].to_numpy(float)
        if len(vals)<2: continue
        for j in range(len(vals)):
            a=max(0,j-seq_len+1); seq=vals[a:j+1]
            if len(seq)<seq_len: seq=np.vstack([np.repeat(seq[[0]],seq_len-len(seq),axis=0),seq])
            xs.append(seq); ys.append(tgt[j]); rows.append(inds[j]); groups.append(wid)
    return np.asarray(xs,dtype=np.float32),np.asarray(ys,dtype=np.float32),np.asarray(rows,int),np.asarray(groups)


def fit_lstm(d,X,y,f,seq_len=4):
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
        wells=d["well_id"].astype(str).unique(); rng=np.random.default_rng(42); rng.shuffle(wells); test=set(wells[:max(1,int(.2*len(wells)))])
        idx_test=np.array([i for i,w in enumerate(d["well_id"].astype(str)) if w in test]); idx_train=np.array([i for i in range(len(d)) if i not in set(idx_test)])
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


def grid_surface(df,col,grid_n=40):
    """Simple inverse-distance weighted map preview bounded by the Rizin extent."""
    q=df[["longitude","latitude",col]].dropna().copy()
    if len(q)<6 or q[col].nunique()<2:return None
    minx,miny,maxx,maxy=AOI_POLY.bounds if AOI_POLY is not None else (q.longitude.min(),q.latitude.min(),q.longitude.max(),q.latitude.max())
    gx=np.linspace(minx,maxx,grid_n); gy=np.linspace(miny,maxy,grid_n)
    xx,yy=np.meshgrid(gx,gy); pxv=q.longitude.to_numpy(float); pyv=q.latitude.to_numpy(float); zv=q[col].to_numpy(float)
    z=np.empty(xx.size,float)
    for i,(a,b) in enumerate(zip(xx.ravel(),yy.ravel())):
        dd=np.sqrt(((pxv-a)*np.cos(np.deg2rad((pyv+b)/2)))**2+(pyv-b)**2)+1e-9
        w=1/(dd**2); z[i]=np.sum(w*zv)/np.sum(w)
    return gx,gy,z.reshape(xx.shape)

# ============================================================
# MAP HELPERS
# ============================================================

def teal_template(fig):
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#ffffff",font=dict(color="#17363a",family="Inter,Arial"),margin=dict(l=8,r=8,t=48,b=8),legend=dict(orientation="h",y=1.08,x=0))
    fig.update_xaxes(showgrid=True,gridcolor="#e7f0ee"); fig.update_yaxes(showgrid=True,gridcolor="#e7f0ee"); return fig


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
    hover_cols=[c for c in ["dem_m","geology","distance_coast_m","distance_lake_wangary_m","groundwater_level_mAHD",value_col,"year","season"] if c in df.columns]
    if hasattr(px,"scatter_map"):
        fig=px.scatter_map(df,lat="latitude",lon="longitude",color=value_col,hover_name="well_id",hover_data=hover_cols,color_continuous_scale=[[0,"#075a67"],[.48,"#37b9ae"],[1,"#d3b355"]],zoom=zoom,height=height,center=center,opacity=.90,size_max=11)
        fig.update_layout(map_style="open-street-map")
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
        if show_coastline:add_coastline_trace(fig,coastline_gdf)
        add_boundary_trace(fig)
    fig.update_layout(margin=dict(l=0,r=0,t=28,b=0),paper_bgcolor="rgba(0,0,0,0)",legend=dict(orientation="h",y=1.02,x=.01),coloraxis_colorbar=dict(title="m AHD"))
    return fig


def draw_surface(fig,df,col):
    surf=grid_surface(df,col)
    if surf is None:return
    lon,lat,zi=surf; slon,slat=np.meshgrid(lon,lat)
    if hasattr(go,"Scattermap"):
        fig.add_trace(go.Scattermap(lat=slat.ravel(),lon=slon.ravel(),mode="markers",marker=dict(size=8,opacity=.16,color=zi.ravel(),colorscale=[[0,"#075a67"],[.48,"#37b9ae"],[1,"#d3b355"]],showscale=False),hoverinfo="skip",name="Piezometric surface preview"))
    else:
        fig.add_trace(go.Scattermapbox(lat=slat.ravel(),lon=slon.ravel(),mode="markers",marker=dict(size=8,opacity=.16,color=zi.ravel(),colorscale=[[0,"#075a67"],[.48,"#37b9ae"],[1,"#d3b355"]],showscale=False),hoverinfo="skip",name="Piezometric surface preview"))

# ============================================================
# STATE / SIDEBAR
# ============================================================
if "dataset" not in st.session_state: st.session_state.dataset="Upload CSV"
if "uploaded_df" not in st.session_state: st.session_state.uploaded_df=None
if "active_model" not in st.session_state: st.session_state.active_model=None
if "models_loaded" not in st.session_state: st.session_state.models_loaded=False
if "coastline" not in st.session_state: st.session_state.coastline=None
if "coastline_status" not in st.session_state: st.session_state.coastline_status="Not loaded"
if "coast_year" not in st.session_state: st.session_state.coast_year=2024
if "loaded_coast_year" not in st.session_state: st.session_state.loaded_coast_year=None

with st.sidebar:
    st.markdown("# CB / HYDRO")
    st.caption("Coffin Bay physical-geography intelligence workspace")
    view=st.radio("Workspace",["Overview","Piezometric map","Model lab","Model drivers","Well explorer","Diagnostics","Scenario lab","Data & export"],label_visibility="collapsed")
    st.markdown("### Data source")
    mode=st.radio("Source",["Upload CSV","Use demo data"],index=0 if st.session_state.dataset=="Upload CSV" else 1)
    if mode=="Upload CSV":
        up=st.file_uploader("Upload well observations",type=["csv"],help="CSV should contain longitude, latitude and groundwater level; year/well_id are recommended for temporal modelling.")
        if up is not None:
            try:
                incoming=normalise_columns(pd.read_csv(up))
                st.session_state.uploaded_df=incoming
                st.session_state.dataset="Upload CSV"
                st.session_state.models_loaded=False
                st.session_state.active_model=None
            except Exception as exc: st.error(f"CSV could not be read: {exc}")
    else:
        st.session_state.dataset="Use demo data"
        st.caption("Optional demonstration dataset for interface testing only.")

base=st.session_state.uploaded_df if st.session_state.dataset=="Upload CSV" and st.session_state.uploaded_df is not None else make_data()
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
    selected=st.multiselect("Models to train / compare",available_models,default=available_models)
    st.markdown("### Coastline reference")
    coast_year=st.selectbox("DEA annual shoreline",list(range(1988,2026)),index=list(range(1988,2026)).index(st.session_state.get("coast_year",2024)))
    st.session_state["coast_year"]=int(coast_year)
    show_coastline=st.checkbox("Show DEA coastline",value=True)
    st.caption("Well-to-coast distance is calculated from the selected annual DEA shoreline clipped to Rizin. 2025 is interim.")
    st.markdown("### Hydrologic controls")
    coastal_anchor=st.number_input("Coastal boundary · m AHD",value=0.0,step=.1,format="%.1f")
    lake_selected=st.checkbox("Use Lake Wangary anchor",value=True)
    lake_level=st.number_input("Lake Wangary · m AHD",value=3.0,step=.1,format="%.1f",disabled=not lake_selected)
    LAKE_WANGARY["level_mAHD"] = float(lake_level)
    COAST_ANCHOR["level_mAHD"] = float(coastal_anchor)
    st.caption("Datum anchors are used by the conceptual surface preview; validate surveyed levels before scientific use.")
    st.markdown("### Filters")
    geos=st.multiselect("Geology",sorted(base.geology.dropna().astype(str).unique()),sorted(base.geology.dropna().astype(str).unique()))
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

if not selected:
    st.warning("Select at least one model. Open Model lab to train and load an active model."); st.stop()

# Train only after the user explicitly asks for model training on Model lab, or on first page load.
train_now = (not st.session_state.models_loaded)
d0,X,y,f,results,preds,imps,models,comparison=train_models(base,tuple(selected))
st.session_state.models_loaded=True
valid_models=[name for name in selected if name in comparison.Model.tolist()] if not comparison.empty else []
if st.session_state.active_model not in valid_models:
    st.session_state.active_model = None
active=st.session_state.active_model

res=d0.copy()
for name,p in preds.items(): res[f"{name}_predicted_mAHD"]=p
res["active_prediction_mAHD"]=res.get(f"{active}_predicted_mAHD",np.nan)
res["active_residual_m"]=res["groundwater_level_mAHD"]-res["active_prediction_mAHD"]
res["active_abs_error_m"]=res["active_residual_m"].abs()
d=res[res.geology.astype(str).isin(geos) & res.season.astype(str).isin(seasons) & res.year.between(years[0],years[1])].copy()

data_label = "UPLOAD" if st.session_state.dataset=="Upload CSV" else "DEMO"
active_label = active.upper().replace(" ", " · ") if active else "NOT LOADED"
status_chip = f"DATA · {data_label} &nbsp;|&nbsp; ACTIVE MODEL · {active.upper()}" if active else f"DATA · {data_label} &nbsp;|&nbsp; MODEL · NOT LOADED"
st.markdown(f'<div class="hero"><div class="hero-title">CB / Groundwater Intelligence</div><div class="hero-sub">Physical-geography workspace · Rizin study boundary · piezometric controls · spatial extraction · multi-model comparison</div><span class="chip">{status_chip}</span></div>',unsafe_allow_html=True)

# ============================================================
# PAGES
# ============================================================
if view=="Overview":
    st.markdown('<div class="section">Hydrologic context</div>',unsafe_allow_html=True)
    a,b,c,e=st.columns(4); a.metric("Wells / observations",f"{len(d):,}"); b.metric("Study boundary","Rizin"); c.metric("DEA coastline",str(st.session_state.get("coast_year")) if coastline_gdf is not None else "Unavailable"); e.metric("Lake Wangary",f"{lake_level:.1f} m AHD" if lake_selected else "Off")
    if coastline_gdf is not None:
        st.markdown(f'<div class="hydro-good"><b>Distance-to-coast engine:</b> shortest perpendicular distance from each well to the clipped DEA annual shoreline for <b>{st.session_state.get("coast_year")}</b>, calculated in EPSG:28353 metres.</div>',unsafe_allow_html=True)
    else:
        st.warning(st.session_state.get("coastline_status","DEA coastline not loaded"))
    st.markdown('<div class="panel panel-accent"><b>Conceptual control points</b><div class="small">The piezometric surface is interpreted relative to two explicit surface-water / datum anchors rather than as an unconstrained black-box prediction.</div><div class="anchor-card">COASTAL BOUNDARY · <b>0.0 m AHD</b> — near-coast hydraulic datum</div><div class="anchor-card">LAKE WANGARY · <b>3.0 m AHD</b> — selected surface-water level anchor</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="section">Hydrogeographic response</div>',unsafe_allow_html=True)
    if active:
        fig=px.scatter(d,x="dem_m",y="groundwater_level_mAHD",color="active_prediction_mAHD",hover_name="well_id",hover_data=["geology","distance_coast_m","year","season"],color_continuous_scale=[[0,"#075a67"],[.48,"#37b9ae"],[1,"#d3b355"]],labels={"dem_m":"DEM elevation (m)","groundwater_level_mAHD":"Observed groundwater (m AHD)","active_prediction_mAHD":f"{active} prediction (m AHD)"})
        st.plotly_chart(teal_template(fig),use_container_width=True)
        st.plotly_chart(make_map(d,"active_prediction_mAHD",f"{active} · spatial prediction layer",coastline_gdf=coastline_gdf,show_coastline=show_coastline),use_container_width=True)
    else:
        st.info("No model is loaded. Train models in Model Lab and choose **Load as active model** to populate prediction layers.")
        st.plotly_chart(make_map(d,"groundwater_level_mAHD","Observed groundwater · hydrogeographic reference",coastline_gdf=coastline_gdf,show_coastline=show_coastline),use_container_width=True)
    st.markdown('<div class="map-caption"><span>Rizin boundary shown as the study-area frame.</span><span>Lake Wangary = 3 m AHD anchor · coastal datum = 0 m AHD.</span></div>',unsafe_allow_html=True)

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
            curve=pt.get("curve_number",pt.get("curveNumber",0)); idx=pt.get("point_index",pt.get("pointNumber",None))
            if curve==0 and idx is not None: ids.append(int(idx))
    except Exception: pass
    ids=sorted(set(i for i in ids if 0<=i<len(d)))
    selected_rows=d.iloc[ids].copy() if ids else d.iloc[0:0].copy()
    st.markdown('<div class="panel"><b>Spatial extraction</b><div class="small">Draw a lasso or box over any part of the Coffin Bay region to extract the enclosed well observations. Point mode is useful for one-well inspection.</div></div>',unsafe_allow_html=True)
    a,b=st.columns([1,2]); a.metric("Selected observations",f"{len(selected_rows):,}")
    if len(selected_rows):
        b.download_button("Export selected wells · CSV",selected_rows.to_csv(index=False).encode(),"coffin_bay_selected_wells.csv","text/csv",use_container_width=True); st.dataframe(selected_rows,use_container_width=True,hide_index=True)
    else: st.caption("No spatial selection yet.")

elif view=="Model lab":
    st.markdown('<div class="section">Model training & active-model selection</div>',unsafe_allow_html=True)
    st.markdown('<div class="panel panel-accent"><b>Train → compare → load</b><div class="small">Select candidate models, train them on the same holdout structure, compare RMSE / MAE / R², then explicitly load one model as the active prediction layer. All downstream pages follow the active model selected here.</div></div>',unsafe_allow_html=True)
    available_results=sorted(comparison.Model.tolist()) if not comparison.empty else []
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
    scatter=px.scatter(res,x=feat,y="active_prediction_mAHD",color="geology",hover_name="well_id",labels={"active_prediction_mAHD":f"{active} predicted groundwater (m AHD)"}); tmp=res[[feat,"active_prediction_mAHD"]].dropna()
    if len(tmp)>2 and tmp[feat].nunique()>1:
        slope,intercept=np.polyfit(tmp[feat].to_numpy(float),tmp["active_prediction_mAHD"].to_numpy(float),1); xl=np.linspace(tmp[feat].min(),tmp[feat].max(),50); scatter.add_trace(go.Scatter(x=xl,y=slope*xl+intercept,mode="lines",name="Linear guide"))
    st.plotly_chart(teal_template(scatter),use_container_width=True)

elif view=="Well explorer":
    st.markdown('<div class="section">Well-by-well inspection</div>',unsafe_allow_html=True)
    if not active:
        st.info("Load an active model from Model Lab first.")
        st.stop()
    w=st.selectbox("Well",d.well_id.astype(str).unique().tolist()); rd=d[d.well_id.astype(str)==w].sort_values("year"); r=rd.iloc[-1]
    a,b,c,e=st.columns(4); a.metric("Observed",f"{r.groundwater_level_mAHD:.2f} m AHD"); b.metric(f"{active} prediction",f"{r.active_prediction_mAHD:.2f} m AHD" if pd.notna(r.active_prediction_mAHD) else "—"); c.metric("Residual",f"{r.active_residual_m:.2f} m" if pd.notna(r.active_residual_m) else "—"); e.metric("DEM",f"{r.dem_m:.2f} m")
    st.dataframe(rd[["well_id","year","season","groundwater_level_mAHD","active_prediction_mAHD","active_residual_m","dem_m","distance_coast_m","distance_lake_wangary_m","geology"]],use_container_width=True,hide_index=True)
    fig=px.line(rd,x="year",y=["groundwater_level_mAHD","active_prediction_mAHD"],markers=True,title=f"Observed vs {active} prediction — {w}"); st.plotly_chart(teal_template(fig),use_container_width=True)

elif view=="Diagnostics":
    st.markdown('<div class="section">Active model diagnostics</div>',unsafe_allow_html=True)
    if not active:
        st.info("Load an active model from Model Lab first.")
        st.stop()
    fig=px.scatter(res,x="groundwater_level_mAHD",y="active_prediction_mAHD",color="active_abs_error_m",hover_name="well_id",color_continuous_scale=[[0,"#075a67"],[.48,"#37b9ae"],[1,"#d3b355"]],labels={"groundwater_level_mAHD":"Observed (m AHD)","active_prediction_mAHD":f"{active} prediction (m AHD)"})
    lo=min(res.groundwater_level_mAHD.min(),res.active_prediction_mAHD.min()); hi=max(res.groundwater_level_mAHD.max(),res.active_prediction_mAHD.max()); fig.add_shape(type="line",x0=lo,y0=lo,x1=hi,y1=hi,line=dict(dash="dash",color="#557b7f")); st.plotly_chart(teal_template(fig),use_container_width=True)
    hist=px.histogram(res,x="active_residual_m",nbins=35,title="Residual distribution"); st.plotly_chart(teal_template(hist),use_container_width=True); st.dataframe(res.sort_values("active_abs_error_m",ascending=False)[["well_id","geology","dem_m","distance_coast_m","groundwater_level_mAHD","active_prediction_mAHD","active_residual_m","active_abs_error_m"]].head(80),use_container_width=True,hide_index=True)

elif view=="Scenario lab":
    st.markdown('<div class="section">Hydrologic scenario lab</div>',unsafe_allow_html=True)
    if not active:
        st.info("Load an active model from Model Lab first.")
        st.stop()
    st.markdown('<div class="hydro-note">Scenarios are conceptual. The controls make the coastal and Lake Wangary anchors explicit so scenario changes are visible relative to a physical datum framework.</div>',unsafe_allow_html=True)
    rr=res.iloc[0].copy(); c1,c2,c3=st.columns(3); dem=c1.slider("DEM (m)",0.,40.,float(np.clip(rr.dem_m,0,40)),.5); rain=c2.slider("Rainfall (mm)",250,820,int(rr.rainfall_mm),10); nd=c3.slider("NDVI",.1,.95,float(rr.ndvi_mean),.01); coast=c1.slider("Distance to coast (m)",20,16000,int(np.clip(rr.distance_coast_m,20,16000)),100); et=c2.slider("ET (mm)",700,1400,int(rr.et_mm),10); sw=c3.slider("Surface-water distance (m)",20,11000,int(rr.surface_water_distance_m),100)
    row=rr.copy(); row.update({"dem_m":dem,"rainfall_mm":rain,"ndvi_mean":nd,"distance_coast_m":coast,"et_mm":et,"surface_water_distance_m":sw}); Xrow=pd.DataFrame([{q:(row[q] if pd.notna(row[q]) else res[q].median()) for q in f}]); mdl=models.get(active)
    if active!="LSTM": pred=float(mdl.predict(Xrow)[0]); st.metric("Scenario prediction",f"{pred:.2f} m AHD")
    else: st.info("The LSTM is sequence-based; use Well explorer or Model lab for temporal predictions and benchmark results.")

else:
    st.markdown('<div class="section">Data, spatial selection & export</div>',unsafe_allow_html=True)
    st.markdown('<div class="panel">The recommended extraction workflow is <b>Piezometric map → Lasso / Box → Export selected wells</b>. The current filters can also be exported here.</div>',unsafe_allow_html=True)
    a,b,c=st.columns(3); a.metric("Filtered observations",f"{len(d):,}"); b.metric("Longitude span",f"{d.longitude.max()-d.longitude.min():.3f}°" if len(d) else "—"); c.metric("Latitude span",f"{d.latitude.max()-d.latitude.min():.3f}°" if len(d) else "—")
    st.download_button("Export current filtered dataset · CSV",d.to_csv(index=False).encode(),"coffin_bay_filtered_wells.csv","text/csv",use_container_width=True); st.dataframe(d.head(700),use_container_width=True,hide_index=True)
    st.markdown('<div class="hydro-good">Study-area boundary: <b>Rizin</b>. The supplied geometry is bundled into this build as <code>rizin.geojson</code> plus a complete <code>rizin.shp</code> set so the app can display and validate the boundary reliably.</div>',unsafe_allow_html=True)

st.markdown("---")
if st.session_state.dataset=="Use demo data":
    st.caption("Research prototype · Demo data mode is for interface and workflow testing only.")
else:
    st.caption(f"Research workspace · Rizin AOI + DEA Coastlines {st.session_state.get('coast_year','—')} + hydrologic anchors + spatial well extraction + RF / GAM / XGBoost / LSTM model comparison on the uploaded dataset.")
