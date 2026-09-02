import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

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
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Visual system
# -----------------------------
st.markdown(
    """
    <style>
    :root{--ink:#143f45;--deep:#0a6b70;--teal:#149995;--aqua:#e2f7f2;--mint:#f0faf7;--sand:#f7f2e5;--line:#c7e5df;--card:#fff}
    .stApp{background:linear-gradient(135deg,#eef9f6 0%,#f8f5ea 100%);color:var(--ink)}
    [data-testid="stHeader"]{background:rgba(255,255,255,.76)}
    [data-testid="stSidebar"]{background:linear-gradient(180deg,#0a686d 0%,#084b54 100%)}
    [data-testid="stSidebar"] *{color:#f2fffc!important}
    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"]{background:#43c7b8!important}
    .cbg-topbar{background:rgba(255,255,255,.9);border:1px solid var(--line);padding:18px 22px;border-radius:18px;box-shadow:0 7px 24px rgba(23,54,58,.07)}
    .cbg-title{font-size:30px;font-weight:800;margin:0;color:var(--ink);letter-spacing:-.6px}
    .cbg-sub{margin-top:4px;color:#5c7879;font-size:14px}
    .cbg-badge{display:inline-block;margin-top:10px;padding:5px 10px;border-radius:999px;background:#e5f7ef;color:#0d6b58;font-size:12px;font-weight:700}
    .cbg-section{margin-top:18px;margin-bottom:8px;font-size:18px;font-weight:800;color:var(--ink)}
    .cbg-card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:16px 18px;box-shadow:0 6px 20px rgba(20,76,72,.06)}
    .cbg-note{background:#fffdf2;border:1px solid #e7dfb2;border-left:5px solid #c8a83e;border-radius:13px;padding:12px 14px;color:#5c552e;font-size:13px}
    .cbg-good{background:#e9f8f2;border:1px solid #bfe3d8;border-left:5px solid #15937f;border-radius:13px;padding:12px 14px;color:#245e54;font-size:13px}
    .cbg-muted{color:#648080;font-size:13px}
    div[data-testid="stMetric"]{background:#fff;border:1px solid var(--line);padding:12px 14px;border-radius:14px;box-shadow:0 6px 18px rgba(20,76,72,.05)}
    .stButton>button,.stDownloadButton>button{border-radius:10px;border:1px solid #9bd6cb}
    </style>
    """, unsafe_allow_html=True,
)

# -----------------------------
# Synthetic spatio-temporal data
# -----------------------------
@st.cache_data
def make_data(n_wells=300, years=6, seed=42):
    rng = np.random.default_rng(seed)
    center_lon, center_lat = 135.46, -34.445
    x = rng.normal(0, 1, n_wells)
    y = rng.normal(0, 1, n_wells)
    lon0 = center_lon + np.clip(x * 0.34 + 0.05*y, -0.52, 0.52)
    lat0 = center_lat + np.clip(y * 0.23 - 0.03*x, -0.38, 0.38)
    dem0 = np.clip(1.2 + 31*rng.beta(2.1,4.4,n_wells) + 0.7*np.sin(x), .2, 38)
    coast0 = np.clip(np.sqrt(((lon0-center_lon)/.006)**2 + ((lat0-center_lat)/.0047)**2)*100 + rng.normal(850,400,n_wells), 30, 16000)
    gs = np.array(["Bridgewater Formation","Uley Formation","Wanilla Formation","Sleaford Complex","Hutchison Supergroup","Kiana Granite"])
    geo0 = rng.choice(gs, n_wells, p=[.33,.15,.12,.16,.10,.14])
    gf0 = pd.Series(geo0).map({"Bridgewater Formation":1.2,"Uley Formation":.75,"Wanilla Formation":.35,"Sleaford Complex":1.6,"Hutchison Supergroup":-.35,"Kiana Granite":1.05}).to_numpy()

    rows=[]
    start_year = 2026 - years
    for i in range(n_wells):
        for j, yr in enumerate(range(start_year, 2026)):
            seasonal = ["Summer","Autumn","Winter","Spring"][(j+i)%4]
            sf = {"Summer":-.35,"Autumn":.05,"Winter":.55,"Spring":.30}[seasonal]
            rain = np.clip(rng.normal(520,90) + 18*np.sin(j), 260, 820)
            et = np.clip(rng.normal(1050,120),700,1400)
            nd = np.clip(rng.normal(.50,.11),.18,.84)
            nda = np.clip(rng.normal(0,.08),-.25,.25)
            sw = np.clip(rng.gamma(2.0,700),20,11000)
            pressure = rng.normal(1013,8)
            dem = float(np.clip(dem0[i] + rng.normal(0,.08),.2,38))
            coast = float(coast0[i])
            trend = 0.08*j + 0.25*np.sin((j+i)*.7)
            gw = (3 + .52*dem - .00010*coast + 1.05*gf0[i] + .006*(rain-500)
                  - .0022*(et-1000) + 1.35*nd + .65*nda - .000055*sw
                  + .025*(pressure-1013) + sf + trend + .10*np.sin(dem/3)
                  + .000012*coast**1.25 + rng.normal(0,.65))
            rows.append([f"CB_{i+1:05d}",lon0[i],lat0[i],dem,coast,geo0[i],gf0[i],nd,nda,rain,et,sw,pressure,yr,seasonal,gw])
    return pd.DataFrame(rows, columns=["well_id","longitude","latitude","dem_m","distance_coast_m","geology","geology_factor","ndvi_mean","ndvi_anomaly","rainfall_mm","et_mm","surface_water_distance_m","pressure_hpa","year","season","groundwater_level_mAHD"])


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

FEATURE_CANDIDATES=["longitude","latitude","dem_m","distance_coast_m","geology_factor","ndvi_mean","ndvi_anomaly","rainfall_mm","et_mm","surface_water_distance_m","pressure_hpa","year"]


def prepare_features(data):
    d=data.copy(); y=pd.to_numeric(d["groundwater_level_mAHD"],errors="coerce"); d=d[y.notna()].copy(); y=y[y.notna()]
    f=[c for c in FEATURE_CANDIDATES if c in d.columns and (pd.to_numeric(d[c],errors="coerce").notna().any())]
    if "longitude" not in f:f.append("longitude")
    if "latitude" not in f:f.append("latitude")
    X=d[f].copy()
    for c in f:
        X[c]=pd.to_numeric(X[c],errors="coerce")
        X[c]=X[c].fillna(X[c].median() if X[c].notna().any() else 0.0)
    return d.reset_index(drop=True), X.reset_index(drop=True), y.reset_index(drop=True), f


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
    xtr=torch.tensor(scaler.transform(xs[train_mask].reshape(-1,len(f))).reshape(-1,seq_len,len(f)),dtype=torch.float32)
    ytr=torch.tensor(ys[train_mask],dtype=torch.float32)
    xall=torch.tensor(scaler.transform(xs.reshape(-1,len(f))).reshape(-1,seq_len,len(f)),dtype=torch.float32)
    model=LSTMRegressor(len(f)); opt=torch.optim.Adam(model.parameters(),lr=.003); loss_fn=nn.MSELoss()
    model.train()
    batch=128
    for _ in range(45):
        perm=torch.randperm(len(xtr))
        for s in range(0,len(xtr),batch):
            ix=perm[s:s+batch]; opt.zero_grad(); loss=loss_fn(model(xtr[ix]),ytr[ix]); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad(): pred=model(xall).numpy()
    test_pred=pred[test_mask]; test_y=ys[test_mask]
    imp=[]
    base=mean_squared_error(test_y,test_pred)**.5
    # Permute one feature across sequence positions to estimate predictive reliance.
    rng=np.random.default_rng(42)
    for k,feat in enumerate(f):
        xp=xall.clone(); perm=rng.permutation(len(xp)); xp[:,:,k]=xp[perm,:,k]
        with torch.no_grad(): pp=model(xp).numpy()
        imp.append(max(0,mean_squared_error(test_y,pp[test_mask])**.5-base))
    importance=pd.Series(imp,index=f).sort_values(ascending=False)
    full_pred=pd.Series(np.nan,index=np.arange(len(d))); full_pred.iloc[rows]=pred
    return (model,scaler,seq_len),full_pred.to_numpy(),metrics_dict(test_y,test_pred),importance


@st.cache_data(show_spinner=False)
def train_models(data,selected_models):
    d,X,y,f=prepare_features(data)
    # Prefer well-grouped holdout for temporal synthetic data and random holdout otherwise.
    if d["well_id"].nunique()<len(d):
        wells=d["well_id"].astype(str).unique(); rng=np.random.default_rng(42); rng.shuffle(wells); test=set(wells[:max(1,int(.2*len(wells)))])
        idx_test=np.array([i for i,w in enumerate(d["well_id"].astype(str)) if w in test]); idx_train=np.array([i for i in range(len(d)) if i not in set(idx_test)])
    else:
        idx_train,idx_test=train_test_split(np.arange(len(d)),test_size=.2,random_state=42)
    results={}; preds={}; imps={}; models={}
    for name in selected_models:
        try:
            if name=="Random Forest": out=fit_rf(d,X,y,f,idx_train,idx_test)
            elif name=="XGBoost": out=fit_xgb(d,X,y,f,idx_train,idx_test)
            elif name=="GAM": out=fit_gam(d,X,y,f,idx_train,idx_test)
            else: out=fit_lstm(d,X,y,f)
            models[name],preds[name],results[name],imps[name]=out
        except Exception as exc:
            results[name]={"Error":str(exc)}
    table=[]
    for name,m in results.items():
        if "Error" not in m: table.append({"Model":name,**m,"Top feature":imps[name].index[0] if len(imps[name]) else "—"})
    comparison=pd.DataFrame(table).sort_values(["RMSE (m)","MAE (m)"]) if table else pd.DataFrame()
    return d,X,y,f,results,preds,imps,models,comparison


# Helpers

def teal_template(fig):
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#fff",font=dict(color="#17363a",family="Inter,Arial"),margin=dict(l=12,r=12,t=45,b=12),legend=dict(orientation="h",y=1.08,x=0))
    fig.update_xaxes(showgrid=True,gridcolor="#e5efed"); fig.update_yaxes(showgrid=True,gridcolor="#e5efed"); return fig


def make_map(df,value_col,title):
    if df.empty:return go.Figure()
    center_lat,center_lon=float(df.latitude.median()),float(df.longitude.median())
    hover_cols=[c for c in ["dem_m","geology","distance_coast_m","groundwater_level_mAHD",value_col,"year","season"] if c in df.columns]
    use_new=hasattr(px,"scatter_map")
    if use_new:
        fig=px.scatter_map(df,lat="latitude",lon="longitude",color=value_col,hover_name="well_id",hover_data=hover_cols,color_continuous_scale=[[0,"#216e70"],[.5,"#4fb9ae"],[1,"#dfb94d"]],zoom=8.8,height=700,center={"lat":center_lat,"lon":center_lon},opacity=.84,size_max=12)
        fig.update_layout(map_style="open-street-map")
    else:
        fig=px.scatter_mapbox(df,lat="latitude",lon="longitude",color=value_col,hover_name="well_id",hover_data=hover_cols,color_continuous_scale=[[0,"#216e70"],[.5,"#4fb9ae"],[1,"#dfb94d"]],zoom=8.8,height=700,center={"lat":center_lat,"lon":center_lon},opacity=.84,size_max=12)
        fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(title=dict(text=title,x=.02,xanchor="left"),coloraxis_colorbar=dict(title="")); return fig


def grid_surface(df,value_col,nx=40,ny=32):
    d=df.dropna(subset=["longitude","latitude",value_col]);
    if len(d)<6:return None
    lon=np.linspace(d.longitude.min(),d.longitude.max(),nx); lat=np.linspace(d.latitude.min(),d.latitude.max(),ny); gx,gy=np.meshgrid(lon,lat); x=d.longitude.to_numpy(); y=d.latitude.to_numpy(); z=d[value_col].to_numpy(float)
    zi=np.empty_like(gx); pts=np.column_stack([x,y]); grid=np.column_stack([gx.ravel(),gy.ravel()])
    for start in range(0,len(grid),2500):
        g=grid[start:start+2500]; dist2=((g[:,None,0]-pts[None,:,0])**2+(g[:,None,1]-pts[None,:,1])**2); w=1/(dist2+1e-8); zi.ravel()[start:start+len(g)]=(w*z[None,:]).sum(axis=1)/w.sum(axis=1)
    return lon,lat,zi

# -----------------------------
# State + data
# -----------------------------
if "dataset" not in st.session_state: st.session_state.dataset="Synthetic demonstration"
if "uploaded_df" not in st.session_state: st.session_state.uploaded_df=None

with st.sidebar:
    st.markdown("## 💧 Coffin Bay")
    st.caption("Groundwater intelligence workspace")
    view=st.radio("Workspace",["Overview","Piezometric map","Model lab","Model drivers","Well explorer","Diagnostics","Scenario lab","Data & export"])
    st.markdown("### Dataset")
    mode=st.radio("Source",["Synthetic demonstration","Upload CSV"],index=0 if st.session_state.dataset=="Synthetic demonstration" else 1)
    if mode=="Upload CSV":
        up=st.file_uploader("Upload well observations",type=["csv"])
        if up is not None:
            try: st.session_state.uploaded_df=normalise_columns(pd.read_csv(up)); st.session_state.dataset="Uploaded CSV"
            except Exception as exc: st.error(f"CSV could not be read: {exc}")
    else: st.session_state.dataset="Synthetic demonstration"

base=st.session_state.uploaded_df if st.session_state.dataset=="Uploaded CSV" and st.session_state.uploaded_df is not None else make_data()
if "geology_factor" not in base.columns:
    base=base.copy(); base["geology_factor"]=0.0
for c in FEATURE_CANDIDATES:
    if c not in base.columns: base[c]=np.nan

available_models=["Random Forest","GAM"]
if XGB_OK: available_models.append("XGBoost")
if TORCH_OK: available_models.append("LSTM")
with st.sidebar:
    st.markdown("### Model training")
    selected=st.multiselect("Train / compare models",available_models,default=available_models)
    active=st.selectbox("Active prediction layer",selected if selected else available_models[:1])
    geos=st.multiselect("Geology",sorted(base.geology.dropna().astype(str).unique()),sorted(base.geology.dropna().astype(str).unique()))
    seasons=st.multiselect("Season",sorted(base.season.dropna().astype(str).unique()),sorted(base.season.dropna().astype(str).unique()))
    ymin,ymax=int(pd.to_numeric(base.year,errors="coerce").min()),int(pd.to_numeric(base.year,errors="coerce").max()); years=st.slider("Observation year",ymin,ymax,(ymin,ymax))
    st.markdown('<div class="cbg-note">Synthetic demonstration data are for workflow testing only. Model rankings are not evidence of real Coffin Bay groundwater behaviour.</div>',unsafe_allow_html=True)

if not selected: st.warning("Select at least one model in the sidebar."); st.stop()

d0,X,y,f,results,preds,imps,models,comparison=train_models(base,tuple(selected))
res=d0.copy()
for name,p in preds.items():
    res[f"{name}_predicted_mAHD"]=p
res["active_prediction_mAHD"]=res.get(f"{active}_predicted_mAHD",np.nan)
res["active_residual_m"]=res["groundwater_level_mAHD"]-res["active_prediction_mAHD"]
res["active_abs_error_m"]=res["active_residual_m"].abs()

d=res[res.geology.astype(str).isin(geos) & res.season.astype(str).isin(seasons) & res.year.between(years[0],years[1])].copy()

st.markdown('<div class="cbg-topbar"><div class="cbg-title">💧 Coffin Bay Groundwater Intelligence</div><div class="cbg-sub">Interactive hydrogeology workspace · map extraction · multi-model training · validation and model comparison</div><span class="cbg-badge">'+st.session_state.dataset+' · Active model: '+active+'</span></div>',unsafe_allow_html=True)

# -----------------------------
# Pages
# -----------------------------
if view=="Overview":
    st.markdown('<div class="cbg-section">Study snapshot</div>',unsafe_allow_html=True)
    a,b,c,e=st.columns(4); a.metric("Wells / observations",f"{len(d):,}"); b.metric("Active model",active); m=results.get(active,{})
    c.metric("RMSE",f"{m.get('RMSE (m)',np.nan):.2f} m" if 'RMSE (m)' in m else "—"); e.metric("MAE",f"{m.get('MAE (m)',np.nan):.2f} m" if 'MAE (m)' in m else "—")
    st.markdown('<div class="cbg-section">Active model groundwater response</div>',unsafe_allow_html=True)
    fig=px.scatter(d,x="dem_m",y="groundwater_level_mAHD",color="active_prediction_mAHD",hover_name="well_id",hover_data=["geology","distance_coast_m","year","season"],color_continuous_scale=[[0,"#216e70"],[.5,"#64c9be"],[1,"#dfb94d"]],labels={"dem_m":"DEM elevation (m)","groundwater_level_mAHD":"Observed groundwater (m AHD)","active_prediction_mAHD":f"{active} prediction (m AHD)"})
    st.plotly_chart(teal_template(fig),use_container_width=True)
    st.plotly_chart(make_map(d,"active_prediction_mAHD",f"{active} predicted groundwater · filtered observations"),use_container_width=True)

elif view=="Piezometric map":
    st.markdown('<div class="cbg-section">Piezometric surface explorer</div>',unsafe_allow_html=True)
    c1,c2,c3=st.columns([1.1,1,1]); variable=c1.selectbox("Map variable",[f"{active} predicted groundwater","Observed groundwater","Absolute prediction error"]); col={f"{active} predicted groundwater":"active_prediction_mAHD","Observed groundwater":"groundwater_level_mAHD","Absolute prediction error":"active_abs_error_m"}[variable]; mode=c2.selectbox("Selection mode",["Lasso","Box","Point"]); layer=c3.checkbox("Show continuous IDW preview",True)
    fig=make_map(d,col,f"{variable} · select wells on the map")
    if layer:
        surf=grid_surface(d,col)
        if surf is not None:
            lon,lat,zi=surf; slon,slat=np.meshgrid(lon,lat); use_new=hasattr(px,"scatter_map")
            trace=(go.Scattermap(lat=slat.ravel(),lon=slon.ravel(),mode="markers",marker=dict(size=9,opacity=.13,color=zi.ravel(),colorscale=[[0,"#216e70"],[.5,"#4fb9ae"],[1,"#dfb94d"]],showscale=False),hoverinfo="skip",name="IDW preview") if use_new else go.Scattermapbox(lat=slat.ravel(),lon=slon.ravel(),mode="markers",marker=dict(size=8,opacity=.13,color=zi.ravel(),colorscale=[[0,"#216e70"],[.5,"#4fb9ae"],[1,"#dfb94d"]],showscale=False),hoverinfo="skip",name="IDW preview")); fig.add_trace(trace)
    event=st.plotly_chart(fig,use_container_width=True,key="piezometric_map",on_select="rerun",selection_mode={"Lasso":"lasso","Box":"box","Point":"points"}[mode])
    ids=[]
    try:
        for pt in event.selection.points:
            curve=pt.get("curve_number",pt.get("curveNumber",0)); idx=pt.get("point_index",pt.get("pointNumber",None))
            if curve==0 and idx is not None:ids.append(int(idx))
    except Exception: pass
    ids=sorted(set(i for i in ids if 0<=i<len(d)))
    selected_rows=d.iloc[ids].copy() if ids else d.iloc[0:0].copy()
    st.markdown('<div class="cbg-card"><b>Spatial extraction</b><br><span class="cbg-muted">Use lasso, box, or point selection directly on the map. The selected well observations remain editable as a table and can be exported as CSV.</span></div>',unsafe_allow_html=True)
    a,b=st.columns([1,2]); a.metric("Selected observations",f"{len(selected_rows):,}")
    if len(selected_rows):
        b.download_button("Export selected wells · CSV",selected_rows.to_csv(index=False).encode(),"coffin_bay_selected_wells.csv","text/csv",use_container_width=True)
        st.dataframe(selected_rows,use_container_width=True,hide_index=True)
    else: st.caption("No spatial selection yet.")

elif view=="Model lab":
    st.markdown('<div class="cbg-section">Multi-model training & champion selection</div>',unsafe_allow_html=True)
    st.markdown('<div class="cbg-card"><b>Workflow:</b> choose models in the sidebar → train on the same holdout structure → compare RMSE/MAE/R² → identify the best model → inspect its leading prediction feature. The synthetic dataset has repeated annual observations to support an exploratory LSTM sequence model.</div>',unsafe_allow_html=True)
    if not comparison.empty:
        st.dataframe(comparison,use_container_width=True,hide_index=True)
        champion=comparison.iloc[0]
        a,b,c=st.columns(3); a.metric("Best model",champion["Model"]); b.metric("Best RMSE",f"{champion['RMSE (m)']:.3f} m"); c.metric("Leading predictor",champion["Top feature"])
        st.markdown(f'<div class="cbg-good">Champion by lowest holdout RMSE: <b>{champion["Model"]}</b>. Most influential predictor for this champion: <b>{champion["Top feature"]}</b>. This is a synthetic-data benchmark, not a real-world causal finding.</div>',unsafe_allow_html=True)
        melted=comparison.melt(id_vars=["Model"],value_vars=["RMSE (m)","MAE (m)"],var_name="Metric",value_name="Value")
        fig=px.bar(melted,x="Model",y="Value",color="Metric",barmode="group",title="Holdout error comparison")
        st.plotly_chart(teal_template(fig),use_container_width=True)
        fig2=px.bar(comparison.sort_values("R²"),x="R²",y="Model",orientation="h",title="Holdout R² comparison")
        st.plotly_chart(teal_template(fig2),use_container_width=True)
    for name,rr in results.items():
        if "Error" in rr: st.error(f"{name}: {rr['Error']}")

elif view=="Model drivers":
    st.markdown('<div class="cbg-section">Prediction drivers for the active model</div>',unsafe_allow_html=True)
    imp=imps.get(active,pd.Series(dtype=float)); imp_df=imp.rename("Importance").reset_index().rename(columns={"index":"Feature"})
    st.caption("Importance is a predictive diagnostic. It is not causal evidence.")
    fig=px.bar(imp_df.sort_values("Importance"),x="Importance",y="Feature",orientation="h",title=f"{active} feature importance")
    st.plotly_chart(teal_template(fig),use_container_width=True)
    feat=st.selectbox("Inspect predictor",imp_df.Feature.tolist() if not imp_df.empty else f)
    scatter=px.scatter(res,x=feat,y="active_prediction_mAHD",color="geology",hover_name="well_id",labels={"active_prediction_mAHD":f"{active} predicted groundwater (m AHD)"})
    tmp=res[[feat,"active_prediction_mAHD"]].dropna()
    if len(tmp)>2 and tmp[feat].nunique()>1:
        slope,intercept=np.polyfit(tmp[feat].to_numpy(float),tmp["active_prediction_mAHD"].to_numpy(float),1); xl=np.linspace(tmp[feat].min(),tmp[feat].max(),50); scatter.add_trace(go.Scatter(x=xl,y=slope*xl+intercept,mode="lines",name="Linear guide"))
    st.plotly_chart(teal_template(scatter),use_container_width=True)

elif view=="Well explorer":
    st.markdown('<div class="cbg-section">Well-by-well inspection</div>',unsafe_allow_html=True)
    w=st.selectbox("Well",d.well_id.astype(str).unique().tolist()); rd=d[d.well_id.astype(str)==w].sort_values("year"); r=rd.iloc[-1]
    a,b,c,e=st.columns(4); a.metric("Observed",f"{r.groundwater_level_mAHD:.2f} m AHD"); b.metric(f"{active} prediction",f"{r.active_prediction_mAHD:.2f} m AHD" if pd.notna(r.active_prediction_mAHD) else "—"); c.metric("Residual",f"{r.active_residual_m:.2f} m" if pd.notna(r.active_residual_m) else "—"); e.metric("DEM",f"{r.dem_m:.2f} m")
    st.dataframe(rd[["well_id","year","season","groundwater_level_mAHD","active_prediction_mAHD","active_residual_m","dem_m","distance_coast_m","geology"]],use_container_width=True,hide_index=True)
    fig=px.line(rd,x="year",y=["groundwater_level_mAHD","active_prediction_mAHD"],markers=True,title=f"Observed vs {active} prediction — {w}")
    st.plotly_chart(teal_template(fig),use_container_width=True)

elif view=="Diagnostics":
    st.markdown('<div class="cbg-section">Active model diagnostics</div>',unsafe_allow_html=True)
    fig=px.scatter(res,x="groundwater_level_mAHD",y="active_prediction_mAHD",color="active_abs_error_m",hover_name="well_id",color_continuous_scale=[[0,"#216e70"],[.5,"#64c9be"],[1,"#dfb94d"]],labels={"groundwater_level_mAHD":"Observed (m AHD)","active_prediction_mAHD":f"{active} prediction (m AHD)"})
    lo=min(res.groundwater_level_mAHD.min(),res.active_prediction_mAHD.min()); hi=max(res.groundwater_level_mAHD.max(),res.active_prediction_mAHD.max()); fig.add_shape(type="line",x0=lo,y0=lo,x1=hi,y1=hi,line=dict(dash="dash",color="#335d60")); st.plotly_chart(teal_template(fig),use_container_width=True)
    hist=px.histogram(res,x="active_residual_m",nbins=35,title="Residual distribution"); st.plotly_chart(teal_template(hist),use_container_width=True)
    st.dataframe(res.sort_values("active_abs_error_m",ascending=False)[["well_id","geology","dem_m","distance_coast_m","groundwater_level_mAHD","active_prediction_mAHD","active_residual_m","active_abs_error_m"]].head(80),use_container_width=True,hide_index=True)

elif view=="Scenario lab":
    st.markdown('<div class="cbg-section">Scenario lab</div>',unsafe_allow_html=True)
    st.markdown('<div class="cbg-note">Exploratory scenario testing. Not a validated groundwater forecast.</div>',unsafe_allow_html=True)
    rr=res.iloc[0].copy(); c1,c2,c3=st.columns(3); dem=c1.slider("DEM (m)",0.,38.,float(rr.dem_m),.5); rain=c2.slider("Rainfall (mm)",250,820,int(rr.rainfall_mm),10); nd=c3.slider("NDVI",.1,.95,float(rr.ndvi_mean),.01); coast=c1.slider("Distance to coast (m)",20,16000,int(rr.distance_coast_m),100); et=c2.slider("ET (mm)",700,1400,int(rr.et_mm),10); sw=c3.slider("Surface-water distance (m)",20,11000,int(rr.surface_water_distance_m),100)
    row=rr.copy(); row.update({"dem_m":dem,"rainfall_mm":rain,"ndvi_mean":nd,"distance_coast_m":coast,"et_mm":et,"surface_water_distance_m":sw}); Xrow=pd.DataFrame([{q:(row[q] if pd.notna(row[q]) else res[q].median()) for q in f}])
    mdl=models.get(active)
    if active!="LSTM": pred=float(mdl.predict(Xrow)[0]); st.metric("Scenario prediction",f"{pred:.2f} m AHD")
    else: st.info("The LSTM is sequence-based; use Well explorer or Model lab for temporal predictions and benchmark results.")

else:
    st.markdown('<div class="cbg-section">Data, selection and export centre</div>',unsafe_allow_html=True)
    st.markdown('<div class="cbg-card">Use the sidebar to load a CSV, filter observations, then use the <b>Piezometric map</b> to draw a box or lasso around any part of the region. Selected observations can be exported immediately.</div>',unsafe_allow_html=True)
    a,b,c=st.columns(3); a.metric("Filtered observations",f"{len(d):,}"); b.metric("Longitude span",f"{d.longitude.max()-d.longitude.min():.3f}°" if len(d) else "—"); c.metric("Latitude span",f"{d.latitude.max()-d.latitude.min():.3f}°" if len(d) else "—")
    st.download_button("Export current filtered dataset · CSV",d.to_csv(index=False).encode(),"coffin_bay_filtered_wells.csv","text/csv",use_container_width=True)
    st.dataframe(d.head(700),use_container_width=True,hide_index=True)
    st.caption("CSV upload accepts longitude/lon, latitude/lat, groundwater_level_mAHD/water_level, DEM/elevation, geology/formation and optional temporal/model predictors.")

st.markdown("---")
st.caption("Research prototype: synthetic or uploaded observations → multi-model training (RF / GAM / XGBoost / LSTM) → model comparison → best-model feature diagnostics → spatial exploration → CSV extraction. Validate CRS, observations and modelling choices before scientific use.")
