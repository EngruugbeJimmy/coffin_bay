import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Coffin Bay Groundwater Intelligence", page_icon="💧", layout="wide")

@st.cache_data
def make_data(n=1200, seed=42):
    rng=np.random.default_rng(seed)
    e=rng.uniform(540000,585000,n); no=rng.uniform(6050000,6095000,n)
    lon=135.45+(e-e.min())/(e.max()-e.min())*.65
    lat=-35.65+(no-no.min())/(no.max()-no.min())*.55
    dem=2+28*rng.beta(2,4,n)
    coast=np.clip(rng.gamma(2.2,900,n),20,12000)
    gs=np.array(["Bridgewater Formation","Uley Formation","Wanilla Formation","Sleaford Complex","Hutchison Supergroup","Kiana Granite"])
    geo=rng.choice(gs,n,p=[.35,.15,.12,.14,.10,.14])
    gf=pd.Series(geo).map({"Bridgewater Formation":1.2,"Uley Formation":.75,"Wanilla Formation":.35,
                           "Sleaford Complex":1.6,"Hutchison Supergroup":-.35,"Kiana Granite":1.05}).to_numpy()
    rain=rng.normal(520,95,n).clip(250,800); et=rng.normal(1050,120,n).clip(700,1400)
    nd=rng.uniform(.2,.82,n); nda=rng.normal(0,.08,n).clip(-.25,.25)
    sw=np.clip(rng.gamma(2,700,n),20,10000); p=rng.normal(1013,8,n); year=rng.integers(2000,2026,n)
    season=rng.choice(["Summer","Autumn","Winter","Spring"],n)
    sf=pd.Series(season).map({"Summer":-.35,"Autumn":.05,"Winter":.55,"Spring":.30}).to_numpy()
    gw=3+.52*dem-.00010*coast+1.05*gf+.006*(rain-500)-.0022*(et-1000)+1.35*nd+.65*nda-.000055*sw+.025*(p-1013)+sf+.002*(year-2010)+.10*np.sin(dem/3)+.000012*coast**1.25+rng.normal(0,.65,n)
    return pd.DataFrame({"well_id":[f"CB_{i:05d}" for i in range(1,n+1)],"easting":e,"northing":no,"longitude":lon,"latitude":lat,
        "dem_m":dem,"distance_coast_m":coast,"geology":geo,"geology_factor":gf,"ndvi_mean":nd,"ndvi_anomaly":nda,
        "rainfall_mm":rain,"et_mm":et,"surface_water_distance_m":sw,"pressure_hpa":p,"year":year,"season":season,
        "groundwater_level_mAHD":gw})

@st.cache_resource
def fit(data):
    f=["longitude","latitude","dem_m","distance_coast_m","geology_factor","ndvi_mean","ndvi_anomaly",
       "rainfall_mm","et_mm","surface_water_distance_m","pressure_hpa","year"]
    Xtr,Xte,ytr,yte=train_test_split(data[f],data.groundwater_level_mAHD,test_size=.2,random_state=42)
    m=RandomForestRegressor(n_estimators=500,max_depth=15,min_samples_leaf=3,random_state=42,n_jobs=-1).fit(Xtr,ytr)
    out=data.copy(); out["rf_predicted_mAHD"]=m.predict(data[f]); out["residual_m"]=out.groundwater_level_mAHD-out.rf_predicted_mAHD; out["abs_error_m"]=out.residual_m.abs()
    met={"R²":r2_score(yte,m.predict(Xte)),"MAE (m)":mean_absolute_error(yte,m.predict(Xte)),"RMSE (m)":mean_squared_error(yte,m.predict(Xte))**.5}
    imp=pd.DataFrame({"Feature":f,"Importance":m.feature_importances_}).sort_values("Importance",ascending=False)
    return m,out,met,imp

data=make_data(); model,res,metrics,imp=fit(data)
st.title("💧 Coffin Bay Groundwater Intelligence")
st.caption("Synthetic research prototype • Random Forest • spatial groundwater exploration")

with st.sidebar:
    view=st.radio("Dashboard",["Overview","Piezometric map","Model drivers","Well explorer","Diagnostics","Scenario lab"])
    geos=st.multiselect("Geology",sorted(res.geology.unique()),sorted(res.geology.unique()))
    seasons=st.multiselect("Season",sorted(res.season.unique()),sorted(res.season.unique()))
    st.warning("Synthetic data only. Do not use this prototype to claim the real Coffin Bay groundwater relationship.")

d=res[res.geology.isin(geos)&res.season.isin(seasons)]

if view=="Overview":
    a,b,c,d1=st.columns(4); a.metric("Wells",f"{len(res):,}"); b.metric("R²",f"{metrics['R²']:.3f}"); c.metric("MAE",f"{metrics['MAE (m)']:.2f} m"); d1.metric("RMSE",f"{metrics['RMSE (m)']:.2f} m")
    st.subheader("Topography versus groundwater")
    st.plotly_chart(px.scatter(d,x="dem_m",y="groundwater_level_mAHD",color="groundwater_level_mAHD",hover_name="well_id",hover_data=["geology","distance_coast_m","rf_predicted_mAHD"],labels={"dem_m":"DEM elevation (m)","groundwater_level_mAHD":"Groundwater level (m AHD)"}),use_container_width=True)
    st.subheader("Predicted groundwater across the study area")
    st.plotly_chart(px.scatter(d,x="easting",y="northing",color="rf_predicted_mAHD",hover_name="well_id",hover_data=["dem_m","geology","abs_error_m"]),use_container_width=True)

elif view=="Piezometric map":
    v=st.selectbox("Map variable",["RF predicted groundwater","Observed synthetic groundwater","Absolute prediction error"])
    col={"RF predicted groundwater":"rf_predicted_mAHD","Observed synthetic groundwater":"groundwater_level_mAHD","Absolute prediction error":"abs_error_m"}[v]
    fig=px.scatter_mapbox(d,lat="latitude",lon="longitude",color=col,hover_name="well_id",hover_data=["dem_m","geology","distance_coast_m"],zoom=9,height=650,mapbox_style="open-street-map")
    st.plotly_chart(fig,use_container_width=True)
    st.info("This map shows point predictions. A continuous piezometric surface should be generated using a validated spatial reconstruction method.")

elif view=="Model drivers":
    st.subheader("Random Forest feature importance")
    st.caption("Importance is predictive association, not causal evidence.")
    fig=px.bar(imp,x="Importance",y="Feature",orientation="h"); fig.update_layout(yaxis={"categoryorder":"total ascending"},height=600)
    st.plotly_chart(fig,use_container_width=True)
    feat=st.selectbox("Inspect predictor",imp.Feature.tolist())
    st.plotly_chart(px.scatter(res,x=feat,y="rf_predicted_mAHD",color="geology",hover_name="well_id",trendline="lowess"),use_container_width=True)

elif view=="Well explorer":
    w=st.selectbox("Well",res.well_id.tolist()); r=res[res.well_id==w].iloc[0]
    a,b,c,e=st.columns(4); a.metric("Observed",f"{r.groundwater_level_mAHD:.2f} m AHD"); b.metric("RF prediction",f"{r.rf_predicted_mAHD:.2f} m AHD"); c.metric("Residual",f"{r.residual_m:.2f} m"); e.metric("DEM",f"{r.dem_m:.2f} m")
    st.write(pd.DataFrame({"Variable":["Geology","Coast distance","NDVI","Rainfall","ET","Surface-water distance","Pressure","Year","Season"],
        "Value":[r.geology,f"{r.distance_coast_m:,.0f} m",f"{r.ndvi_mean:.2f}",f"{r.rainfall_mm:.0f} mm",f"{r.et_mm:.0f} mm",f"{r.surface_water_distance_m:,.0f} m",f"{r.pressure_hpa:.1f} hPa",int(r.year),r.season]}))

elif view=="Diagnostics":
    fig=px.scatter(res,x="groundwater_level_mAHD",y="rf_predicted_mAHD",color="abs_error_m",hover_name="well_id",labels={"groundwater_level_mAHD":"Observed (m AHD)","rf_predicted_mAHD":"Predicted (m AHD)"})
    lo=min(res.groundwater_level_mAHD.min(),res.rf_predicted_mAHD.min()); hi=max(res.groundwater_level_mAHD.max(),res.rf_predicted_mAHD.max()); fig.add_shape(type="line",x0=lo,y0=lo,x1=hi,y1=hi)
    st.plotly_chart(fig,use_container_width=True)
    st.plotly_chart(px.histogram(res,x="residual_m",nbins=35,title="Residual distribution"),use_container_width=True)
    st.dataframe(res.sort_values("abs_error_m",ascending=False)[["well_id","geology","dem_m","distance_coast_m","groundwater_level_mAHD","rf_predicted_mAHD","residual_m","abs_error_m"]].head(50),use_container_width=True)

else:
    st.subheader("Scenario lab")
    st.warning("Exploratory synthetic experiment — not a validated groundwater forecast.")
    base=res.iloc[0]; c1,c2,c3=st.columns(3)
    dem= c1.slider("DEM (m)",0.,35.,float(base.dem_m),.5); rain=c2.slider("Rainfall (mm)",250,800,int(base.rainfall_mm),10); nd=c3.slider("NDVI",.1,.95,float(base.ndvi_mean),.01)
    coast=c1.slider("Distance to coast (m)",20,12000,int(base.distance_coast_m),100); et=c2.slider("ET (mm)",700,1400,int(base.et_mm),10); sw=c3.slider("Surface-water distance (m)",20,10000,int(base.surface_water_distance_m),100)
    row=base.copy(); row["dem_m"]=dem; row["rainfall_mm"]=rain; row["ndvi_mean"]=nd; row["distance_coast_m"]=coast; row["et_mm"]=et; row["surface_water_distance_m"]=sw
    f=["longitude","latitude","dem_m","distance_coast_m","geology_factor","ndvi_mean","ndvi_anomaly","rainfall_mm","et_mm","surface_water_distance_m","pressure_hpa","year"]
    pred=float(model.predict(pd.DataFrame([row[f]]))[0]); st.metric("Scenario prediction",f"{pred:.2f} m AHD")

st.divider()
st.caption("Research prototype: synthetic predictors → Random Forest → diagnostics → spatial exploration. Replace synthetic inputs with validated observations before scientific interpretation.")
