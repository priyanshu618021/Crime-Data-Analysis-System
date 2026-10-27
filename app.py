"""Interactive dashboard.  Run:  streamlit run app.py"""
import os

import pandas as pd
import streamlit as st

from src import analysis as an, models, preprocess as pp
from src.data_generator import generate

st.set_page_config(page_title="Crime Data Analysis System", page_icon="🚔", layout="wide")


@st.cache_data
def get_data():
    path = "data/crime_data.csv"
    if not os.path.exists(path):
        os.makedirs("data", exist_ok=True)
        generate().to_csv(path, index=False)
    df, _ = pp.clean(pp.load(path))
    return pp.add_features(df)


df = get_data()

# ---------- sidebar filters ----------
st.sidebar.header("Filters")
years = st.sidebar.multiselect("Year", sorted(df.year.unique()), default=sorted(df.year.unique()))
dists = st.sidebar.multiselect("District", sorted(df.district.unique()), default=sorted(df.district.unique()))
types = st.sidebar.multiselect("Crime type", sorted(df.crime_type.unique()), default=sorted(df.crime_type.unique()))
f = df[df.year.isin(years) & df.district.isin(dists) & df.crime_type.isin(types)]

st.title("🚔 Crime Data Analysis System")
if f.empty:
    st.warning("Filters ke hisaab se koi data nahi mila.")
    st.stop()

# ---------- KPI row ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total crimes", f"{len(f):,}")
c2.metric("Most common", f.crime_type.mode()[0])
c3.metric("Riskiest district", an.risk_ranking(f).index[0])
c4.metric("Arrest rate", f"{f.arrest.mean():.1%}")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🗺️ Hotspots", "⏰ Time patterns", "🤖 Prediction"])

with tab1:
    a, b = st.columns(2)
    a.subheader("Crimes by type"); a.bar_chart(an.by_crime_type(f))
    b.subheader("Crimes by district"); b.bar_chart(an.by_district(f))
    st.subheader("Monthly trend"); st.line_chart(an.monthly_trend(f))
    st.subheader("District risk ranking"); st.dataframe(an.risk_ranking(f), use_container_width=True)
    st.subheader("Arrest rate by crime type (%)"); st.bar_chart(an.arrest_rate_by_type(f))

with tab2:
    k = st.slider("Number of hotspots", 2, 10, 5)
    clustered, table = models.find_hotspots(f, k=k)
    st.map(f.sample(min(5000, len(f)), random_state=1)[["latitude", "longitude"]])
    st.subheader("Hotspot summary"); st.dataframe(table, use_container_width=True)

with tab3:
    a, b = st.columns(2)
    a.subheader("By hour"); a.bar_chart(an.by_hour(f))
    b.subheader("By weekday"); b.bar_chart(an.by_weekday(f))
    st.subheader("Crime type vs time of day"); st.dataframe(an.crime_vs_time(f), use_container_width=True)

with tab4:
    st.subheader("Forecast (next 6 months)")
    fc, r2 = models.forecast_monthly(an.monthly_trend(df), 6)
    st.line_chart(pd.concat([an.monthly_trend(df), fc.rename("crimes")]))
    st.caption(f"Model R² = {r2:.2f}")

    st.subheader("Arrest predictor")
    res = models.train_arrest_model(df) if "m" not in st.session_state else st.session_state["m"]
    st.session_state["m"] = res
    st.write(f"Model accuracy: **{res['accuracy']:.1%}**")
    x1, x2, x3 = st.columns(3)
    ct = x1.selectbox("Crime type", sorted(df.crime_type.unique()))
    di = x2.selectbox("District", sorted(df.district.unique()))
    lo = x3.selectbox("Location", sorted(df.location_type.unique()))
    y1, y2, y3 = st.columns(3)
    hr = y1.slider("Hour", 0, 23, 21)
    ag = y2.slider("Victim age", 5, 90, 30)
    ge = y3.selectbox("Victim gender", sorted(df.victim_gender.unique()))
    if st.button("Predict"):
        row = pd.DataFrame([{c: 0 for c in res["columns"]}])
        for col, val in [("hour", hr), ("victim_age", ag), ("month", 6), ("is_weekend", 0),
                         (f"crime_type_{ct}", 1), (f"district_{di}", 1),
                         (f"location_type_{lo}", 1), (f"victim_gender_{ge}", 1)]:
            if col in row.columns:
                row[col] = val
        p = res["model"].predict_proba(row)[0][1]
        st.success(f"Arrest probability: {p:.0%}")
