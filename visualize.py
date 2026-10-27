"""All charts. Har function figure ko outputs/ mein save karta hai aur fig return karta hai."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from . import analysis as an

sns.set_theme(style="whitegrid", palette="deep")
OUT = "outputs"


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.png", dpi=130)
    return fig


def crime_types(df):
    s = an.by_crime_type(df)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=s.values, y=s.index, ax=ax)
    ax.set(title="Crimes by Type", xlabel="Number of crimes", ylabel="")
    return _save(fig, "01_crime_types")


def districts(df):
    s = an.by_district(df)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=s.values, y=s.index, ax=ax, color="#c0392b")
    ax.set(title="Crimes by District", xlabel="Number of crimes", ylabel="")
    return _save(fig, "02_districts")


def monthly(df, forecast=None):
    m = an.monthly_trend(df)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(m.index, m.values, marker="o", label="Actual")
    if forecast is not None:
        ax.plot(forecast.index, forecast.values, "r--o", label="Forecast")
    ax.set(title="Monthly Crime Trend", xlabel="", ylabel="Crimes")
    ax.legend()
    return _save(fig, "03_monthly_trend")


def hourly(df):
    s = an.by_hour(df)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(x=s.index, y=s.values, ax=ax, color="#2c3e50")
    ax.set(title="Crimes by Hour of Day", xlabel="Hour (0-23)", ylabel="Crimes")
    return _save(fig, "04_hourly")


def heatmap_day_hour(df):
    p = df.pivot_table(index="day_of_week", columns="hour", values="crime_id", aggfunc="count")
    p = p.reindex(an.DAY_ORDER)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    sns.heatmap(p, cmap="YlOrRd", ax=ax)
    ax.set(title="Crime Heatmap: Weekday x Hour", xlabel="Hour", ylabel="")
    return _save(fig, "05_heatmap_day_hour")


def heatmap_district_type(df):
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.heatmap(an.crime_vs_district(df), annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set(title="District vs Crime Type", xlabel="", ylabel="")
    return _save(fig, "06_district_vs_type")


def arrest_rates(df):
    s = an.arrest_rate_by_type(df)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=s.values, y=s.index, ax=ax, color="#27ae60")
    ax.set(title="Arrest Rate by Crime Type (%)", xlabel="%", ylabel="")
    return _save(fig, "07_arrest_rates")


def hotspot_map(df_clustered, table):
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.scatterplot(data=df_clustered.sample(min(6000, len(df_clustered)), random_state=1),
                    x="longitude", y="latitude", hue="hotspot", palette="tab10", s=8, alpha=0.5, ax=ax)
    ax.scatter(table.lon, table.lat, c="black", marker="X", s=180, label="Hotspot centre")
    ax.set(title="Crime Hotspots (KMeans)", xlabel="Longitude", ylabel="Latitude")
    ax.legend(loc="upper left", fontsize=8)
    return _save(fig, "08_hotspots")


def feature_importance(imp):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=imp.values, y=imp.index, ax=ax, color="#8e44ad")
    ax.set(title="Top Factors Influencing Arrest", xlabel="Importance", ylabel="")
    return _save(fig, "09_feature_importance")


def victims(df):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    an.victim_profile(df).plot(kind="bar", ax=ax, rot=0)
    ax.set(title="Victim Profile: Age Group x Gender", xlabel="Age group", ylabel="Victims")
    return _save(fig, "10_victims")
