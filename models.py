"""Machine-learning part: hotspots, arrest prediction, crime forecast."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def find_hotspots(df, k=5, seed=42):
    """KMeans on (lat, lon) -> k crime hotspots. Returns (df_with_cluster, hotspot_table)."""
    X = df[["latitude", "longitude"]].values
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    out = df.copy()
    out["hotspot"] = km.labels_
    table = (out.groupby("hotspot")
             .agg(crimes=("crime_id", "size"),
                  lat=("latitude", "mean"), lon=("longitude", "mean"),
                  top_crime=("crime_type", lambda s: s.mode()[0]),
                  top_district=("district", lambda s: s.mode()[0]))
             .sort_values("crimes", ascending=False).round(4))
    return out, table


def train_arrest_model(df, seed=42):
    """Predict whether a case ends in an arrest. Returns dict with model + metrics."""
    cols = ["crime_type", "district", "location_type", "victim_gender",
            "hour", "victim_age", "month", "is_weekend"]
    X = pd.get_dummies(df[cols], columns=["crime_type", "district", "location_type", "victim_gender"])
    y = df["arrest"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    model = RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced",
                                   n_jobs=-1, random_state=seed).fit(X_tr, y_tr)
    pred = model.predict(X_te)
    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    return {
        "model": model, "columns": list(X.columns),
        "accuracy": accuracy_score(y_te, pred),
        "report": classification_report(y_te, pred, target_names=["No arrest", "Arrest"]),
        "confusion": confusion_matrix(y_te, pred),
        "importance": imp.head(10),
    }


def forecast_monthly(monthly, horizon=6):
    """Linear trend + month-of-year seasonality forecast on monthly counts."""
    s = monthly.copy()
    t = np.arange(len(s))
    month = s.index.month.values
    X = np.column_stack([t] + [(month == m).astype(int) for m in range(2, 13)])
    lr = LinearRegression().fit(X, s.values)

    f_idx = pd.date_range(s.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    ft = np.arange(len(s), len(s) + horizon)
    fm = f_idx.month.values
    Xf = np.column_stack([ft] + [(fm == m).astype(int) for m in range(2, 13)])
    pred = pd.Series(lr.predict(Xf).round(0), index=f_idx, name="forecast")
    return pred, lr.score(X, s.values)
