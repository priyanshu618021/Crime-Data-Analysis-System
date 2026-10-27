"""Data loading + cleaning + feature engineering."""
import pandas as pd

REQUIRED = ["date", "hour", "crime_type", "district", "latitude", "longitude",
            "location_type", "victim_age", "victim_gender", "arrest"]


def load(path="data/crime_data.csv"):
    df = pd.read_csv(path, parse_dates=["date"])
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"CSV mein ye columns missing hain: {missing}")
    if "crime_id" not in df.columns:
        df.insert(0, "crime_id", range(1, len(df) + 1))
    return df


def clean(df):
    """Returns (clean_df, report_dict)."""
    report = {"rows_raw": len(df)}
    df = df.copy()

    # text standardisation first (so duplicates / categories match properly)
    for c in ["crime_type", "district", "location_type", "victim_gender"]:
        df[c] = df[c].astype(str).str.strip().str.title()

    subset = [c for c in df.columns if c != "crime_id"]
    report["duplicates_removed"] = int(df.duplicated(subset=subset).sum())
    df = df.drop_duplicates(subset=subset)

    report["missing_values_fixed"] = {k: int(v) for k, v in df.isna().sum().items() if v > 0}
    df["victim_age"] = df["victim_age"].fillna(df["victim_age"].median())
    for c in ["latitude", "longitude"]:
        df[c] = df[c].fillna(df.groupby("district")[c].transform("median"))
    df = df.dropna(subset=["date", "crime_type", "district"])

    before = len(df)
    df = df[df["hour"].between(0, 23) & df["victim_age"].between(0, 110)]
    report["invalid_rows_removed"] = before - len(df)
    df["arrest"] = df["arrest"].astype(int)

    report["rows_clean"] = len(df)
    return df.reset_index(drop=True), report


def add_features(df):
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.day_name()
    df["is_weekend"] = df["date"].dt.dayofweek.isin([5, 6]).astype(int)
    df["time_of_day"] = pd.cut(df["hour"], [-1, 5, 11, 17, 21, 23],
                               labels=["Night", "Morning", "Afternoon", "Evening", "Late Night"])
    df["age_group"] = pd.cut(df["victim_age"], [-1, 17, 30, 45, 60, 120],
                             labels=["<18", "18-30", "31-45", "46-60", "60+"])
    return df
