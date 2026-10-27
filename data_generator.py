"""Realistic synthetic crime dataset generator.

Agar tumhare paas real dataset hai (Kaggle / data.gov.in / Chicago / NCRB),
to isko skip karo aur CSV ko data/crime_data.csv naam se rakh do.
Column names same rakhna: see README.
"""
import numpy as np
import pandas as pd

CRIME_TYPES = {
    "Theft": 0.28, "Burglary": 0.12, "Assault": 0.14, "Robbery": 0.08,
    "Vehicle Theft": 0.10, "Fraud": 0.10, "Drug Offense": 0.08,
    "Vandalism": 0.06, "Homicide": 0.01, "Cyber Crime": 0.03,
}
ARREST_RATE = {
    "Theft": 0.18, "Burglary": 0.15, "Assault": 0.45, "Robbery": 0.30,
    "Vehicle Theft": 0.12, "Fraud": 0.22, "Drug Offense": 0.70,
    "Vandalism": 0.10, "Homicide": 0.75, "Cyber Crime": 0.15,
}
# district, lat, lon, relative crime weight
DISTRICTS = [
    ("Central", 28.6139, 77.2090, 1.6), ("North", 28.7041, 77.1025, 1.0),
    ("South", 28.5355, 77.2100, 1.1), ("East", 28.6280, 77.3000, 0.9),
    ("West", 28.6500, 77.0500, 1.2), ("Market Area", 28.6560, 77.2300, 1.8),
    ("Industrial Zone", 28.5800, 77.3300, 0.8), ("Suburb", 28.4600, 77.0300, 0.5),
]
LOCATIONS = ["Street", "Residence", "Market", "Public Transport",
             "Parking Lot", "Office", "Online", "Park"]


def generate(n=20000, seed=42):
    rng = np.random.default_rng(seed)
    p = np.array(list(CRIME_TYPES.values()))
    types = rng.choice(list(CRIME_TYPES), n, p=p / p.sum())
    w = np.array([d[3] for d in DISTRICTS]); w = w / w.sum()
    didx = rng.choice(len(DISTRICTS), n, p=w)

    days = rng.integers(0, 365 * 3, n)
    dates = pd.Timestamp("2022-01-01") + pd.to_timedelta(days, unit="D")
    # mild seasonality + upward trend
    keep_p = np.clip(0.85 + 0.15 * np.sin(2 * np.pi * (dates.month.values - 3) / 12) + 0.0001 * days, 0, 1)
    keep = rng.random(n) < keep_p

    night = np.isin(types, ["Robbery", "Assault", "Burglary", "Drug Offense"])
    hours = np.where(night, rng.normal(22, 4, n), rng.normal(14, 5, n))
    hours = (np.round(hours).astype(int)) % 24

    df = pd.DataFrame({
        "crime_id": np.arange(1, n + 1),
        "date": dates,
        "hour": hours,
        "crime_type": types,
        "district": [DISTRICTS[i][0] for i in didx],
        "latitude": [DISTRICTS[i][1] + rng.normal(0, 0.02) for i in didx],
        "longitude": [DISTRICTS[i][2] + rng.normal(0, 0.02) for i in didx],
        "location_type": rng.choice(LOCATIONS, n),
        "victim_age": np.clip(rng.normal(34, 13, n), 5, 90).astype(float).round(),
        "victim_gender": rng.choice(["Male", "Female", "Other"], n, p=[0.52, 0.46, 0.02]),
    })
    df["arrest"] = [int(rng.random() < ARREST_RATE[t] * (1.15 if h < 6 else 1.0))
                    for t, h in zip(df.crime_type, df.hour)]
    df = df[keep].reset_index(drop=True)

    # inject dirt so cleaning step has real work
    df.loc[df.sample(frac=0.02, random_state=1).index, "district"] = "  central "
    df.loc[df.sample(frac=0.015, random_state=2).index, "victim_age"] = np.nan
    df.loc[df.sample(frac=0.01, random_state=3).index, "latitude"] = np.nan
    df = pd.concat([df, df.sample(60, random_state=4)], ignore_index=True)  # duplicates
    return df.sample(frac=1, random_state=5).reset_index(drop=True)


if __name__ == "__main__":
    out = generate()
    out.to_csv("data/crime_data.csv", index=False)
    print(f"Generated {len(out)} rows -> data/crime_data.csv")
