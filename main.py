"""Crime Data Analysis System - full pipeline.

Run:  python main.py                  (synthetic data auto-generate hoga)
      python main.py --data my.csv    (apna dataset)
"""
import argparse
import os

import pandas as pd

from src import analysis as an, models, preprocess as pp, visualize as viz
from src.data_generator import generate

pd.set_option("display.width", 120)


def banner(t):
    print(f"\n{'=' * 60}\n  {t}\n{'=' * 60}")


def main(path):
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    if not os.path.exists(path):
        print(f"{path} nahi mila -> synthetic dataset bana raha hoon...")
        generate().to_csv(path, index=False)

    banner("1. LOAD & CLEAN")
    raw = pp.load(path)
    df, rep = pp.clean(raw)
    df = pp.add_features(df)
    for k, v in rep.items():
        print(f"  {k:<24}: {v}")
    df.to_csv("data/crime_data_clean.csv", index=False)

    banner("2. SUMMARY")
    for k, v in an.summary(df).items():
        print(f"  {k:<22}: {v}")

    banner("3. ANALYSIS")
    print("Top crime types:\n", an.by_crime_type(df).head(5).to_string())
    print("\nDistrict risk ranking:\n", an.risk_ranking(df).to_string())
    print("\nArrest rate by type (%):\n", an.arrest_rate_by_type(df).to_string())

    banner("4. MACHINE LEARNING")
    clustered, hot = models.find_hotspots(df, k=5)
    print("Hotspots:\n", hot.to_string())
    res = models.train_arrest_model(df)
    print(f"\nArrest prediction accuracy: {res['accuracy']:.1%}")
    print(res["report"])
    fc, r2 = models.forecast_monthly(an.monthly_trend(df), horizon=6)
    print(f"Forecast (next 6 months, R2={r2:.2f}):\n", fc.astype(int).to_string())

    banner("5. CHARTS")
    viz.crime_types(df); viz.districts(df); viz.monthly(df, fc); viz.hourly(df)
    viz.heatmap_day_hour(df); viz.heatmap_district_type(df); viz.arrest_rates(df)
    viz.hotspot_map(clustered, hot); viz.feature_importance(res["importance"]); viz.victims(df)
    print("  10 charts saved in outputs/")

    with open("outputs/report.txt", "w") as f:
        f.write("CRIME DATA ANALYSIS REPORT\n\n")
        for k, v in an.summary(df).items():
            f.write(f"{k}: {v}\n")
        f.write(f"\nRisk ranking:\n{an.risk_ranking(df).to_string()}\n")
        f.write(f"\nHotspots:\n{hot.to_string()}\n")
        f.write(f"\nArrest model accuracy: {res['accuracy']:.1%}\n{res['report']}\n")
        f.write(f"\nForecast:\n{fc.astype(int).to_string()}\n")
    print("  report saved: outputs/report.txt\n\nDONE.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/crime_data.csv")
    main(ap.parse_args().data)
