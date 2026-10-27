"""Statistical analysis functions. Har function ek DataFrame/Series return karta hai."""
import pandas as pd

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
VIOLENT = ["Assault", "Robbery", "Homicide"]


def summary(df):
    return {
        "Total crimes": len(df),
        "Date range": f"{df.date.min().date()} to {df.date.max().date()}",
        "Most common crime": df.crime_type.mode()[0],
        "Most affected district": df.district.mode()[0],
        "Peak hour": int(df.hour.mode()[0]),
        "Overall arrest rate": f"{df.arrest.mean():.1%}",
        "Avg victim age": round(df.victim_age.mean(), 1),
    }


def by_crime_type(df):
    return df.crime_type.value_counts()


def by_district(df):
    return df.district.value_counts()


def monthly_trend(df):
    return df.set_index("date").resample("MS").size().rename("crimes")


def by_hour(df):
    return df.groupby("hour").size()


def by_weekday(df):
    return df.groupby("day_of_week").size().reindex(DAY_ORDER)


def crime_vs_district(df):
    return pd.crosstab(df.district, df.crime_type)


def crime_vs_time(df):
    return pd.crosstab(df.crime_type, df.time_of_day)


def arrest_rate_by_type(df):
    return (df.groupby("crime_type").arrest.mean() * 100).sort_values(ascending=False).round(1)


def victim_profile(df):
    return pd.crosstab(df.age_group, df.victim_gender)


def risk_ranking(df):
    """Risk score = 60% crime volume + 40% violent-crime volume (0-100)."""
    g = df.groupby("district").agg(total=("crime_id", "size"),
                                   violent=("crime_type", lambda s: s.isin(VIOLENT).sum()))
    g["risk_score"] = 0.6 * g.total / g.total.max() * 100 + 0.4 * g.violent / g.violent.max() * 100
    return g.sort_values("risk_score", ascending=False).round(1)
