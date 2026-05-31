import pandas as pd
import numpy as np

def detect_kpis(df: pd.DataFrame):
    """
    Diabetes Clinical KPI Engine
    Converts dataset into structured clinical risk indicators.
    """

    df = df.copy()

    # -----------------------------
    # Normalize column names
    # -----------------------------
    df.columns = [c.lower().strip() for c in df.columns]

    kpi_keywords = [
        "glucose",
        "bmi",
        "insulin",
        "bloodpressure",
        "age",
        "pregnancies",
        "skin"
    ]

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # -----------------------------
    # Clinical KPI selection
    # -----------------------------
    kpi_candidates = [
        col for col in numeric_cols
        if any(k in col for k in kpi_keywords)
    ]

    if not kpi_candidates:
        kpi_candidates = numeric_cols

    # -----------------------------
    # Feature-level KPIs
    # -----------------------------
    kpi_summary = {}

    for col in kpi_candidates:
        series = df[col].dropna()

        mean_val = float(series.mean())
        std_val = float(series.std())

        kpi_summary[col] = {
            "mean": mean_val,
            "std": std_val,
            "min": float(series.min()),
            "max": float(series.max()),

            # Clinical variability risk (important in diabetes)
            "high_variability_risk": bool(std_val > mean_val * 0.5 if mean_val != 0 else False),

            # Outlier burden (important for glucose spikes)
            "outlier_rate": float((series > (mean_val + 2 * std_val)).mean())
        }

    # -----------------------------
    # Global Clinical KPIs
    # -----------------------------
    global_kpis = {}

    def safe_col(name):
        return name.lower() if name.lower() in df.columns else None

    glucose = safe_col("glucose")
    bmi = safe_col("bmi")
    age = safe_col("age")

    if glucose:
        global_kpis["avg_glucose"] = float(df[glucose].mean())
        global_kpis["hyperglycemia_rate"] = float((df[glucose] > 140).mean())
        global_kpis["severe_hyperglycemia_rate"] = float((df[glucose] > 180).mean())

    if bmi:
        global_kpis["avg_bmi"] = float(df[bmi].mean())
        global_kpis["obesity_rate"] = float((df[bmi] > 30).mean())
        global_kpis["severe_obesity_rate"] = float((df[bmi] > 35).mean())

    if age:
        global_kpis["avg_age"] = float(df[age].mean())

    # -----------------------------
    # Clinical Risk Score (NEW)
    # -----------------------------
    risk_score = 0

    if glucose:
        risk_score += (df[glucose] > 140).mean() * 40
        risk_score += (df[glucose] > 180).mean() * 60

    if bmi:
        risk_score += (df[bmi] > 30).mean() * 30

    risk_score = min(100, round(risk_score, 2))

    if risk_score < 30:
        risk_level = "Low Risk 🟢"
    elif risk_score < 60:
        risk_level = "Moderate Risk 🟡"
    else:
        risk_level = "High Risk 🔴"

    return {
        "feature_kpis": kpi_summary,
        "global_kpis": global_kpis,
        "clinical_risk_score": risk_score,
        "risk_level": risk_level
    }

    