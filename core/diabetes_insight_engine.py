import pandas as pd
import numpy as np

def auto_insights(df: pd.DataFrame):
    """
    Diabetes Clinical Insight Engine
    Detects metabolic risk patterns and clinical anomalies.
    """

    df = df.copy()

    # -----------------------------
    # Normalize column names
    # -----------------------------
    df.columns = [c.lower().strip() for c in df.columns]

    insights = []

    # -----------------------------
    # Missing data analysis
    # -----------------------------
    missing = df.isnull().sum().sum()

    if missing == 0:
        insights.append("No missing clinical records detected")
    else:
        insights.append(f"{missing} missing values detected in dataset")

    # -----------------------------
    # Clinical thresholds
    # -----------------------------
    GLUCOSE_HIGH = 140
    GLUCOSE_CRITICAL = 180

    BMI_OVERWEIGHT = 25
    BMI_OBESE = 30

    BP_HIGH = 140

    numeric_cols = df.select_dtypes(include="number").columns

    # -----------------------------
    # Column-level risk detection
    # -----------------------------
    for col in numeric_cols:
        series = df[col].dropna()
        col_name = col.lower()

        # ---------------- GLUCOSE ----------------
        if "glucose" in col_name:
            high = (series > GLUCOSE_HIGH).sum()
            critical = (series > GLUCOSE_CRITICAL).sum()

            if high > 0:
                insights.append(f"{high} elevated glucose cases (>140 mg/dL)")
            if critical > 0:
                insights.append(f"{critical} critical hyperglycemia cases (>180 mg/dL)")

        # ---------------- BMI ----------------
        if "bmi" in col_name:
            overweight = (series > BMI_OVERWEIGHT).sum()
            obese = (series > BMI_OBESE).sum()

            if overweight > 0:
                insights.append(f"{overweight} overweight BMI cases detected")
            if obese > 0:
                insights.append(f"{obese} obesity-level BMI cases detected")

        # ---------------- BLOOD PRESSURE ----------------
        if "bp" in col_name or "bloodpressure" in col_name:
            high_bp = (series > BP_HIGH).sum()
            if high_bp > 0:
                insights.append(f"{high_bp} high blood pressure readings detected")

        # ---------------- VARIABILITY ----------------
        mean_val = series.mean()
        std_val = series.std()

        if mean_val != 0 and std_val / mean_val > 0.5:
            insights.append(f"High variability detected in {col}")

    # -----------------------------
    # Metabolic syndrome detection
    # -----------------------------
    if "glucose" in df.columns and "bmi" in df.columns:
        metabolic_risk = (
            (df["glucose"] > GLUCOSE_HIGH) &
            (df["bmi"] > BMI_OBESE)
        ).sum()

        if metabolic_risk > 0:
            insights.append(
                f"{metabolic_risk} metabolic syndrome risk cases detected (glucose + BMI)"
            )

    # -----------------------------
    # Final clinical summary layer (NEW)
    # -----------------------------
    if len(insights) <= 2:
        insights.append("Low clinical anomaly presence detected")

    return {
        "insights": insights,
        "total_insights": len(insights)
    }