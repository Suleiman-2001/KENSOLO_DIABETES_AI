import pandas as pd

def data_quality_score(df: pd.DataFrame):
    """
    Clinical Data Quality Engine for Diabetes AI System
    Evaluates dataset safety for medical decision support.
    """

    df = df.copy()

    total_missing = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()

    constant_cols = [c for c in df.columns if df[c].nunique() <= 1]

    total_cells = df.size if df.size > 0 else 1
    total_rows = len(df)

    # -----------------------------
    # Base score
    # -----------------------------
    score = 100

    # -----------------------------
    # Missing data penalty (weighted clinical risk)
    # -----------------------------
    missing_ratio = total_missing / total_cells

    if missing_ratio > 0.2:
        score -= 50
    elif missing_ratio > 0.1:
        score -= 30
    else:
        score -= missing_ratio * 100

    # -----------------------------
    # Duplicate penalty (scaled properly)
    # -----------------------------
    duplicate_ratio = duplicate_rows / max(1, total_rows)

    if duplicate_ratio > 0.1:
        score -= 30
    else:
        score -= duplicate_ratio * 100

    # -----------------------------
    # Constant feature penalty
    # -----------------------------
    score -= len(constant_cols) * 5

    # -----------------------------
    # Clinical feature awareness (NEW)
    # -----------------------------
    clinical_keywords = ["glucose", "bmi", "insulin", "bloodpressure", "age"]

    clinical_cols = [
        c for c in df.columns
        if any(k in c.lower() for k in clinical_keywords)
    ]

    # Penalize missing clinical structure
    if len(clinical_cols) == 0:
        score -= 25

    score = max(0, min(100, int(score)))

    # -----------------------------
    # Clinical classification
    # -----------------------------
    if score >= 85:
        status = "Clinically Safe Dataset 🟢"
        risk_level = "Low"
    elif score >= 60:
        status = "Moderate Clinical Risk 🟡"
        risk_level = "Medium"
    else:
        status = "Unsafe for Clinical Use 🔴"
        risk_level = "High"

    # -----------------------------
    # Clinical risk breakdown
    # -----------------------------
    clinical_risks = {
        "missing_data_risk": "High" if missing_ratio > 0.1 else "Low",
        "duplicate_bias_risk": "High" if duplicate_ratio > 0.05 else "Low",
        "feature_irrelevance_risk": "High" if len(constant_cols) > 0 else "Low",
        "clinical_feature_gap": "High" if len(clinical_cols) < 2 else "Low"
    }

    return {
        "data_quality_score": score,
        "clinical_status": status,
        "data_risk_level": risk_level,

        "metrics": {
            "missing_ratio": round(missing_ratio, 4),
            "duplicate_ratio": round(duplicate_ratio, 4),
            "constant_features": constant_cols,
            "total_rows": total_rows
        },

        "clinical_risks": clinical_risks
    }