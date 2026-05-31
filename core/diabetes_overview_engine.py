import pandas as pd

def dataset_overview(df: pd.DataFrame):
    """
    Diabetes Clinical Dataset Overview Engine
    Provides structured patient dataset interpretation.
    """

    df = df.copy()

    # -----------------------------
    # Normalize column names
    # -----------------------------
    df.columns = [
        c.lower().replace(" ", "").replace("-", "_")
        for c in df.columns
    ]

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    # -----------------------------
    # Clinical feature detection
    # -----------------------------
    clinical_keywords = [
        "glucose",
        "bmi",
        "insulin",
        "age",
        "bloodpressure",
        "pregnancies",
        "skin",
        "diabetes"
    ]

    clinical_cols = [
        col for col in df.columns
        if any(k in col for k in clinical_keywords)
    ]

    # -----------------------------
    # Feature categorization (NEW)
    # -----------------------------
    risk_features = [c for c in clinical_cols if "glucose" in c or "bmi" in c or "insulin" in c]
    demographic_features = [c for c in clinical_cols if "age" in c or "pregnancies" in c]
    outcome_features = [c for c in clinical_cols if "diabetes" in c]

    # -----------------------------
    # Dataset structure
    # -----------------------------
    overview = {
        "total_patients": int(df.shape[0]),
        "total_features": int(df.shape[1]),

        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,

        "clinical_features": clinical_cols,
        "risk_features": risk_features,
        "demographic_features": demographic_features,
        "outcome_features": outcome_features,

        "non_clinical_features": [
            col for col in df.columns if col not in clinical_cols
        ]
    }

    # -----------------------------
    # Data quality layer (medical-aware)
    # -----------------------------
    missing_rate = df.isnull().sum().sum() / max(1, df.size)
    duplicate_rate = df.duplicated().mean()

    constant_cols = [c for c in df.columns if df[c].nunique() <= 1]

    overview["data_health"] = {
        "missing_rate": round(missing_rate, 4),
        "duplicate_rate": round(duplicate_rate, 4),
        "constant_columns": constant_cols,
        "completeness_score": round(1 - missing_rate, 4),
        "risk_of_data_quality_issue": missing_rate > 0.1 or duplicate_rate > 0.05
    }

    # -----------------------------
    # Clinical dataset classification (UPGRADED)
    # -----------------------------
    if len(risk_features) > 0:
        dataset_type = "High-Risk Diabetes Clinical Dataset"
    elif len(clinical_cols) > 0:
        dataset_type = "Diabetes-Related Clinical Dataset"
    else:
        dataset_type = "General Dataset (Non-clinical)"

    overview["dataset_type"] = dataset_type

    # -----------------------------
    # Clinical maturity score (NEW)
    # -----------------------------
    clinical_score = 0

    clinical_score += len(risk_features) * 20
    clinical_score += len(demographic_features) * 10
    clinical_score += len(outcome_features) * 30

    clinical_score = min(100, clinical_score)

    if clinical_score < 30:
        maturity = "Low Clinical Relevance"
    elif clinical_score < 70:
        maturity = "Moderate Clinical Dataset"
    else:
        maturity = "High Clinical Dataset (Diabetes Ready)"

    overview["clinical_maturity"] = {
        "score": clinical_score,
        "level": maturity
    }

    return overview