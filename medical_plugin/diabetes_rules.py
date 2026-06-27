import numpy as np
import pandas as pd


def _find_matching_column(columns, tokens):
    for column in columns:
        lower = column.lower()
        if any(token in lower for token in tokens):
            return column
    return None


def _clinical_columns(df):
    columns = df.columns.tolist()
    return {
        "glucose": _find_matching_column(columns, ["glucose", "blood_glucose", "gluc"]),
        "bmi": _find_matching_column(columns, ["bmi", "body_mass_index"]),
        "age": _find_matching_column(columns, ["age"]),
        "insulin": _find_matching_column(columns, ["insulin"]),
        "blood_pressure": _find_matching_column(columns, ["bloodpressure", "blood_pressure", "bp", "pressure"]),
        "pregnancies": _find_matching_column(columns, ["pregnan"]),
    }


def engineer_medical_features(df):
    engineered = df.copy()
    clinical = _clinical_columns(engineered)
    feature_notes = []

    if clinical["glucose"] and clinical["bmi"]:
        engineered["glucose_bmi_interaction"] = engineered[clinical["glucose"]].fillna(0) * engineered[clinical["bmi"]].fillna(0)
        feature_notes.append("glucose_bmi_interaction")

    if clinical["age"] and clinical["bmi"]:
        engineered["age_bmi_interaction"] = engineered[clinical["age"]].fillna(0) * engineered[clinical["bmi"]].fillna(0)
        feature_notes.append("age_bmi_interaction")

    if clinical["glucose"]:
        engineered["glucose_high_flag"] = (engineered[clinical["glucose"]] >= 140).astype(float)
        engineered["glucose_very_high_flag"] = (engineered[clinical["glucose"]] >= 180).astype(float)
        engineered["glucose_band"] = pd.cut(
            engineered[clinical["glucose"]],
            bins=[-np.inf, 99, 125, 180, np.inf],
            labels=["normal", "elevated", "high", "very_high"],
        ).astype(str)
        feature_notes.extend(["glucose_high_flag", "glucose_very_high_flag", "glucose_band"])

    if clinical["bmi"]:
        engineered["bmi_overweight_flag"] = (engineered[clinical["bmi"]] >= 25).astype(float)
        engineered["bmi_obesity_flag"] = (engineered[clinical["bmi"]] >= 30).astype(float)
        engineered["bmi_band"] = pd.cut(
            engineered[clinical["bmi"]],
            bins=[-np.inf, 18.5, 25, 30, np.inf],
            labels=["underweight", "normal", "overweight", "obese"],
        ).astype(str)
        feature_notes.extend(["bmi_overweight_flag", "bmi_obesity_flag", "bmi_band"])

    if clinical["age"]:
        engineered["age_senior_flag"] = (engineered[clinical["age"]] >= 50).astype(float)
        engineered["age_band"] = pd.cut(
            engineered[clinical["age"]],
            bins=[-np.inf, 30, 45, 60, np.inf],
            labels=["young", "midlife", "older", "senior"],
        ).astype(str)
        feature_notes.extend(["age_senior_flag", "age_band"])

    if clinical["insulin"]:
        engineered["insulin_high_flag"] = (engineered[clinical["insulin"]] >= 150).astype(float)
        feature_notes.append("insulin_high_flag")

    if clinical["blood_pressure"]:
        engineered["blood_pressure_high_flag"] = (engineered[clinical["blood_pressure"]] >= 80).astype(float)
        feature_notes.append("blood_pressure_high_flag")

    clinical_risk_index = np.zeros(len(engineered), dtype=float)
    weight_total = 0.0

    if clinical["glucose"]:
        clinical_risk_index += engineered[clinical["glucose"]].fillna(engineered[clinical["glucose"]].median()).astype(float) * 0.35
        weight_total += 0.35

    if clinical["bmi"]:
        clinical_risk_index += engineered[clinical["bmi"]].fillna(engineered[clinical["bmi"]].median()).astype(float) * 0.25
        weight_total += 0.25

    if clinical["age"]:
        clinical_risk_index += engineered[clinical["age"]].fillna(engineered[clinical["age"]].median()).astype(float) * 0.15
        weight_total += 0.15

    if clinical["insulin"]:
        clinical_risk_index += engineered[clinical["insulin"]].fillna(engineered[clinical["insulin"]].median()).astype(float) * 0.15
        weight_total += 0.15

    if clinical["blood_pressure"]:
        clinical_risk_index += engineered[clinical["blood_pressure"]].fillna(engineered[clinical["blood_pressure"]].median()).astype(float) * 0.10
        weight_total += 0.10

    if weight_total > 0:
        clinical_risk_index = clinical_risk_index / weight_total
        engineered["clinical_risk_index"] = pd.Series(clinical_risk_index, index=engineered.index)
        feature_notes.append("clinical_risk_index")

    datetime_cols = engineered.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    for column in datetime_cols:
        engineered[f"{column}_year"] = engineered[column].dt.year
        engineered[f"{column}_month"] = engineered[column].dt.month
        engineered[f"{column}_dayofweek"] = engineered[column].dt.dayofweek
        feature_notes.extend([f"{column}_year", f"{column}_month", f"{column}_dayofweek"])

    return engineered, {
        "clinical_columns": clinical,
        "engineered_features": feature_notes,
        "row_count": int(len(engineered)),
        "column_count": int(engineered.shape[1]),
    }


def build_surrogate_diabetes_target(df):
    clinical = _clinical_columns(df)
    score = np.zeros(len(df), dtype=float)
    evidence = 0.0

    if clinical["glucose"]:
        score += (df[clinical["glucose"]].fillna(df[clinical["glucose"]].median()).astype(float) >= 140).astype(float) * 0.45
        score += (df[clinical["glucose"]].fillna(df[clinical["glucose"]].median()).astype(float) >= 180).astype(float) * 0.15
        evidence += 0.60

    if clinical["bmi"]:
        score += (df[clinical["bmi"]].fillna(df[clinical["bmi"]].median()).astype(float) >= 30).astype(float) * 0.20
        evidence += 0.20

    if clinical["age"]:
        score += (df[clinical["age"]].fillna(df[clinical["age"]].median()).astype(float) >= 50).astype(float) * 0.10
        evidence += 0.10

    if clinical["insulin"]:
        score += (df[clinical["insulin"]].fillna(df[clinical["insulin"]].median()).astype(float) >= 150).astype(float) * 0.05
        evidence += 0.05

    if clinical["blood_pressure"]:
        score += (df[clinical["blood_pressure"]].fillna(df[clinical["blood_pressure"]].median()).astype(float) >= 80).astype(float) * 0.05
        evidence += 0.05

    if evidence == 0:
        numeric = df.select_dtypes(include=[np.number])
        if numeric.empty:
            surrogate = pd.Series([0] * len(df), index=df.index, dtype=int)
            label_source = "fallback_constant"
        else:
            rolling = numeric.mean(axis=1)
            threshold = rolling.median()
            surrogate = (rolling >= threshold).astype(int)
            label_source = "median_numeric_proxy"
        return surrogate, {
            "target_name": "future_diabetes_likelihood",
            "source": label_source,
            "positive_rate": float(surrogate.mean()),
        }

    if score.max() == score.min():
        surrogate = pd.Series([0] * len(df), index=df.index, dtype=int)
    else:
        threshold = np.nanpercentile(score, 65)
        surrogate = (score >= threshold).astype(int)

    if surrogate.nunique(dropna=True) < 2:
        surrogate = (score >= np.nanmedian(score)).astype(int)

    return surrogate.astype(int), {
        "target_name": "future_diabetes_likelihood",
        "source": "clinical_surrogate",
        "positive_rate": float(np.mean(surrogate)),
    }


def clinical_risk_assessment(df):
    risk = {
        "glucose_risk": 0.0,
        "bmi_risk": 0.0,
        "age_risk": 0.0,
        "overall_risk_level": "Low",
        "risk_score": 0.0,
    }

    glucose_col = next((c for c in df.columns if "glucose" in c.lower()), None)
    if glucose_col:
        risk["glucose_risk"] = float((df[glucose_col] > 140).mean())

    bmi_col = next((c for c in df.columns if "bmi" in c.lower()), None)
    if bmi_col:
        risk["bmi_risk"] = float((df[bmi_col] > 30).mean())

    age_col = next((c for c in df.columns if "age" in c.lower()), None)
    if age_col:
        risk["age_risk"] = float((df[age_col] > 50).mean())

    score = (
        risk["glucose_risk"] * 0.5
        + risk["bmi_risk"] * 0.3
        + risk["age_risk"] * 0.2
    )

    risk["risk_score"] = float(score)

    if score > 0.6:
        risk["overall_risk_level"] = "High"
    elif score > 0.3:
        risk["overall_risk_level"] = "Medium"

    return risk
