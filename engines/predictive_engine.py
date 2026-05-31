import numpy as np
import pandas as pd
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


# =========================================================
# 🧠 CLINICAL RISK ENGINE
# =========================================================
def _clinical_risk_assessment(df):
    """
    Converts dataset into diabetes risk indicators.
    """

    risk = {
        "glucose_risk": 0.0,
        "bmi_risk": 0.0,
        "age_risk": 0.0,
        "overall_risk_level": "Low",
        "risk_score": 0.0
    }

    # Detect glucose
    glucose_col = next((c for c in df.columns if "glucose" in c.lower()), None)
    if glucose_col:
        risk["glucose_risk"] = float((df[glucose_col] > 140).mean())

    # Detect BMI
    bmi_col = next((c for c in df.columns if "bmi" in c.lower()), None)
    if bmi_col:
        risk["bmi_risk"] = float((df[bmi_col] > 30).mean())

    # Detect Age
    age_col = next((c for c in df.columns if "age" in c.lower()), None)
    if age_col:
        risk["age_risk"] = float((df[age_col] > 50).mean())

    # Weighted risk score
    score = (
        risk["glucose_risk"] * 0.5 +
        risk["bmi_risk"] * 0.3 +
        risk["age_risk"] * 0.2
    )

    risk["risk_score"] = float(score)

    if score > 0.6:
        risk["overall_risk_level"] = "High"
    elif score > 0.3:
        risk["overall_risk_level"] = "Medium"

    return risk


# =========================================================
# ⚙️ FEATURE PREPARATION
# =========================================================
def _prepare_features(df, target):
    y = df[target]
    X = df.drop(columns=[target])

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if np.issubdtype(X[c].dtype, np.number)]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ]
    )

    return X, y, preprocessor


# =========================================================
# 🤖 MODEL TRAINING
# =========================================================
def _fit_model(name, model, X_train, y_train, X_test, y_test, preprocessor, task):
    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    score = r2_score(y_test, preds)

    return name, pipe, score


# =========================================================
# 🚀 MAIN PREDICTIVE ENGINE
# =========================================================
def run_predictive_model(df, targets_dict):

    results = {}

    # -------------------------
    # REGRESSION TARGETS
    # -------------------------
    for target in targets_dict.get("numerical", []):

        try:
            if target not in df.columns:
                results[target] = {"error": "Target not found"}
                continue

            clean_df = df.dropna(subset=[target])

            if len(clean_df) < 20:
                results[target] = {"error": "Insufficient data"}
                continue

            X, y, preprocessor = _prepare_features(clean_df, target)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            models = {
                "LinearRegression": LinearRegression(),
                "RandomForest": RandomForestRegressor(n_estimators=50, random_state=42),
                "XGB": XGBRegressor(n_estimators=50, verbosity=0)
            }

            best_score = -np.inf
            best_model = None
            best_name = None

            for name, model in models.items():
                name, pipe, score = _fit_model(
                    name, model,
                    X_train, y_train,
                    X_test, y_test,
                    preprocessor,
                    "regression"
                )

                if score > best_score:
                    best_score = score
                    best_model = pipe
                    best_name = name

            sample_preds = best_model.predict(X.head(5))

            # 🔬 Clinical layer
            clinical_risk = _clinical_risk_assessment(clean_df)

            results[target] = {
                "task": "regression",
                "best_model": best_name,
                "r2_score": round(float(best_score), 4),
                "sample_predictions": [float(x) for x in sample_preds],

                # 🧠 NEW CLINICAL OUTPUT
                "clinical_risk": clinical_risk,

                "interpretation": {
                    "meaning": "Prediction reflects metabolic/clinical progression",
                    "risk_level": clinical_risk["overall_risk_level"]
                }
            }

        except Exception as e:
            results[target] = {"error": str(e)}

    # -------------------------
    # CLASSIFICATION TARGETS
    # -------------------------
    for target in targets_dict.get("categorical", []):

        try:
            if target not in df.columns:
                results[target] = {"error": "Target not found"}
                continue

            clean_df = df.dropna(subset=[target])

            if clean_df[target].nunique() < 2:
                results[target] = {"error": "Not enough classes"}
                continue

            X, y, preprocessor = _prepare_features(clean_df, target)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            models = {
                "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42)
            }

            best_score = -np.inf
            best_model = None
            best_name = None

            for name, model in models.items():
                pipe = Pipeline([("prep", preprocessor), ("model", model)])
                pipe.fit(X_train, y_train)
                preds = pipe.predict(X_test)

                score = accuracy_score(y_test, preds)

                if score > best_score:
                    best_score = score
                    best_model = pipe
                    best_name = name

            sample_preds = best_model.predict(X.head(5))

            clinical_risk = _clinical_risk_assessment(clean_df)

            results[target] = {
                "task": "classification",
                "best_model": best_name,
                "accuracy": round(float(best_score), 4),
                "sample_predictions": [str(x) for x in sample_preds],

                # 🧠 Clinical context
                "clinical_risk": clinical_risk
            }

        except Exception as e:
            results[target] = {"error": str(e)}

    return results