import numpy as np
import pandas as pd
import warnings
import os
import pickle
import json
import hashlib

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor

try:
    from medical_plugin.diabetes_rules import clinical_risk_assessment
except Exception:
    def clinical_risk_assessment(_df):
        return {
            "glucose_risk": 0.0,
            "bmi_risk": 0.0,
            "age_risk": 0.0,
            "overall_risk_level": "Low",
            "risk_score": 0.0,
        }

warnings.filterwarnings("ignore")

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
GENERAL_MODEL_CACHE_PATH = os.path.join(MODEL_DIR, "general_predictive_models.pkl")
STRICT_REUSE_IF_MODEL_EXISTS = True


def _load_model_cache():
    if not os.path.exists(GENERAL_MODEL_CACHE_PATH):
        return {}
    try:
        with open(GENERAL_MODEL_CACHE_PATH, "rb") as cache_file:
            data = pickle.load(cache_file)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_model_cache(cache):
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(GENERAL_MODEL_CACHE_PATH, "wb") as cache_file:
            pickle.dump(cache, cache_file)
    except Exception:
        pass


def _align_features_for_inference(frame, feature_columns):
    if not feature_columns:
        return frame.copy()
    return frame.reindex(columns=feature_columns, fill_value=np.nan)


def _compute_dataset_signature(frame, target, task):
    schema = [
        {
            "name": str(column),
            "dtype": str(frame[column].dtype),
        }
        for column in frame.columns
    ]
    payload = {
        "target": str(target),
        "task": str(task),
        "schema": schema,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


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
    model_cache = _load_model_cache()
    cache_updated = False

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
            dataset_signature = _compute_dataset_signature(X, target, "regression")

            cache_key = f"regression::{target}::{dataset_signature}"
            cached_entry = model_cache.get(cache_key, {})
            cached_model = cached_entry.get("pipeline") if isinstance(cached_entry, dict) else None
            if cached_model is not None:
                try:
                    aligned_X = _align_features_for_inference(X, cached_entry.get("feature_columns", []))
                    sample_preds = cached_model.predict(aligned_X.head(5))
                    clinical_risk = clinical_risk_assessment(clean_df)

                    results[target] = {
                        "task": "regression",
                        "best_model": cached_entry.get("best_model", "PersistedModel"),
                        "r2_score": None,
                        "sample_predictions": [float(x) for x in sample_preds],
                        "clinical_risk": clinical_risk,
                        "model_reused": True,
                        "interpretation": {
                            "meaning": "Prediction reflects metabolic/clinical progression",
                            "risk_level": clinical_risk["overall_risk_level"]
                        }
                    }
                    continue
                except Exception as error:
                    if STRICT_REUSE_IF_MODEL_EXISTS:
                        results[target] = {
                            "task": "regression",
                            "error": f"Saved model exists but reuse failed: {error}. Retraining is disabled in strict reuse mode.",
                            "model_reused": True,
                        }
                        continue

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
            clinical_risk = clinical_risk_assessment(clean_df)

            results[target] = {
                "task": "regression",
                "best_model": best_name,
                "r2_score": round(float(best_score), 4),
                "sample_predictions": [float(x) for x in sample_preds],
                "model_reused": False,

                # 🧠 NEW CLINICAL OUTPUT
                "clinical_risk": clinical_risk,

                "interpretation": {
                    "meaning": "Prediction reflects metabolic/clinical progression",
                    "risk_level": clinical_risk["overall_risk_level"]
                }
            }

            model_cache[cache_key] = {
                "pipeline": best_model,
                "best_model": best_name,
                "feature_columns": X.columns.tolist(),
                "dataset_signature": dataset_signature,
            }
            cache_updated = True

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
            dataset_signature = _compute_dataset_signature(X, target, "classification")

            cache_key = f"classification::{target}::{dataset_signature}"
            cached_entry = model_cache.get(cache_key, {})
            cached_model = cached_entry.get("pipeline") if isinstance(cached_entry, dict) else None
            if cached_model is not None:
                try:
                    aligned_X = _align_features_for_inference(X, cached_entry.get("feature_columns", []))
                    sample_preds = cached_model.predict(aligned_X.head(5))
                    clinical_risk = clinical_risk_assessment(clean_df)

                    results[target] = {
                        "task": "classification",
                        "best_model": cached_entry.get("best_model", "PersistedModel"),
                        "accuracy": None,
                        "sample_predictions": [str(x) for x in sample_preds],
                        "clinical_risk": clinical_risk,
                        "model_reused": True,
                    }
                    continue
                except Exception as error:
                    if STRICT_REUSE_IF_MODEL_EXISTS:
                        results[target] = {
                            "task": "classification",
                            "error": f"Saved model exists but reuse failed: {error}. Retraining is disabled in strict reuse mode.",
                            "model_reused": True,
                        }
                        continue

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

            clinical_risk = clinical_risk_assessment(clean_df)

            results[target] = {
                "task": "classification",
                "best_model": best_name,
                "accuracy": round(float(best_score), 4),
                "sample_predictions": [str(x) for x in sample_preds],
                "model_reused": False,

                # 🧠 Clinical context
                "clinical_risk": clinical_risk
            }

            model_cache[cache_key] = {
                "pipeline": best_model,
                "best_model": best_name,
                "feature_columns": X.columns.tolist(),
                "dataset_signature": dataset_signature,
            }
            cache_updated = True

        except Exception as e:
            results[target] = {"error": str(e)}

    if cache_updated:
        _save_model_cache(model_cache)

    return results