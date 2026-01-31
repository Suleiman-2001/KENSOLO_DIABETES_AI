# engines/predictive_engine.py
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBRegressor, XGBClassifier
import warnings

warnings.filterwarnings("ignore")


def _prepare_features(df, target):
    y = df[target]
    X = df.drop(columns=[target])

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if X[c].dtype != "object"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ]
    )
    return X, y, preprocessor


def run_predictive_model(df, targets_dict):
    """
    Returns predictions dictionary.
    - Keeps the best_model_pipeline in memory for SHAP
    - Does NOT break JSON serialization
    """
    results = {}

    # -------- REGRESSION ----------
    for target in targets_dict.get("numerical", []):
        try:
            if target not in df.columns:
                continue

            clean_df = df.dropna(subset=[target])
            if clean_df.shape[0] < 20:
                results[target] = {"error": "Not enough data rows for regression"}
                continue

            X, y, preprocessor = _prepare_features(clean_df, target)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            models = {
                "LinearRegression": LinearRegression(),
                "RandomForestRegressor": RandomForestRegressor(n_estimators=150, random_state=42),
                "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=150, random_state=42),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "XGBRegressor": XGBRegressor(n_estimators=150, random_state=42, verbosity=0)
            }

            best_score = -999
            best_model = None
            best_name = None
            model_scores = {}

            for name, model in models.items():
                pipe = Pipeline([("prep", preprocessor), ("model", model)])
                pipe.fit(X_train, y_train)
                preds = pipe.predict(X_test)
                r2 = r2_score(y_test, preds)
                model_scores[name] = round(float(r2), 4)
                if r2 > best_score:
                    best_score = r2
                    best_model = pipe
                    best_name = name

            sample_preds = best_model.predict(X.head(5))

            # Store pipeline only in memory (not for JSON)
            results[target] = {
                "task": "regression",
                "best_model": best_name,
                "best_model_pipeline": best_model,  # used in memory for Why engine
                "r2_score": round(float(best_score), 4),
                "sample_predictions": [float(x) for x in sample_preds],
                "all_model_scores": model_scores
            }

        except Exception as e:
            results[target] = {"error": str(e)}

    # -------- CLASSIFICATION ----------
    for target in targets_dict.get("categorical", []):
        try:
            if target not in df.columns:
                continue

            clean_df = df.dropna(subset=[target])
            if clean_df.shape[0] < 30:
                results[target] = {"error": "Not enough data rows for classification"}
                continue

            nunique = clean_df[target].nunique()
            if nunique < 2 or nunique > 20:
                results[target] = {"error": "Target has too few or too many categories"}
                continue

            X, y, preprocessor = _prepare_features(clean_df, target)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            models = {
                "LogisticRegression": LogisticRegression(max_iter=2000),
                "RandomForestClassifier": RandomForestClassifier(n_estimators=200, random_state=42),
                "GradientBoostingClassifier": GradientBoostingClassifier(n_estimators=150, random_state=42),
                "XGBClassifier": XGBClassifier(n_estimators=150, random_state=42, use_label_encoder=False, eval_metric='logloss'),
                "KNeighborsClassifier": KNeighborsClassifier(),
                "SVC": SVC(probability=True)
            }

            best_score = -999
            best_model = None
            best_name = None
            model_scores = {}

            for name, model in models.items():
                pipe = Pipeline([("prep", preprocessor), ("model", model)])
                pipe.fit(X_train, y_train)
                preds = pipe.predict(X_test)
                acc = accuracy_score(y_test, preds)
                model_scores[name] = round(float(acc), 4)
                if acc > best_score:
                    best_score = acc
                    best_model = pipe
                    best_name = name

            sample_preds = best_model.predict(X.head(5))

            results[target] = {
                "task": "classification",
                "best_model": best_name,
                "best_model_pipeline": best_model,  # in-memory only
                "accuracy": round(float(best_score), 4),
                "sample_predictions": [str(x) for x in sample_preds],
                "classes": list(map(str, sorted(clean_df[target].unique()))),
                "all_model_scores": model_scores
            }

        except Exception as e:
            results[target] = {"error": str(e)}

    return results
