# engines/predictive_engine.py
import numpy as np
import pandas as pd
import warnings
from joblib import Parallel, delayed

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

warnings.filterwarnings("ignore")


def _prepare_features(df, target, sparse=True, max_categories=20):
    """
    Prepare features for modeling:
    - OneHotEncode categorical columns safely
    - Limit one-hot encoding for high-cardinality columns
    """
    y = df[target]
    X = df.drop(columns=[target])
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if np.issubdtype(X[c].dtype, np.number)]

    # Limit high-cardinality categorical columns by top categories
    limited_cat_cols = []
    for col in cat_cols:
        top_vals = df[col].value_counts().nlargest(max_categories).index
        X[col] = np.where(X[col].isin(top_vals), X[col], "Other")
        limited_cat_cols.append(col)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=sparse), limited_cat_cols)
        ],
        remainder='drop'
    )
    return X, y, preprocessor


def _fit_regression(name, model, X_train, y_train, X_test, y_test, preprocessor):
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    return name, pipe, preds


def _fit_classification(name, model, X_train, y_train, X_test, y_test, preprocessor):
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    return name, pipe, preds


def run_predictive_model(df, targets_dict, max_jobs=2, min_rows_regression=20, min_rows_classification=30):
    """
    Safe and optimized predictive engine:
    - Sparse encoding for high-cardinality features
    - Controlled parallelization (max_jobs)
    - Sequential fallback for small datasets
    - NaN-safe and small-sample-safe
    - Limits high-cardinality categories automatically
    """
    results = {}

    # ---------------- REGRESSION -----------------
    for target in targets_dict.get("numerical", []):
        try:
            if target not in df.columns:
                results[target] = {"error": "Target not in dataframe"}
                continue

            clean_df = df.dropna(subset=[target])
            if clean_df.shape[0] < min_rows_regression:
                results[target] = {"error": f"Not enough data rows for regression (<{min_rows_regression})"}
                continue

            X, y, preprocessor = _prepare_features(clean_df, target, sparse=True)
            if len(X) < 2:
                results[target] = {"error": "Insufficient rows after preprocessing"}
                continue

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            models = {
                "LinearRegression": LinearRegression(),
                "RandomForestRegressor": RandomForestRegressor(n_estimators=50, random_state=42),
                "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=50, random_state=42),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "XGBRegressor": XGBRegressor(n_estimators=50, random_state=42, verbosity=0)
            }

            use_parallel = len(models) >= 3 and len(X_train) >= min_rows_regression
            if use_parallel:
                trained_models = Parallel(n_jobs=max_jobs)(
                    delayed(_fit_regression)(name, model, X_train, y_train, X_test, y_test, preprocessor)
                    for name, model in models.items()
                )
            else:
                trained_models = [_fit_regression(name, model, X_train, y_train, X_test, y_test, preprocessor)
                                  for name, model in models.items()]

            best_score = -np.inf
            best_model = None
            best_name = None
            model_scores = {}

            for name, pipe, preds in trained_models:
                r2 = r2_score(y_test, preds)
                model_scores[name] = round(float(r2), 4)
                if r2 > best_score:
                    best_score = r2
                    best_model = pipe
                    best_name = name

            sample_preds = best_model.predict(X.head(min(5, len(X))))
            results[target] = {
                "task": "regression",
                "best_model": best_name,
                "best_model_pipeline": best_model,
                "r2_score": round(float(best_score), 4),
                "sample_predictions": [float(x) for x in sample_preds],
                "all_model_scores": model_scores
            }

        except Exception as e:
            results[target] = {"error": str(e)}

    # ---------------- CLASSIFICATION -----------------
    for target in targets_dict.get("categorical", []):
        try:
            if target not in df.columns:
                results[target] = {"error": "Target not in dataframe"}
                continue

            clean_df = df.dropna(subset=[target])
            if clean_df.shape[0] < min_rows_classification:
                results[target] = {"error": f"Not enough data rows for classification (<{min_rows_classification})"}
                continue

            nunique = clean_df[target].nunique()
            if nunique < 2 or nunique > 20:
                results[target] = {"error": "Target has too few or too many categories"}
                continue

            X, y, preprocessor = _prepare_features(clean_df, target, sparse=True)
            stratify_param = y if len(np.unique(y)) > 1 else None
            if len(X) < 2:
                results[target] = {"error": "Insufficient rows after preprocessing"}
                continue

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=stratify_param
            )

            models = {
                "LogisticRegression": LogisticRegression(max_iter=2000),
                "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42),
                "GradientBoostingClassifier": GradientBoostingClassifier(n_estimators=50, random_state=42),
                "XGBClassifier": XGBClassifier(n_estimators=50, random_state=42, use_label_encoder=False, eval_metric='logloss'),
                "KNeighborsClassifier": KNeighborsClassifier(),
                "SVC": SVC(probability=True)
            }

            use_parallel = len(models) >= 3 and len(X_train) >= min_rows_classification
            if use_parallel:
                trained_models = Parallel(n_jobs=max_jobs)(
                    delayed(_fit_classification)(name, model, X_train, y_train, X_test, y_test, preprocessor)
                    for name, model in models.items()
                )
            else:
                trained_models = [_fit_classification(name, model, X_train, y_train, X_test, y_test, preprocessor)
                                  for name, model in models.items()]

            best_score = -np.inf
            best_model = None
            best_name = None
            model_scores = {}

            for name, pipe, preds in trained_models:
                acc = accuracy_score(y_test, preds)
                model_scores[name] = round(float(acc), 4)
                if acc > best_score:
                    best_score = acc
                    best_model = pipe
                    best_name = name

            sample_preds = best_model.predict(X.head(min(5, len(X))))
            results[target] = {
                "task": "classification",
                "best_model": best_name,
                "best_model_pipeline": best_model,
                "accuracy": round(float(best_score), 4),
                "sample_predictions": [str(x) for x in sample_preds],
                "classes": list(map(str, sorted(clean_df[target].unique()))),
                "all_model_scores": model_scores
            }

        except Exception as e:
            results[target] = {"error": str(e)}

    return results