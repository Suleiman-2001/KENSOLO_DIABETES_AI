import warnings
from copy import deepcopy

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except Exception:
    XGBClassifier = None
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier

    LIGHTGBM_AVAILABLE = True
except Exception:
    LGBMClassifier = None
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier

    CATBOOST_AVAILABLE = True
except Exception:
    CatBoostClassifier = None
    CATBOOST_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    SMOTE_AVAILABLE = True
except Exception:
    SMOTE = None
    ImbPipeline = None
    SMOTE_AVAILABLE = False

try:
    import optuna

    OPTUNA_AVAILABLE = True
except Exception:
    optuna = None
    OPTUNA_AVAILABLE = False

try:
    import shap

    SHAP_AVAILABLE = True
except Exception:
    shap = None
    SHAP_AVAILABLE = False

try:
    from engines.memory_engine import track_dataset_history
except Exception:
    track_dataset_history = None

try:
    from medical_plugin.diabetes_rules import (
        build_surrogate_diabetes_target,
        engineer_medical_features,
    )
except Exception:
    build_surrogate_diabetes_target = None
    engineer_medical_features = None


POSITIVE_TOKENS = {
    "1",
    "yes",
    "y",
    "true",
    "positive",
    "diabetes",
    "diabetic",
    "has diabetes",
    "with diabetes",
    "sick",
}

NEGATIVE_TOKENS = {
    "0",
    "no",
    "n",
    "false",
    "negative",
    "non-diabetic",
    "nondiabetic",
    "healthy",
    "control",
    "without diabetes",
}


def _safe_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _normalize_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _detect_diabetes_targets(df):
    explicit_names = {
        "outcome",
        "diabetes",
        "diabetes_outcome",
        "target",
        "label",
        "class",
        "diagnosis",
        "has_diabetes",
        "dm",
    }

    keyword_priority = {
        "outcome": 0,
        "diabetes": 1,
        "diagnosis": 2,
        "target": 3,
        "label": 4,
        "class": 5,
    }

    candidates = []
    for col in df.columns:
        low = str(col).lower().strip()
        if low == "id" or low.endswith("id") or "_id" in low:
            continue

        series = df[col]
        nunique = int(series.nunique(dropna=True))
        if nunique < 2 or nunique > 12:
            continue

        if pd.api.types.is_datetime64_any_dtype(series):
            continue

        is_explicit = low in explicit_names
        has_keyword = any(k in low for k in ("diabet", "outcome", "diagnos", "target", "label", "class"))
        if not is_explicit and not has_keyword:
            continue

        score = keyword_priority.get(low, 10)
        if has_keyword and "diabet" in low:
            score -= 1
        score += max(0, nunique - 2) * 0.1
        candidates.append((score, col))

    candidates.sort(key=lambda item: item[0])
    return [col for _, col in candidates]


def _select_preferred_diabetes_target(candidates):
    if not candidates:
        return []

    return [candidates[0]]


def _engineer_features(df):
    if callable(engineer_medical_features):
        return engineer_medical_features(df)

    return df.copy(), {
        "clinical_columns": {},
        "engineered_features": [],
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
    }


def _build_surrogate_diabetes_target(df):
    if callable(build_surrogate_diabetes_target):
        try:
            surrogate, metadata = build_surrogate_diabetes_target(df)
            surrogate_series = pd.Series(surrogate, index=df.index)
            surrogate_binary = pd.to_numeric(surrogate_series, errors="coerce").fillna(0)
            surrogate_binary = (surrogate_binary >= surrogate_binary.median()).astype(int)
            if surrogate_binary.nunique() >= 2:
                metadata = metadata or {}
                metadata.setdefault("target_name", "future_diabetes_likelihood")
                metadata.setdefault("source", "plugin_surrogate")
                metadata["positive_rate"] = float(surrogate_binary.mean())
                return surrogate_binary, metadata
        except Exception:
            pass

    feature_rules = [
        ("glucose", ["glucose", "plasma", "fasting_glucose"], 0.30),
        ("hba1c", ["hba1c", "a1c", "glycated"], 0.22),
        ("bmi", ["bmi", "body_mass", "body mass"], 0.16),
        ("age", ["age"], 0.12),
        ("insulin", ["insulin"], 0.10),
        ("blood_pressure", ["blood_pressure", "bp", "pressure", "hypertension"], 0.10),
    ]

    used_features = []
    weighted_components = []
    total_weight = 0.0

    for feature_name, tokens, weight in feature_rules:
        matched_col = None
        for col in df.columns:
            low = str(col).lower()
            if any(token in low for token in tokens):
                matched_col = col
                break

        if matched_col is None:
            continue

        numeric = pd.to_numeric(df[matched_col], errors="coerce")
        if numeric.notna().mean() < 0.4 or numeric.nunique(dropna=True) < 3:
            continue

        rank_score = numeric.rank(pct=True).fillna(0.5)
        weighted_components.append(rank_score * weight)
        total_weight += weight
        used_features.append(matched_col)

    if weighted_components and total_weight > 0:
        combined_score = sum(weighted_components) / total_weight
    else:
        numeric_df = df.select_dtypes(include=[np.number]).copy()
        numeric_df = numeric_df.loc[:, numeric_df.nunique(dropna=True) >= 3]
        if numeric_df.shape[1] >= 1:
            ranked = numeric_df.rank(pct=True)
            combined_score = ranked.mean(axis=1).fillna(0.5)
            used_features = numeric_df.columns.tolist()[:5]
        else:
            combined_score = pd.Series(np.linspace(0.0, 1.0, num=max(1, len(df))), index=df.index)

    threshold = float(combined_score.quantile(0.65)) if len(combined_score) > 0 else 0.5
    surrogate = (combined_score >= threshold).astype(int)

    if surrogate.nunique() < 2 and len(combined_score) > 0:
        median_threshold = float(combined_score.median())
        surrogate = (combined_score >= median_threshold).astype(int)

    if surrogate.nunique() < 2:
        surrogate = pd.Series((np.arange(len(df)) % 2).astype(int), index=df.index)

    return surrogate.astype(int), {
        "target_name": "future_diabetes_likelihood",
        "source": "surrogate_rule_engine",
        "positive_rate": float(surrogate.mean()),
        "features_used": used_features,
    }


def _normalize_target_values(series):
    values = series.copy()
    text_values = values.map(_normalize_text)
    mapped = text_values.map(lambda value: 1 if value in POSITIVE_TOKENS else 0 if value in NEGATIVE_TOKENS else np.nan)

    if mapped.notna().sum() >= max(2, int(0.6 * len(mapped))):
        if mapped.isna().any():
            fill_value = int(mapped.dropna().mode().iloc[0]) if not mapped.dropna().empty else 0
            mapped = mapped.fillna(fill_value)
        return mapped.astype(int), {"strategy": "token_map", "positive_token_rate": float(mapped.mean())}

    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.dropna().nunique() <= 2:
            if numeric.dropna().nunique() == 2:
                return (numeric.fillna(numeric.median()) > numeric.dropna().median()).astype(int), {"strategy": "numeric_binary_threshold"}
            return numeric.fillna(0).astype(int), {"strategy": "numeric_binary_passthrough"}

        threshold = numeric.dropna().median()
        return (numeric.fillna(threshold) >= threshold).astype(int), {"strategy": "numeric_median_split"}

    labels = values.astype(str).fillna("unknown")
    uniques = labels.nunique(dropna=True)
    if uniques <= 2:
        ordered = labels.dropna().unique().tolist()
        mapping = {ordered[0]: 0, ordered[1]: 1} if len(ordered) == 2 else {ordered[0]: 0}
        normalized = labels.map(mapping).fillna(0).astype(int)
        return normalized, {"strategy": "binary_label_encoding", "mapping": mapping}

    positive_mask = labels.str.contains("diabet|positive|yes|true|1", case=False, na=False)
    negative_mask = labels.str.contains("non|healthy|negative|no|false|0", case=False, na=False)
    if positive_mask.sum() > 0 and negative_mask.sum() > 0:
        normalized = positive_mask.astype(int)
        return normalized, {"strategy": "keyword_presence"}

    return (labels.astype("category").cat.codes > labels.astype("category").cat.codes.median()).astype(int), {"strategy": "category_median_split"}


def _build_preprocessor(frame):
    numeric_columns = []
    categorical_columns = []

    for column in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[column]) or pd.api.types.is_datetime64_any_dtype(frame[column]):
            numeric_columns.append(column)
        elif pd.api.types.is_object_dtype(frame[column]) or pd.api.types.is_categorical_dtype(frame[column]):
            if frame[column].nunique(dropna=True) <= 30:
                categorical_columns.append(column)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", _safe_one_hot_encoder()),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return preprocessor, {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "dropped_high_cardinality_columns": [
            column
            for column in frame.columns
            if column not in numeric_columns and column not in categorical_columns
        ],
    }


def _build_candidate_factories(imbalance_ratio):
    candidates = []

    candidates.append(
        {
            "name": "LogisticRegression",
            "builder": lambda params=None: LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                solver="lbfgs",
                **(params or {}),
            ),
            "sampler": lambda trial: {
                "C": trial.suggest_float("C", 0.05, 10.0, log=True),
            },
        }
    )

    candidates.append(
        {
            "name": "RandomForest",
            "builder": lambda params=None: RandomForestClassifier(
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
                **(params or {}),
            ),
            "sampler": lambda trial: {
                "n_estimators": trial.suggest_int("n_estimators", 150, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 18),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
            },
        }
    )

    candidates.append(
        {
            "name": "DecisionTree",
            "builder": lambda params=None: DecisionTreeClassifier(
                random_state=42,
                class_weight="balanced",
                **(params or {}),
            ),
            "sampler": lambda trial: {
                "max_depth": trial.suggest_int("max_depth", 2, 12),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            },
        }
    )

    if XGBOOST_AVAILABLE:
        candidates.append(
            {
                "name": "XGBoost",
                "builder": lambda params=None: XGBClassifier(
                    random_state=42,
                    n_jobs=-1,
                    eval_metric="logloss",
                    tree_method="hist",
                    scale_pos_weight=max(1.0, imbalance_ratio),
                    **(params or {}),
                ),
                "sampler": lambda trial: {
                    "n_estimators": trial.suggest_int("n_estimators", 150, 500),
                    "max_depth": trial.suggest_int("max_depth", 2, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                },
            }
        )

    if LIGHTGBM_AVAILABLE:
        candidates.append(
            {
                "name": "LightGBM",
                "builder": lambda params=None: LGBMClassifier(
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced",
                    **(params or {}),
                ),
                "sampler": lambda trial: {
                    "n_estimators": trial.suggest_int("n_estimators", 150, 500),
                    "num_leaves": trial.suggest_int("num_leaves", 16, 64),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "max_depth": trial.suggest_int("max_depth", -1, 12),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                },
            }
        )

    if CATBOOST_AVAILABLE:
        candidates.append(
            {
                "name": "CatBoost",
                "builder": lambda params=None: CatBoostClassifier(
                    random_seed=42,
                    verbose=0,
                    loss_function="Logloss",
                    allow_writing_files=False,
                    auto_class_weights="Balanced",
                    **(params or {}),
                ),
                "sampler": lambda trial: {
                    "iterations": trial.suggest_int("iterations", 150, 500),
                    "depth": trial.suggest_int("depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 8.0),
                },
            }
        )

    return candidates


def _build_pipeline(preprocessor, estimator, use_smote):
    if use_smote and SMOTE_AVAILABLE and ImbPipeline is not None:
        return ImbPipeline(
            steps=[
                ("preprocess", preprocessor),
                ("smote", SMOTE(random_state=42)),
                ("model", estimator),
            ]
        )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", estimator),
        ]
    )


def _optimize_estimator(candidate, X_train, y_train, preprocessor, cv, scoring, use_smote):
    if not OPTUNA_AVAILABLE or len(X_train) < 40:
        estimator = candidate["builder"]()
        return estimator, {"used_optuna": False, "best_cv_score": None, "best_params": {}}

    trial_budget = 12 if len(X_train) < 2000 else 8

    def objective(trial):
        params = candidate["sampler"](trial)
        estimator = candidate["builder"](params)
        pipeline = _build_pipeline(preprocessor, estimator, use_smote)
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            error_score="raise",
        )
        return float(np.nanmean(scores["test_score"]))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trial_budget, show_progress_bar=False)
    best_estimator = candidate["builder"](study.best_params)
    return best_estimator, {
        "used_optuna": True,
        "best_cv_score": float(study.best_value),
        "best_params": deepcopy(study.best_params),
        "n_trials": trial_budget,
    }


def _select_primary_metric(y):
    positive_rate = float(np.mean(y)) if len(y) else 0.0
    if 0.10 <= positive_rate <= 0.90:
        return "roc_auc"
    return "f1"


def _safe_auc(y_true, y_score):
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return None


def _build_feature_names(preprocessor, frame):
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        names = []
        for column in frame.columns:
            names.append(column)
        return names


def _extract_transformed_frame(pipeline, frame, sample_size=200):
    preprocessor = pipeline.named_steps["preprocess"]
    sample = frame.sample(n=min(len(frame), sample_size), random_state=42) if len(frame) > 0 else frame
    transformed = preprocessor.transform(sample)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    columns = _build_feature_names(preprocessor, sample)
    if transformed.ndim == 1:
        transformed = transformed.reshape(-1, 1)
    if len(columns) != transformed.shape[1]:
        columns = [f"feature_{index}" for index in range(transformed.shape[1])]
    return sample, pd.DataFrame(transformed, columns=columns, index=sample.index)


def _build_explanations(pipeline, frame, top_n=10):
    explanations = {
        "method": "model_feature_importance",
        "feature_importance": [],
        "sample_explanations": [],
        "shap_used": False,
    }

    try:
        sample, transformed_frame = _extract_transformed_frame(pipeline, frame, sample_size=200)
        estimator = pipeline.named_steps["model"]

        if SHAP_AVAILABLE:
            try:
                if hasattr(estimator, "predict_proba"):
                    explainer = shap.Explainer(estimator, transformed_frame)
                else:
                    explainer = shap.Explainer(estimator, transformed_frame)

                shap_values = explainer(transformed_frame)
                values = shap_values.values
                if isinstance(values, list):
                    values = values[-1]
                values = np.asarray(values)
                if values.ndim == 3:
                    values = values[..., -1]

                mean_abs_shap = np.abs(values).mean(axis=0)
                ranked = pd.DataFrame(
                    {
                        "feature": transformed_frame.columns,
                        "importance": mean_abs_shap,
                    }
                ).sort_values("importance", ascending=False)

                explanations["feature_importance"] = ranked.head(top_n).to_dict(orient="records")
                explanations["shap_used"] = True

                sample_rows = min(3, len(transformed_frame))
                for row_index in range(sample_rows):
                    feature_contributions = dict(zip(transformed_frame.columns, values[row_index].tolist()))
                    explanations["sample_explanations"].append(
                        {
                            "row": int(row_index),
                            "feature_contributions": dict(sorted(feature_contributions.items(), key=lambda item: abs(item[1]), reverse=True)[:8]),
                        }
                    )

                return explanations
            except Exception as shap_error:
                explanations["shap_error"] = str(shap_error)

        if hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_
            ranked = pd.DataFrame(
                {
                    "feature": transformed_frame.columns,
                    "importance": importances,
                }
            ).sort_values("importance", ascending=False)
            explanations["feature_importance"] = ranked.head(top_n).to_dict(orient="records")

        elif hasattr(estimator, "coef_"):
            coefficients = np.abs(np.asarray(estimator.coef_)).mean(axis=0)
            ranked = pd.DataFrame(
                {
                    "feature": transformed_frame.columns,
                    "importance": coefficients,
                }
            ).sort_values("importance", ascending=False)
            explanations["feature_importance"] = ranked.head(top_n).to_dict(orient="records")

        if not explanations["feature_importance"]:
            explanations["feature_importance"] = [
                {"feature": column, "importance": 0.0}
                for column in transformed_frame.columns[:top_n]
            ]

        for row_index in range(min(3, len(sample))):
            row = sample.iloc[row_index].to_dict()
            explanations["sample_explanations"].append(
                {
                    "row": int(sample.index[row_index]),
                    "feature_snapshot": {key: row.get(key) for key in list(sample.columns)[:10]},
                }
            )

    except Exception as error:
        explanations["error"] = str(error)

    return explanations


def run_predictive_model(df, targets_dict=None):
    """
    Diabetes AutoML engine.

    Builds a clinical classification stack with feature engineering, cross-validation,
    Optuna tuning, SMOTE balancing, SHAP explainability, and risk scoring.
    """

    working_df = df.copy()
    working_df.columns = [str(column).strip() for column in working_df.columns]

    missing_before = int(working_df.isna().sum().sum())
    duplicate_rows = int(working_df.duplicated().sum())
    working_df = working_df.drop_duplicates().reset_index(drop=True)

    for column in working_df.columns:
        if pd.api.types.is_datetime64_any_dtype(working_df[column]):
            continue
        if "date" in column.lower() or "time" in column.lower():
            parsed = pd.to_datetime(working_df[column], errors="coerce")
            if parsed.notna().sum() > 0:
                working_df[column] = parsed

    working_df, feature_engineering_summary = _engineer_features(working_df)
    missing_after = int(working_df.isna().sum().sum())

    diabetes_targets = _detect_diabetes_targets(working_df)
    diabetes_targets = _select_preferred_diabetes_target(diabetes_targets)
    modeling_frame = working_df.copy()
    target_source = "observed"

    if diabetes_targets:
        target_name = diabetes_targets[0]
        target_series = modeling_frame[target_name]
        target_series, target_normalization = _normalize_target_values(target_series)
        modeling_frame[target_name] = target_series.astype(int)
    else:
        surrogate_target, surrogate_meta = _build_surrogate_diabetes_target(modeling_frame)
        target_name = str((surrogate_meta or {}).get("target_name", "future_diabetes_likelihood"))
        target_source = str((surrogate_meta or {}).get("source", "surrogate_rule_engine"))
        target_normalization = {
            "strategy": "surrogate_target",
            "positive_rate": float((surrogate_meta or {}).get("positive_rate", 0.0)),
            "features_used": (surrogate_meta or {}).get("features_used", []),
        }
        modeling_frame[target_name] = pd.to_numeric(pd.Series(surrogate_target, index=modeling_frame.index), errors="coerce").fillna(0).astype(int)
        diabetes_targets = [target_name]

    modeling_frame = modeling_frame.dropna(subset=[target_name]).reset_index(drop=True)
    y = modeling_frame[target_name].astype(int)
    X = modeling_frame.drop(columns=[target_name], errors="ignore")
    if target_name in X.columns:
        X = X.drop(columns=[target_name], errors="ignore")

    if y.nunique() < 2:
        return {
            "predictions": {
                target_name: {
                    "error": "Not enough target classes to train a diabetes classifier",
                    "task": "classification",
                }
            },
            "feature_engineering": feature_engineering_summary,
            "diabetes_detection": {
                "detected_targets": diabetes_targets,
                "prediction_target": target_name,
                "strategy": target_source,
                "future_likelihood_supported": True,
            },
            "model_monitoring": {
                "status": "blocked",
                "reason": "Single-class target after cleaning",
                "missing_before": missing_before,
                "missing_after": missing_after,
                "duplicate_rows_removed": duplicate_rows,
            },
            "risk_scoring": {},
            "model_leaderboard": [],
            "shap_explanations": {},
            "diabetes_targets": diabetes_targets,
            "modeling_frame": modeling_frame,
        }

    if int(y.value_counts().min()) < 2:
        return {
            "predictions": {
                target_name: {
                    "error": "Target classes are too imbalanced to build a stable classifier",
                }
            },
            "feature_engineering": feature_engineering_summary,
            "diabetes_detection": {
                "detected_targets": diabetes_targets or [],
                "prediction_target": target_name,
                "strategy": target_source,
                "future_likelihood_supported": True,
            },
            "model_monitoring": {
                "status": "insufficient_class_balance",
                "row_count": int(len(modeling_frame)),
                "column_count": int(modeling_frame.shape[1]),
            },
            "risk_scoring": {
                "mode": target_source,
                "high_risk_share": 0.0,
                "average_risk": 0.0,
                "sample_scores": [],
            },
        }

    preprocessor, preprocessing_summary = _build_preprocessor(X)

    positive_rate = float(y.mean())
    imbalance_ratio = float(max(1.0, (1.0 - positive_rate) / max(positive_rate, 1e-6)))
    use_smote = SMOTE_AVAILABLE and positive_rate < 0.45 and y.value_counts().min() >= 6
    primary_metric = _select_primary_metric(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.nunique() > 1 and int(y.value_counts().min()) >= 2 else None,
    )

    train_min_class_count = int(y_train.value_counts().min())
    if train_min_class_count < 2:
        return {
            "predictions": {
                target_name: {
                    "error": "Training split does not contain enough examples per class for cross-validation",
                    "task": "classification",
                }
            },
            "feature_engineering": feature_engineering_summary,
            "diabetes_detection": {
                "detected_targets": diabetes_targets,
                "prediction_target": target_name,
                "strategy": target_source,
                "future_likelihood_supported": True,
            },
            "model_monitoring": {
                "status": "insufficient_class_balance_after_split",
                "missing_before": missing_before,
                "missing_after": missing_after,
                "duplicate_rows_removed": duplicate_rows,
            },
            "risk_scoring": {},
            "model_leaderboard": [],
            "shap_explanations": {},
            "diabetes_targets": diabetes_targets,
            "modeling_frame": modeling_frame,
        }

    cv_splits = max(2, min(5, train_min_class_count))
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    scoring = primary_metric

    candidates = _build_candidate_factories(imbalance_ratio)
    leaderboard = []
    best_candidate = None
    best_estimator = None
    best_cv_score = -np.inf

    for candidate in candidates:
        tuned_estimator, tuning_summary = _optimize_estimator(candidate, X_train, y_train, preprocessor, cv, scoring, use_smote)
        tuned_pipeline = _build_pipeline(preprocessor, tuned_estimator, use_smote)

        try:
            cv_scores = cross_validate(
                tuned_pipeline,
                X_train,
                y_train,
                cv=cv,
                scoring={
                    "primary": scoring,
                    "accuracy": "accuracy",
                    "balanced_accuracy": "balanced_accuracy",
                    "f1": "f1",
                },
                n_jobs=-1,
                error_score="raise",
            )
            cv_primary = float(np.nanmean(cv_scores["test_primary"]))
            cv_accuracy = float(np.nanmean(cv_scores["test_accuracy"]))
            cv_balanced_accuracy = float(np.nanmean(cv_scores["test_balanced_accuracy"]))
            cv_f1 = float(np.nanmean(cv_scores["test_f1"]))
        except Exception as error:
            leaderboard.append(
                {
                    "model": candidate["name"],
                    "status": "failed",
                    "error": str(error),
                }
            )
            continue

        final_pipeline = _build_pipeline(preprocessor, tuned_estimator, use_smote)
        final_pipeline.fit(X_train, y_train)

        predictions = final_pipeline.predict(X_test)
        probability_scores = None
        if hasattr(final_pipeline, "predict_proba"):
            try:
                probability_scores = final_pipeline.predict_proba(X_test)[:, 1]
            except Exception:
                probability_scores = None

        holdout_accuracy = float(accuracy_score(y_test, predictions))
        holdout_balanced_accuracy = float(balanced_accuracy_score(y_test, predictions))
        holdout_f1 = float(f1_score(y_test, predictions, zero_division=0))
        holdout_precision = float(precision_score(y_test, predictions, zero_division=0))
        holdout_recall = float(recall_score(y_test, predictions, zero_division=0))
        holdout_auc = _safe_auc(y_test, probability_scores) if probability_scores is not None else None

        leaderboard.append(
            {
                "model": candidate["name"],
                "status": "success",
                "cv_primary": round(cv_primary, 4),
                "cv_accuracy": round(cv_accuracy, 4),
                "cv_balanced_accuracy": round(cv_balanced_accuracy, 4),
                "cv_f1": round(cv_f1, 4),
                "holdout_accuracy": round(holdout_accuracy, 4),
                "holdout_balanced_accuracy": round(holdout_balanced_accuracy, 4),
                "holdout_f1": round(holdout_f1, 4),
                "holdout_precision": round(holdout_precision, 4),
                "holdout_recall": round(holdout_recall, 4),
                "holdout_auc": round(holdout_auc, 4) if holdout_auc is not None else None,
                "optuna": tuning_summary,
                "smote_used": bool(use_smote),
            }
        )

        if cv_primary > best_cv_score:
            best_cv_score = cv_primary
            best_candidate = candidate
            best_estimator = tuned_estimator

    if best_estimator is None:
        return {
            "predictions": {
                target_name: {
                    "error": "All candidate models failed during training",
                    "task": "classification",
                }
            },
            "feature_engineering": feature_engineering_summary,
            "model_monitoring": {
                "status": "failed",
                "missing_before": missing_before,
                "missing_after": missing_after,
                "duplicate_rows_removed": duplicate_rows,
            },
            "risk_scoring": {},
            "model_leaderboard": leaderboard,
            "shap_explanations": {},
            "diabetes_targets": diabetes_targets,
            "modeling_frame": modeling_frame,
        }

    final_pipeline = _build_pipeline(preprocessor, best_estimator, use_smote)
    final_pipeline.fit(X, y)

    all_probabilities = None
    if hasattr(final_pipeline, "predict_proba"):
        try:
            all_probabilities = final_pipeline.predict_proba(X)[:, 1]
        except Exception:
            all_probabilities = None

    full_predictions = final_pipeline.predict(X)
    predicted_probabilities = all_probabilities if all_probabilities is not None else full_predictions.astype(float)

    risk_score_values = np.clip(np.asarray(predicted_probabilities, dtype=float) * 100.0, 0.0, 100.0)
    risk_band = pd.cut(
        risk_score_values,
        bins=[-np.inf, 35, 70, np.inf],
        labels=["Low", "Moderate", "High"],
    ).astype(str)

    top_risk_indices = np.argsort(-risk_score_values)[: min(10, len(risk_score_values))]
    top_risk_cases = []
    for index in top_risk_indices:
        top_risk_cases.append(
            {
                "row": int(index),
                "risk_score": round(float(risk_score_values[index]), 2),
                "risk_band": str(risk_band[index]),
                "predicted_probability": round(float(predicted_probabilities[index]), 4),
            }
        )

    clinical_risk_summary = {
        "mean_risk_score": round(float(np.mean(risk_score_values)), 2),
        "median_risk_score": round(float(np.median(risk_score_values)), 2),
        "max_risk_score": round(float(np.max(risk_score_values)), 2),
        "high_risk_count": int((risk_score_values >= 70).sum()),
        "moderate_risk_count": int(((risk_score_values >= 35) & (risk_score_values < 70)).sum()),
        "low_risk_count": int((risk_score_values < 35).sum()),
        "high_risk_share": round(float((risk_score_values >= 70).mean()), 4),
        "top_risk_cases": top_risk_cases,
        "target_source": target_source,
        "positive_rate": round(float(y.mean()), 4),
    }
    clinical_risk_summary["average_risk"] = clinical_risk_summary["mean_risk_score"]

    shap_explanations = {
        target_name: _build_explanations(final_pipeline, X)
    }

    diabetes_detection = {
        "detected_targets": diabetes_targets,
        "prediction_target": target_name,
        "strategy": target_source,
        "future_likelihood_supported": True,
        "explicit_label_detected": bool(target_source == "observed"),
        "proxy_target_generated": bool(target_source != "observed"),
    }

    confidence = max(0.5, min(0.99, float(best_cv_score)))
    predictions = {
        target_name: {
            "task": "classification",
            "target_source": target_source,
            "best_model": best_candidate["name"],
            "best_model_pipeline": final_pipeline,
            "confidence": round(confidence, 4),
            "cv_primary_metric": primary_metric,
            "cv_primary_score": round(float(best_cv_score), 4),
            "sample_predictions": [int(value) for value in full_predictions[:5].tolist()],
            "sample_probabilities": [round(float(value), 4) for value in np.asarray(predicted_probabilities[:5])],
            "sample_risk_scores": [round(float(value), 2) for value in risk_score_values[:5]],
            "accuracy": round(float(accuracy_score(y, full_predictions)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y, full_predictions)), 4),
            "f1_score": round(float(f1_score(y, full_predictions, zero_division=0)), 4),
            "precision": round(float(precision_score(y, full_predictions, zero_division=0)), 4),
            "recall": round(float(recall_score(y, full_predictions, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y, predicted_probabilities)), 4) if len(np.unique(y)) > 1 else None,
            "risk_score_summary": clinical_risk_summary,
            "classification_report": classification_report(y, full_predictions, output_dict=True, zero_division=0),
            "feature_engineering_summary": feature_engineering_summary,
            "preprocessing_summary": preprocessing_summary,
        }
    }

    monitoring_summary = {
        "status": "active",
        "dataset": {
            "rows": int(len(modeling_frame)),
            "columns": int(modeling_frame.shape[1]),
            "missing_before": missing_before,
            "missing_after": missing_after,
            "duplicate_rows_removed": duplicate_rows,
        },
        "target": {
            "name": target_name,
            "positive_rate": round(float(y.mean()), 4),
            "negative_rate": round(float(1.0 - y.mean()), 4),
        },
        "training": {
            "cv_folds": int(cv_splits),
            "primary_metric": primary_metric,
            "best_model": best_candidate["name"],
            "best_cv_score": round(float(best_cv_score), 4),
            "smote_used": bool(use_smote),
            "optuna_available": bool(OPTUNA_AVAILABLE),
        },
        "leaderboard": leaderboard,
        "risk_distribution": {
            "mean": round(float(np.mean(risk_score_values)), 2),
            "std": round(float(np.std(risk_score_values)), 2),
            "p90": round(float(np.percentile(risk_score_values, 90)), 2),
        },
    }

    if track_dataset_history is not None:
        try:
            tracked_predictions = {
                target_name: {
                    "confidence": confidence,
                    "risk_score": clinical_risk_summary["mean_risk_score"],
                }
            }
            _, memory_summary = track_dataset_history(modeling_frame, tracked_predictions)
            monitoring_summary["memory_tracking"] = memory_summary
            if memory_summary.get("data_drift"):
                monitoring_summary["data_drift"] = memory_summary.get("data_drift")
        except Exception as error:
            monitoring_summary["memory_tracking_error"] = str(error)

    return {
        "predictions": predictions,
        "feature_engineering": feature_engineering_summary,
        "model_monitoring": monitoring_summary,
        "risk_scoring": clinical_risk_summary,
        "diabetes_detection": diabetes_detection,
        "model_leaderboard": leaderboard,
        "shap_explanations": shap_explanations,
        "diabetes_targets": diabetes_targets,
        "modeling_frame": modeling_frame,
    }