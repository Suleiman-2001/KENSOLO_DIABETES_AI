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


def _find_matching_column(columns, tokens):
    for column in columns:
        lower = column.lower()
        if any(token in lower for token in tokens):
            return column
    return None


def _detect_diabetes_targets(df):
    candidates = []
    for column in df.columns:
        lower = column.lower()
        if any(token in lower for token in ["diabetes", "diabetic", "has_diabetes", "diabetes_status", "diabetes_flag", "diagnosis", "outcome", "label", "target"]):
            candidates.append(column)
            continue

        if df[column].nunique(dropna=True) <= 2:
            values = {_normalize_text(v) for v in df[column].dropna().unique()}
            if values & POSITIVE_TOKENS and values & NEGATIVE_TOKENS:
                candidates.append(column)

    return list(dict.fromkeys(candidates))


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


def _engineer_features(df):
    engineered = df.copy()
    clinical = _clinical_columns(engineered)
    feature_notes = []

    numeric_like = engineered.select_dtypes(include=[np.number]).copy()

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


def _build_surrogate_diabetes_target(df):
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
        return surrogate, {"target_name": "future_diabetes_likelihood", "source": label_source, "positive_rate": float(surrogate.mean())}

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
    modeling_frame = working_df.copy()
    target_source = "observed"

    if diabetes_targets:
        target_name = diabetes_targets[0]
        target_series = modeling_frame[target_name]
        target_series, target_normalization = _normalize_target_values(target_series)
        modeling_frame[target_name] = target_series.astype(int)
    else:
        target_series, target_normalization = _build_surrogate_diabetes_target(modeling_frame)
        target_name = target_normalization["target_name"]
        modeling_frame[target_name] = target_series.astype(int)
        target_source = target_normalization["source"]
        diabetes_targets = [target_name]

    modeling_frame = modeling_frame.dropna(subset=[target_name]).reset_index(drop=True)
    y = modeling_frame[target_name].astype(int)
    X = modeling_frame.drop(columns=[target_name])

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
                "strategy": target_info["strategy"],
                "future_likelihood_supported": True,
            },
            "model_monitoring": {
                "status": "insufficient_class_balance",
                "row_count": int(len(modeling_df)),
                "column_count": int(modeling_df.shape[1]),
            },
            "risk_scoring": {
                "mode": target_info["strategy"],
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