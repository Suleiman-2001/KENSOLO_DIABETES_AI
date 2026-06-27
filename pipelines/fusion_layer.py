from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGET_HINTS = ["outcome", "diabetes", "label", "target", "has_diabetes", "diagnosis"]


def _find_binary_target(df: pd.DataFrame) -> str | None:
    for hint in TARGET_HINTS:
        if hint in df.columns:
            series = pd.to_numeric(df[hint], errors="coerce")
            values = set(series.dropna().unique().tolist())
            if values.issubset({0, 1}) and values:
                return hint

    for col in df.columns:
        if col.endswith("id"):
            continue

        series = pd.to_numeric(df[col], errors="coerce")
        values = set(series.dropna().unique().tolist())
        if values.issubset({0, 1}) and values:
            return col

    return None


def _prepare_anchor_frame(tabular_df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    work = tabular_df.copy()
    work.columns = [str(c).strip().lower() for c in work.columns]

    target_col = _find_binary_target(work)

    if "patient_id" not in work.columns:
        work = work.reset_index(drop=True)
        work["patient_id"] = np.arange(1, len(work) + 1)

    return work, target_col


def _safe_merge(anchor: pd.DataFrame, features: pd.DataFrame, suffix: str) -> pd.DataFrame:
    if features is None or features.empty:
        return anchor

    merged = anchor.copy()
    rhs = features.copy()

    if "patient_id" in rhs.columns:
        merged = merged.merge(rhs, on="patient_id", how="left")
    else:
        rhs = rhs.add_prefix(f"{suffix}_")
        for col in rhs.columns:
            if col not in merged.columns:
                merged[col] = rhs.iloc[0][col] if len(rhs) else np.nan

    return merged


def _train_fusion_classifier(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    feature_df = df.drop(columns=[target_col], errors="ignore")
    feature_df = feature_df.drop(columns=["patient_id"], errors="ignore")
    feature_df = feature_df.select_dtypes(include=[np.number])

    y = pd.to_numeric(df[target_col], errors="coerce")
    mask = y.notna()
    x = feature_df.loc[mask]
    y = y.loc[mask].astype(int)

    if x.empty or y.nunique() < 2 or len(x) < 10:
        return {
            "status": "insufficient_data",
            "samples": int(len(x)),
            "reason": "Need at least 10 labeled samples with both classes for fusion training.",
        }

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1500)),
        ]
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    auc = None
    try:
        auc = float(roc_auc_score(y_test, y_prob))
    except Exception:
        auc = None

    classifier = model.named_steps["classifier"]
    coefs = np.abs(classifier.coef_[0]) if hasattr(classifier, "coef_") else np.zeros(len(x.columns))
    coef_df = pd.DataFrame({"feature": x.columns, "abs_coef": coefs})

    modality_scores = {
        "tabular": float(coef_df[coef_df["feature"].str.startswith("tabular_")]["abs_coef"].sum()),
        "ehr": float(coef_df[coef_df["feature"].str.startswith("ehr_")]["abs_coef"].sum()),
        "imaging": float(coef_df[coef_df["feature"].str.startswith("imaging_")]["abs_coef"].sum()),
    }

    return {
        "status": "trained",
        "samples": int(len(x)),
        "features": int(x.shape[1]),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": auc,
        "top_features": coef_df.sort_values("abs_coef", ascending=False).head(8).to_dict(orient="records"),
        "modality_contribution": modality_scores,
        "fusion_model": model,
        "_x": x,
        "_y": y,
    }


def _evaluate_modality_ablation(x: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    if x.empty or y.nunique() < 2 or len(x) < 10:
        return {
            "status": "insufficient_data",
            "note": "Ablation requires at least 10 labeled samples with both classes.",
        }

    modality_sets = {
        "tabular_only": [c for c in x.columns if c.startswith("tabular_")],
        "tabular_ehr": [c for c in x.columns if c.startswith("tabular_") or c.startswith("ehr_")],
        "tabular_imaging": [c for c in x.columns if c.startswith("tabular_") or c.startswith("imaging_")],
        "all_modalities": list(x.columns),
    }

    results: Dict[str, Any] = {}

    for name, cols in modality_sets.items():
        if not cols:
            results[name] = {"status": "missing_features", "accuracy": None, "roc_auc": None}
            continue

        sub_x = x[cols]
        x_train, x_test, y_train, y_test = train_test_split(
            sub_x,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=1500)),
            ]
        )

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)[:, 1]

        auc = None
        try:
            auc = float(roc_auc_score(y_test, y_prob))
        except Exception:
            auc = None

        results[name] = {
            "status": "success",
            "features": int(len(cols)),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "roc_auc": auc,
        }

    all_auc = results.get("all_modalities", {}).get("roc_auc")
    tab_auc = results.get("tabular_only", {}).get("roc_auc")
    if all_auc is not None and tab_auc is not None:
        results["uplift_vs_tabular_only_roc_auc"] = float(all_auc - tab_auc)

    return results


def late_fusion(
    tabular_output: Dict[str, Any],
    ehr_output: Dict[str, Any],
    imaging_output: Dict[str, Any] | None = None,
    tabular_df: pd.DataFrame | None = None,
    ehr_features: pd.DataFrame | None = None,
    imaging_features: pd.DataFrame | None = None,
) -> Dict[str, Any]:
    """
    Real multimodal fusion using feature-level integration.
    Trains a fusion classifier when a binary diabetes target is available.
    """
    imaging_output = imaging_output or {}

    if tabular_df is None or tabular_df.empty:
        return {
            "fusion_strategy": "feature_level_fusion",
            "status": "skipped",
            "reason": "Tabular anchor dataset is required for multimodal alignment.",
            "tabular_signal": tabular_output,
            "ehr_signal": ehr_output,
            "imaging_signal": imaging_output,
        }

    anchor_df, target_col = _prepare_anchor_frame(tabular_df)

    tabular_numeric = anchor_df.select_dtypes(include=[np.number]).copy()
    if target_col is not None and target_col in tabular_numeric.columns:
        tabular_numeric = tabular_numeric.drop(columns=[target_col])

    protected = {"patient_id", target_col}
    for col in list(tabular_numeric.columns):
        if col in protected:
            continue
        tabular_numeric = tabular_numeric.rename(columns={col: f"tabular_{col}"})

    fusion_df = anchor_df[["patient_id"]].copy()
    if target_col is not None:
        fusion_df[target_col] = pd.to_numeric(anchor_df[target_col], errors="coerce")

    fusion_df = _safe_merge(fusion_df, tabular_numeric, "tabular")
    fusion_df = _safe_merge(fusion_df, ehr_features, "ehr")
    fusion_df = _safe_merge(fusion_df, imaging_features, "imaging")

    metrics = {}
    if target_col is not None:
        metrics = _train_fusion_classifier(fusion_df, target_col)
        if metrics.get("status") == "trained":
            metrics["ablation"] = _evaluate_modality_ablation(metrics.get("_x"), metrics.get("_y"))
            metrics.pop("_x", None)
            metrics.pop("_y", None)
    else:
        metrics = {
            "status": "no_target",
            "reason": "Binary diabetes target not found in anchor dataset; generated fused feature matrix only.",
            "samples": int(len(fusion_df)),
            "features": int(fusion_df.drop(columns=["patient_id"], errors="ignore").shape[1]),
        }

    available_modalities: List[str] = ["tabular"]
    if ehr_output.get("available"):
        available_modalities.append("ehr")
    if imaging_output.get("available"):
        available_modalities.append("imaging")

    return {
        "fusion_strategy": "feature_level_fusion",
        "status": metrics.get("status"),
        "available_modalities": available_modalities,
        "target": target_col,
        "tabular_signal": tabular_output,
        "ehr_signal": ehr_output,
        "imaging_signal": imaging_output,
        "fusion_metrics": metrics,
        "fused_rows": int(len(fusion_df)),
        "fused_columns": int(fusion_df.shape[1]),
    }
