import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


def explain_predictions(model_pipeline, df: pd.DataFrame, target: str, top_n: int = 10):
    """
    Diabetes Explainability Engine (Hybrid Version)

    Combines:
    1. SHAP-based explainability (if available)
    2. Clinical rule-based interpretation (fallback / enhancement)
    """

    explanations = {
        "target": target,
        "method": "hybrid_shap_clinical",
        "feature_importance": [],
        "sample_explanations": [],
        "clinical_risk_drivers": [],
        "global_interpretation": {}
    }

    # ----------------------------
    # 1️⃣ Prepare features
    # ----------------------------
    try:
        X = df.drop(columns=[target])
    except Exception:
        return {"error": "Target column not found in dataframe"}

    # ----------------------------
    # 2️⃣ SHAP EXPLANATION (PRIMARY IF AVAILABLE)
    # ----------------------------
    shap_importance = None

    if SHAP_AVAILABLE:
        try:
            explainer = shap.Explainer(model_pipeline.predict, X)
            shap_values = explainer(X)

            mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

            shap_importance = pd.DataFrame({
                "feature": X.columns,
                "importance": mean_abs_shap
            }).sort_values("importance", ascending=False)

            explanations["feature_importance"] = shap_importance.head(top_n).to_dict(orient="records")

            # Sample explanations (first 3 rows only for performance)
            sample_expl = []
            for i in range(min(3, X.shape[0])):
                contrib = dict(zip(X.columns, shap_values[i].values))
                sample_expl.append({
                    "row": i,
                    "feature_contributions": contrib
                })

            explanations["sample_explanations"] = sample_expl

        except Exception as e:
            explanations["shap_error"] = str(e)

    # ----------------------------
    # 3️⃣ FALLBACK FEATURE IMPORTANCE (NON-SHAP MODELS)
    # ----------------------------
    if not explanations["feature_importance"]:
        model = None

        try:
            if hasattr(model_pipeline, "named_steps"):
                model = list(model_pipeline.named_steps.values())[-1]
            else:
                model = model_pipeline

            importance = {}

            if hasattr(model, "feature_importances_"):
                importance = dict(zip(X.columns, model.feature_importances_))

            elif hasattr(model, "coef_"):
                importance = dict(zip(X.columns, np.abs(model.coef_).flatten()))

            sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)

            explanations["feature_importance"] = [
                {"feature": k, "importance": v} for k, v in sorted_imp[:top_n]
            ]

        except Exception:
            explanations["feature_importance"] = []

    # ----------------------------
    # 4️⃣ CLINICAL INTERPRETATION LAYER (DIABETES LOGIC)
    # ----------------------------
    feature_names = [f["feature"].lower() for f in explanations["feature_importance"]]

    for f in feature_names:

        if "glucose" in f:
            explanations["clinical_risk_drivers"].append(
                "Elevated glucose is a primary diabetes risk indicator"
            )

        if "bmi" in f:
            explanations["clinical_risk_drivers"].append(
                "High BMI contributes to insulin resistance risk"
            )

        if "age" in f:
            explanations["clinical_risk_drivers"].append(
                "Age increases long-term diabetes risk probability"
            )

        if "blood" in f or "pressure" in f:
            explanations["clinical_risk_drivers"].append(
                "Blood pressure reflects metabolic syndrome risk"
            )

        if "insulin" in f:
            explanations["clinical_risk_drivers"].append(
                "Insulin levels reflect pancreatic function stability"
            )

    # ----------------------------
    # 5️⃣ GLOBAL INTERPRETATION
    # ----------------------------
    top_feature = explanations["feature_importance"][0]["feature"] if explanations["feature_importance"] else None

    explanations["global_interpretation"] = {
        "primary_driver": top_feature,
        "model_behavior": (
            "Metabolic indicators dominate prediction logic"
            if any("glucose" in f for f in feature_names)
            else "Mixed clinical and non-clinical influence detected"
        )
    }

    return explanations