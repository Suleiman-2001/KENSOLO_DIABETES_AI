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
        "global_interpretation": {},
        "future_prediction_layer": {}
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

    # ----------------------------
    # 6️⃣ FUTURE PREDICTION LAYER (SCENARIO SIMULATION)
    # ----------------------------
    def _get_numeric_series_by_keyword(keyword):
        for c in X.columns:
            if keyword in str(c).lower() and pd.api.types.is_numeric_dtype(X[c]):
                series = pd.to_numeric(X[c], errors="coerce").dropna()
                if not series.empty:
                    return c, series
        return None, None

    age_col, age_series = _get_numeric_series_by_keyword("age")
    glucose_col, glucose_series = _get_numeric_series_by_keyword("glucose")
    bmi_col, bmi_series = _get_numeric_series_by_keyword("bmi")
    insulin_col, insulin_series = _get_numeric_series_by_keyword("insulin")
    bp_col, bp_series = _get_numeric_series_by_keyword("pressure")

    if glucose_series is not None:
        baseline_glucose = float(glucose_series.median())
        baseline_age = float(age_series.median()) if age_series is not None else 50.0
        baseline_bmi = float(bmi_series.median()) if bmi_series is not None else 28.0
        baseline_insulin = float(insulin_series.median()) if insulin_series is not None else 90.0
        baseline_bp = float(bp_series.median()) if bp_series is not None else 75.0

        def _risk_score(sim_age, sim_glucose):
            glucose_effect = max(0.0, (sim_glucose - 100.0) / 100.0) * 0.45
            age_effect = max(0.0, (sim_age - 45.0) / 100.0) * 0.20
            bmi_effect = max(0.0, (baseline_bmi - 25.0) / 100.0) * 0.20
            insulin_effect = max(0.0, (120.0 - baseline_insulin) / 100.0) * 0.10
            bp_effect = max(0.0, (baseline_bp - 80.0) / 100.0) * 0.05
            score = 0.15 + glucose_effect + age_effect + bmi_effect + insulin_effect + bp_effect
            return round(float(min(1.0, max(0.0, score))), 3)

        risk_trajectory = []
        for years_ahead in [0, 5, 10, 15]:
            sim_age = baseline_age + years_ahead
            risk_trajectory.append(
                {
                    "years_ahead": years_ahead,
                    "projected_age": round(sim_age, 1),
                    "simulated_risk": _risk_score(sim_age, baseline_glucose)
                }
            )

        glucose_scenarios = [
            ("improved_control", -15.0),
            ("stable", 0.0),
            ("worsening_control", 15.0),
            ("severe_progression", 30.0)
        ]

        glucose_progression = []
        for name, delta in glucose_scenarios:
            sim_glucose = baseline_glucose + delta
            glucose_progression.append(
                {
                    "scenario": name,
                    "simulated_glucose": round(sim_glucose, 1),
                    "simulated_risk": _risk_score(baseline_age, sim_glucose)
                }
            )

        explanations["future_prediction_layer"] = {
            "age_reference_column": age_col,
            "glucose_reference_column": glucose_col,
            "risk_trajectory_over_age": risk_trajectory,
            "glucose_progression_scenarios": glucose_progression,
            "note": "Scenario-based simulation for future work planning; not a causal forecast."
        }
    else:
        explanations["future_prediction_layer"] = {
            "risk_trajectory_over_age": [],
            "glucose_progression_scenarios": [],
            "note": "Future prediction layer unavailable because no numeric glucose column was detected."
        }

    return explanations