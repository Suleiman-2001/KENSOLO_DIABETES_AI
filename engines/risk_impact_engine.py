# engines/impact_engine.py

import numpy as np

def calculate_revenue_cost_impact(predictions, business_insights=None):
    """
    AI Impact Engine (Revenue + Cost + Risk-aware)
    Converts predictions into business impact estimates
    using confidence-aware scaling instead of fixed multipliers.
    """

    impact_report = {}

    global_risk_multiplier = 1.0

    # ----------------------------
    # 1. Adjust based on business insights (optional)
    # ----------------------------
    if business_insights:
        risk_flags = business_insights.get("risk_flags", [])

        if len(risk_flags) > 3:
            global_risk_multiplier = 0.7  # high uncertainty
        elif len(risk_flags) > 0:
            global_risk_multiplier = 0.85

    # ----------------------------
    # 2. Compute impact per target
    # ----------------------------
    for target, info in predictions.items():

        if "error" in info:
            continue

        preds = info.get("sample_predictions", [])
        if not preds:
            continue

        # ----------------------------
        # Core statistics
        # ----------------------------
        avg_pred = float(np.mean(preds))
        std_pred = float(np.std(preds)) if len(preds) > 1 else 0

        confidence = info.get("r2_score") or info.get("accuracy") or 0.5

        # ----------------------------
        # Dynamic scaling (instead of fixed multipliers)
        # ----------------------------
        base_value = 1000 if info.get("task") == "regression" else 500

        volatility_factor = 1 + (std_pred / (avg_pred + 1))
        confidence_factor = confidence

        revenue_impact = (
            avg_pred *
            base_value *
            confidence_factor *
            global_risk_multiplier
        )

        cost_impact = (
            avg_pred *
            (base_value * 0.5) *
            volatility_factor *
            (1 - confidence_factor)
        )

        # ----------------------------
        # Risk classification
        # ----------------------------
        if confidence >= 0.8:
            risk_level = "Low"
        elif confidence >= 0.6:
            risk_level = "Medium"
        else:
            risk_level = "High"

        impact_report[target] = {
            "avg_prediction": avg_pred,
            "std_prediction": std_pred,
            "confidence": round(confidence, 3),

            "estimated_revenue_impact": round(revenue_impact, 2),
            "estimated_cost_impact": round(cost_impact, 2),

            "net_impact": round(revenue_impact - cost_impact, 2),

            "risk_level": risk_level,

            "interpretation": (
                "High business value opportunity"
                if revenue_impact > cost_impact
                else "Cost-heavy scenario requiring optimization"
            )
        }

    return impact_report