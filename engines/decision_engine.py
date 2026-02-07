# engines/decision_engine.py

def run_decision_intelligence(predictions, business_insights, self_critic):
    """
    Business Decision Intelligence Engine
    Converts insights + predictions into executive-level decisions
    """

    decisions = []

    # ----------------------------
    # 0️⃣ Trust Gate (Critical)
    # ----------------------------
    if self_critic.get("trust_level") == "low":
        return {
            "status": "blocked",
            "reason": "Low trust score – decisions blocked",
            "decisions": []
        }

    # ----------------------------
    # 1️⃣ Sales Growth Decisions
    # ----------------------------
    sales_trends = business_insights.get("sales_trends", {})
    monthly = sales_trends.get("monthly_trend", {})

    if isinstance(monthly, dict) and len(monthly) >= 2:
        values = list(monthly.values())
        if values[-1] > values[-2]:
            decisions.append({
                "decision": "Increase sales investment",
                "confidence": 0.80,
                "expected_impact": {
                    "revenue": "Increase",
                    "risk": "Medium"
                },
                "reasoning": [
                    "Positive month-over-month sales trend",
                    "Sustained revenue growth detected"
                ],
                "recommended_action": "Increase marketing or inventory allocation"
            })

    # ----------------------------
    # 2️⃣ Prediction-Based Decisions
    # ----------------------------
    for target, info in predictions.items():
        if info.get("task") == "regression" and info.get("r2_score", 0) >= 0.7:
            decisions.append({
                "decision": f"Operationalize {target} forecast",
                "confidence": round(info.get("r2_score", 0), 2),
                "expected_impact": {
                    "efficiency": "Improved",
                    "risk": "Low"
                },
                "reasoning": [
                    "High predictive confidence",
                    "Stable model performance across algorithms"
                ],
                "recommended_action": f"Use {target} predictions for planning and optimization"
            })

        if info.get("task") == "classification" and info.get("accuracy", 0) >= 0.75:
            decisions.append({
                "decision": f"Act on {target} classification outcomes",
                "confidence": round(info.get("accuracy", 0), 2),
                "expected_impact": {
                    "decision_quality": "Improved",
                    "risk": "Medium"
                },
                "reasoning": [
                    "High classification accuracy",
                    "Clear segment differentiation"
                ],
                "recommended_action": f"Deploy class-based strategies for {target}"
            })

    # ----------------------------
    # 3️⃣ Risk-Aware Decisions
    # ----------------------------
    if self_critic.get("risk_flags"):
        decisions.append({
            "decision": "Apply conservative strategy",
            "confidence": 0.65,
            "expected_impact": {
                "stability": "High",
                "risk": "Low"
            },
            "reasoning": self_critic.get("risk_flags"),
            "recommended_action": "Limit aggressive actions until data quality improves"
        })

    return {
        "status": "active",
        "decision_count": len(decisions),
        "decisions": decisions
    }
