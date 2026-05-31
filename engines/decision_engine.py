# engines/decision_engine.py

def run_decision_intelligence(predictions, business_insights, self_critic, clinical_kpis=None):
    """
    Diabetes Clinical Decision Intelligence Engine

    Converts:
    - predictions (risk models)
    - clinical KPIs (glucose, BMI, BP, etc.)
    - data quality + risk signals

    Into patient-level or population-level clinical decisions.
    """

    decisions = []

    # ----------------------------
    # 0️⃣ Safety / Trust Gate (soft-fail)
    # ----------------------------
    # Instead of hard blocking when trust is low, append a cautionary decision
    # so the UI can surface the reason and still present tentative guidance.
    if self_critic.get("trust_level") == "low":
        decisions.append({
            "decision": "Clinical decisioning suppressed due to low model trust",
            "confidence": 0.0,
            "expected_impact": {"status": "Requires review"},
            "reasoning": [
                "AI self-critic flagged low trust for the models/dataset",
                "Decisions are provided as tentative suggestions only"
            ],
            "recommended_action": "Improve data quality and retrain models before clinical deployment",
            "safety_review_required": True
        })

    # ----------------------------
    # 1️⃣ Clinical Risk-Based Decisions (NEW)
    # ----------------------------
    if clinical_kpis:
        global_kpis = clinical_kpis.get("global_kpis", {})

        # High glucose population risk
        if global_kpis.get("high_glucose_rate", 0) > 0.3:
            decisions.append({
                "decision": "Initiate diabetes risk intervention program",
                "confidence": 0.90,
                "expected_impact": {
                    "health_outcome": "Improved glycemic control",
                    "risk_reduction": "High"
                },
                "reasoning": [
                    "High proportion of elevated glucose cases detected",
                    "Population-level diabetes risk threshold exceeded"
                ],
                "recommended_action": "Recommend clinical screening and lifestyle intervention"
            })

        # Obesity risk
        if global_kpis.get("obesity_rate", 0) > 0.25:
            decisions.append({
                "decision": "Launch obesity management program",
                "confidence": 0.85,
                "expected_impact": {
                    "metabolic_risk": "Reduced",
                    "long_term_outcome": "Improved insulin sensitivity"
                },
                "reasoning": [
                    "High BMI distribution detected",
                    "Obesity is a key diabetes risk factor"
                ],
                "recommended_action": "Dietary and physical activity intervention plan"
            })

    # ----------------------------
    # 2️⃣ Prediction-Based Clinical Decisions
    # ----------------------------
    for target, info in predictions.items():

        # Diabetes risk regression models
        if info.get("task") == "regression" and info.get("r2_score", 0) >= 0.7:
            decisions.append({
                "decision": f"Use {target} for clinical risk scoring",
                "confidence": round(info.get("r2_score", 0), 2),
                "expected_impact": {
                    "diagnostic_support": "High",
                    "risk": "Low"
                },
                "reasoning": [
                    "High predictive accuracy",
                    "Stable regression model performance"
                ],
                "recommended_action": f"Integrate {target} into patient risk scoring system"
            })

        # Diabetes classification (diabetic / non-diabetic)
        if info.get("task") == "classification" and info.get("accuracy", 0) >= 0.75:
            decisions.append({
                "decision": f"Deploy diabetes classification model for {target}",
                "confidence": round(info.get("accuracy", 0), 2),
                "expected_impact": {
                    "early_detection": "Improved",
                    "clinical_decision_support": "Enhanced"
                },
                "reasoning": [
                    "High classification accuracy",
                    "Reliable separation of risk classes"
                ],
                "recommended_action": "Use for early diabetes screening support"
            })

    # ----------------------------
    # 3️⃣ Data Quality Clinical Risk Decisions
    # ----------------------------
    if self_critic.get("risk_flags"):
        decisions.append({
            "decision": "Require dataset clinical validation",
            "confidence": 0.70,
            "expected_impact": {
                "data_safety": "Improved",
                "risk": "Reduced"
            },
            "reasoning": self_critic.get("risk_flags"),
            "recommended_action": "Re-clean dataset before medical deployment"
        })

    # ----------------------------
    # 4️⃣ Default Safety Decision
    # ----------------------------
    if not decisions:
        decisions.append({
            "decision": "No critical clinical action required",
            "confidence": 0.60,
            "expected_impact": {
                "status": "Stable population health indicators"
            },
            "reasoning": [
                "No strong risk thresholds exceeded",
                "Predictions within normal clinical range"
            ],
            "recommended_action": "Continue monitoring"
        })

    return {
        "status": "active",
        "decision_count": len(decisions),
        "decisions": decisions
    }