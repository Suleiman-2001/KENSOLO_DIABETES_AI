# engines/self_critic.py
def self_critic(df, predictions):
    """
    Data-agnostic risk evaluator
    """
    risk_flags = []

    if df.shape[0] < 50:
        risk_flags.append("Dataset too small for reliable learning")

    if df.isnull().sum().sum() > 0:
        risk_flags.append("Missing values present")

    for target, result in predictions.items():
        if result.get("error"):
            risk_flags.append(f"Prediction failed for {target}")
        elif result.get("accuracy", 1) < 0.6:
            risk_flags.append(f"Low model confidence for {target}")

    blocked = len(risk_flags) > 2
    trust_level = "high" if not risk_flags else "medium" if len(risk_flags) <= 2 else "low"

    return {"trust_level": trust_level, "risk_flags": risk_flags, "blocked": blocked}
