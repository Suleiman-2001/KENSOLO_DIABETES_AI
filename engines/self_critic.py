# engines/self_critic.py

import numpy as np

def self_critic(df, predictions):
    """
    AI Self-Critic Engine (Risk + Trust Evaluator)

    Evaluates:
    - Data quality risk
    - Model reliability
    - Prediction uncertainty
    - System trust score
    """

    risk_flags = []
    risk_score = 0

    # ----------------------------
    # 1. DATASET RISKS
    # ----------------------------
    if df.shape[0] < 50:
        risk_flags.append("Dataset too small for reliable learning")
        risk_score += 2

    missing_rate = df.isnull().mean().mean()
    if missing_rate > 0.1:
        risk_flags.append("High missing data rate detected")
        risk_score += 2
    elif missing_rate > 0:
        risk_flags.append("Missing values present")
        risk_score += 1

    duplicates = df.duplicated().sum()
    if duplicates > 10:
        risk_flags.append("High duplicate record count detected")
        risk_score += 2

    constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
    if constant_cols:
        risk_flags.append("Constant/low-variance features detected")
        risk_score += 1

    # ----------------------------
    # 2. MODEL RISKS
    # ----------------------------
    for target, result in predictions.items():

        if "error" in result:
            risk_flags.append(f"Prediction failed for {target}")
            risk_score += 3
            continue

        task = result.get("task")

        if task == "classification":
            metric = result.get("accuracy", 0)
        else:
            metric = result.get("r2_score", 0)

        if metric < 0.6:
            risk_flags.append(f"Low model performance for {target}")
            risk_score += 2
        elif metric < 0.75:
            risk_flags.append(f"Moderate model performance for {target}")
            risk_score += 1

        # instability check
        preds = result.get("sample_predictions", [])
        if preds and len(preds) > 1:
            variability = np.std(preds)
            if variability > 1.0:
                risk_flags.append(f"High prediction volatility for {target}")
                risk_score += 1

    # ----------------------------
    # 3. TRUST LEVEL (IMPROVED LOGIC)
    # ----------------------------
    if risk_score >= 6:
        trust_level = "low"
    elif risk_score >= 3:
        trust_level = "medium"
    else:
        trust_level = "high"

    blocked = trust_level == "low"

    return {
        "trust_level": trust_level,
        "risk_score": risk_score,
        "risk_flags": risk_flags,
        "blocked": blocked
    }