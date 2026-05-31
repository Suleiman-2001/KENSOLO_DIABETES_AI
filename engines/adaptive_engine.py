# engines/adaptive_engine.py
import numpy as np

def run_adaptive_analytics(
    df,
    predictions,
    recommendations=None,
    industry=None,
    std_multiplier=2.0,
    max_flags=50
):
    """
    Self-learning Adaptive Analytics Engine

    Purpose:
    - Detect distribution drift in predictions
    - Identify anomalies using statistical bounds
    - Provide adaptive improvement signals for decision layer
    """

    insights = {}

    for target, info in predictions.items():

        # ----------------------------
        # Safe prediction extraction
        # ----------------------------
        sample_preds = np.asarray(info.get("sample_predictions", []), dtype=np.float64)
        sample_preds = sample_preds[~np.isnan(sample_preds)]

        if sample_preds.size == 0:
            insights[target] = {
                "mean": None,
                "std": None,
                "anomalies": [],
                "recommendation_signal": "No prediction data available"
            }
            continue

        mean = float(sample_preds.mean())
        std = float(sample_preds.std())

        # ----------------------------
        # Handle no variance case
        # ----------------------------
        if std == 0:
            insights[target] = {
                "mean": mean,
                "std": 0.0,
                "anomalies": [],
                "recommendation_signal": "Stable predictions (no variation detected)"
            }
            continue

        # ----------------------------
        # Adaptive thresholds
        # ----------------------------
        lower_bound = mean - std_multiplier * std
        upper_bound = mean + std_multiplier * std

        low_anomalies = sample_preds[sample_preds < lower_bound][:max_flags]
        high_anomalies = sample_preds[sample_preds > upper_bound][:max_flags]

        anomalies = []

        for v in low_anomalies:
            anomalies.append({
                "value": float(v),
                "type": "LOW_DEVIATION"
            })

        for v in high_anomalies:
            anomalies.append({
                "value": float(v),
                "type": "HIGH_DEVIATION"
            })

        # ----------------------------
        # Industry-aware adjustment layer
        # ----------------------------
        signal = "Thresholds OK"

        if anomalies:
            if industry == "healthcare":
                signal = "Clinical review recommended due to abnormal variation"
            elif industry == "finance":
                signal = "Financial volatility detected"
            elif industry == "retail":
                signal = "Demand instability detected"
            else:
                signal = "Prediction instability detected"

        # ----------------------------
        # Output structure
        # ----------------------------
        insights[target] = {
            "mean": mean,
            "std": std,
            "anomaly_count": len(anomalies),
            "anomalies": anomalies,
            "recommendation_signal": signal
        }

    return insights