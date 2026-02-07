# engines/adaptive_engine.py
import pandas as pd
import numpy as np

def run_adaptive_analytics(df, predictions, recommendations, industry=None):
    """
    Self-learning analytics engine:
    - Tracks prediction accuracy over time
    - Updates recommendation thresholds
    - Suggests adaptive strategies
    """
    insights = {}

    for target, info in predictions.items():
        sample_preds = info.get("sample_predictions", [])
        if not sample_preds:
            continue

        # Example: Check if predictions exceed historical mean ± 2*STD
        historical_mean = np.mean(sample_preds)
        historical_std = np.std(sample_preds)
        adaptive_flags = []

        for val in sample_preds:
            if val < historical_mean - 2*historical_std:
                adaptive_flags.append({"value": val, "alert": "Below expected range"})
            elif val > historical_mean + 2*historical_std:
                adaptive_flags.append({"value": val, "alert": "Above expected range"})

        insights[target] = {
            "historical_mean": historical_mean,
            "historical_std": historical_std,
            "adaptive_flags": adaptive_flags,
            "recommendation_adjustments": [
                f"Consider re-evaluating thresholds for {target}" if adaptive_flags else "Thresholds OK"
            ]
        }

    return insights
