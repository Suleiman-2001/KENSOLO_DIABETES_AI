# engines/adaptive_engine.py
import numpy as np

def run_adaptive_analytics(
    df,
    predictions,
    recommendations,
    industry=None,
    std_multiplier=2.0,
    max_flags=50
):
    """
    Safe and optimized Self-learning analytics engine:
    - Fully vectorized (no Python loops over predictions)
    - Memory-efficient
    - Scales to tiny or large prediction arrays
    - Limits flagged items for large datasets
    - Configurable anomaly threshold via std_multiplier
    - Streamlit-safe: avoids hanging on empty or NaN-heavy predictions
    """
    insights = {}

    for target, info in predictions.items():

        # Get predictions safely
        sample_preds = np.asarray(info.get("sample_predictions", []), dtype=np.float64)
        sample_preds = sample_preds[~np.isnan(sample_preds)]  # Remove NaNs

        if sample_preds.size == 0:
            insights[target] = {
                "historical_mean": None,
                "historical_std": None,
                "adaptive_flags": [],
                "recommendation_adjustments": ["No predictions available"]
            }
            continue

        # Compute statistics safely
        historical_mean = sample_preds.mean()
        historical_std = sample_preds.std()

        # Handle zero-variance
        if historical_std == 0:
            insights[target] = {
                "historical_mean": float(historical_mean),
                "historical_std": 0.0,
                "adaptive_flags": [],
                "recommendation_adjustments": ["No variation detected"]
            }
            continue

        # Compute bounds for anomaly detection
        lower_bound = historical_mean - std_multiplier * historical_std
        upper_bound = historical_mean + std_multiplier * historical_std

        # Vectorized anomaly detection
        mask_low = sample_preds < lower_bound
        mask_high = sample_preds > upper_bound

        flagged_low = sample_preds[mask_low][:max_flags]  # Limit flagged items
        flagged_high = sample_preds[mask_high][:max_flags]

        adaptive_flags = []
        if flagged_low.size > 0:
            adaptive_flags.extend([{"value": float(v), "alert": "Below expected range"} for v in flagged_low])
        if flagged_high.size > 0:
            adaptive_flags.extend([{"value": float(v), "alert": "Above expected range"} for v in flagged_high])

        insights[target] = {
            "historical_mean": float(historical_mean),
            "historical_std": float(historical_std),
            "adaptive_flags": adaptive_flags,
            "recommendation_adjustments": [
                f"Consider re-evaluating thresholds for {target}" if adaptive_flags else "Thresholds OK"
            ]
        }

    return insights