# engines/assumption_engine.py
import numpy as np

def run_scenario_analysis(predictions, business_insights=None, recommendations=None):
    """
    Scenario Analysis Engine

    Builds:
    - Best case (optimistic bound)
    - Worst case (risk bound)
    - Most likely (robust central estimate)
    - Risk-adjusted scenario framing

    Designed for decision intelligence systems.
    """

    scenarios = {}

    for target, info in predictions.items():

        sample_vals = np.asarray(info.get("sample_predictions", []), dtype=np.float64)
        sample_vals = sample_vals[~np.isnan(sample_vals)]

        if sample_vals.size == 0:
            scenarios[target] = {
                "best_case": None,
                "worst_case": None,
                "most_likely": None,
                "risk_level": "Unknown",
                "recommendation_summary": []
            }
            continue

        # ----------------------------
        # Core statistics
        # ----------------------------
        mean = float(sample_vals.mean())
        std = float(sample_vals.std())

        # More stable than raw min/max
        best_case = float(mean + 1.5 * std)
        worst_case = float(mean - 1.5 * std)

        # Most likely = median (more robust than mean)
        most_likely = float(np.median(sample_vals))

        # ----------------------------
        # Risk classification
        # ----------------------------
        if std / (abs(mean) + 1e-6) > 0.5:
            risk_level = "High volatility"
        elif std / (abs(mean) + 1e-6) > 0.2:
            risk_level = "Medium volatility"
        else:
            risk_level = "Stable"

        # ----------------------------
        # Recommendation alignment
        # ----------------------------
        recs = recommendations.get(target, []) if recommendations else []

        scenarios[target] = {
            "best_case": best_case,
            "worst_case": worst_case,
            "most_likely": most_likely,
            "mean": mean,
            "std": std,
            "risk_level": risk_level,
            "recommendation_summary": recs[:5]  # avoid overload
        }

    return scenarios