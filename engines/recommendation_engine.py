def run_recommendations(predictions):
    """
    Generates actionable recommendations based on predictive model outputs.
    - Regression: classifies predictions into risk/priority levels
    - Classification: returns predicted class as recommendation
    """
    recommendations = {}

    for target, info in predictions.items():
        if "error" in info:
            # Skip targets with errors
            continue

        recs = []
        task_type = info.get("task")
        sample_preds = info.get("sample_predictions", [])

        if task_type == "regression":
            for p in sample_preds:
                try:
                    value = float(p)
                    if value < 50:
                        recs.append({"prediction": value, "recommendation": "Immediate action required"})
                    elif 50 <= value < 75:
                        recs.append({"prediction": value, "recommendation": "Monitor closely"})
                    else:
                        recs.append({"prediction": value, "recommendation": "Maintain current strategy"})
                except Exception:
                    recs.append({"prediction": p, "recommendation": "Invalid prediction value"})

        elif task_type == "classification":
            for p in sample_preds:
                recs.append({"prediction": str(p), "recommendation": f"Predicted class = {p}"})

        # Store recommendations per target
        recommendations[target] = recs

    return recommendations
