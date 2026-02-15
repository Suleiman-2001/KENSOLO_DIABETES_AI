# engines/assumption_engine.py

def run_scenario_analysis(predictions, business_insights, recommendations):
    """
    Performs scenario analysis based on assumptions in predictions and business context.
    Returns a scenario analysis dictionary.
    """
    scenarios = {}

    for target, info in predictions.items():
        sample_vals = info.get("sample_predictions", [])
        scenarios[target] = {
            "best_case": max(sample_vals) if sample_vals else None,
            "worst_case": min(sample_vals) if sample_vals else None,
            "most_likely": sum(sample_vals)/len(sample_vals) if sample_vals else None,
            "recommendation_summary": recommendations.get(target, [])
        }

    return scenarios
