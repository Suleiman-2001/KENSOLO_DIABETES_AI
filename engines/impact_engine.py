# engines/impact_engine.py

def calculate_revenue_cost_impact(predictions, business_insights):
    """
    Converts model predictions + business insights into estimated revenue/cost impacts.
    Returns a dictionary with impact analysis.
    """
    impact = {}

    for target, info in predictions.items():
        predicted_vals = info.get("sample_predictions", [])
        avg_prediction = sum(predicted_vals)/len(predicted_vals) if predicted_vals else 0

        # Simple impact logic (customize for real BI data)
        revenue_impact = avg_prediction * 1000  # placeholder multiplier
        cost_impact = avg_prediction * 500      # placeholder multiplier

        impact[target] = {
            "avg_prediction": avg_prediction,
            "estimated_revenue_impact": revenue_impact,
            "estimated_cost_impact": cost_impact
        }

    return impact
