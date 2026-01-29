# engines/recommendation_engine.py
def categorize_prediction(value):
    """Bin numeric predictions into Low/Medium/High"""
    if value < 50:
        return "Low"
    elif value < 80:
        return "Medium"
    else:
        return "High"

def run_recommendations(predictions):
    """
    Generates recommendations from predictions.
    Works for regression and classification outputs.
    """
    recs = {}
    for target, result in predictions.items():
        recs[target] = []
        if "sample_predictions" in result:
            for pred in result["sample_predictions"]:
                if isinstance(pred, (int, float)):
                    category = categorize_prediction(pred)
                else:
                    category = str(pred)
                action = {
                    "Low": "Immediate support required",
                    "Medium": "Monitor progress",
                    "High": "Offer advanced opportunities"
                }.get(category, "Review")
                recs[target].append({"prediction": pred, "category": category, "recommendation": action})
    return recs
