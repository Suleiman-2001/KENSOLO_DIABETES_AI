# engines/industry_engine.py

def adapt_recommendations_by_industry(recommendations, industry):
    """
    Adjusts recommendation thresholds and messages based on the industry.
    Returns a modified recommendations dictionary.
    """
    adjusted = {}
    industry_lower = industry.lower()

    for target, rec_list in recommendations.items():
        new_list = []
        for rec in rec_list:
            new_rec = rec.copy()
            if industry_lower == "finance":
                # Finance: be stricter on low predictions
                if rec["category"] == "Low":
                    new_rec["recommendation"] = "Immediate financial review required"
            elif industry_lower == "healthcare":
                if rec["category"] == "Low":
                    new_rec["recommendation"] = "Investigate patient safety risks"
            elif industry_lower == "retail":
                if rec["category"] == "Low":
                    new_rec["recommendation"] = "Optimize sales and inventory immediately"

            new_list.append(new_rec)
        adjusted[target] = new_list

    return adjusted
