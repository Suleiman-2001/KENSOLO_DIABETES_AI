def adapt_recommendations_by_industry(recommendations, industry):
    """
    Clinical Industry Engine (Diabetes-Aware Version)

    Converts generic recommendations into medical decision support language.
    """

    adjusted = {}
    industry_lower = (industry or "").lower()

    for target, rec_list in recommendations.items():
        new_list = []

        for rec in rec_list:
            new_rec = rec.copy()

            category = rec.get("category", "").lower()

            # =====================================================
            # 🏥 DIABETES / HEALTHCARE MODE (MAIN FOCUS)
            # =====================================================
            if industry_lower in ["healthcare", "diabetes", "clinical", "medical"]:

                if category == "low":
                    new_rec["recommendation"] = "⚠️ High metabolic risk detected – immediate patient review required"
                    new_rec["clinical_action"] = "Order glucose and HbA1c assessment"

                elif category == "medium":
                    new_rec["recommendation"] = "🧠 Moderate risk – monitor patient trends closely"
                    new_rec["clinical_action"] = "Schedule follow-up and lifestyle intervention"

                elif category == "high":
                    new_rec["recommendation"] = "✅ Stable metabolic profile"
                    new_rec["clinical_action"] = "Continue standard monitoring protocol"

            # =====================================================
            # FINANCE (kept for compatibility)
            # =====================================================
            elif industry_lower == "finance":
                if category == "low":
                    new_rec["recommendation"] = "Immediate financial risk review required"

            # =====================================================
            # RETAIL (kept for compatibility)
            # =====================================================
            elif industry_lower == "retail":
                if category == "low":
                    new_rec["recommendation"] = "Optimize demand forecasting and inventory"

            # =====================================================
            # DEFAULT FALLBACK
            # =====================================================
            else:
                if category == "low":
                    new_rec["recommendation"] = "Investigate anomaly in data"

            new_list.append(new_rec)

        adjusted[target] = new_list

    return adjusted