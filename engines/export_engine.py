import pandas as pd
import os
import json


def save_to_excel(output_dict, folder_path="outputs/excel_exports"):
    """
    Diabetes AI Export Engine (Clinical + ML Output Formatter)

    Converts:
    - predictions
    - recommendations
    - problem discovery
    - business/clinical intelligence

    Into structured Excel reports for medical/analytics users.
    """

    os.makedirs(folder_path, exist_ok=True)
    excel_file = os.path.join(folder_path, "DIABETES_AI_REPORT.xlsx")

    with pd.ExcelWriter(excel_file, engine="xlsxwriter") as writer:

        # ============================
        # 1️⃣ PROBLEM DISCOVERY
        # ============================
        problems = output_dict.get("problem_discovery", {})

        if isinstance(problems, dict) and problems:
            df_problems = pd.DataFrame.from_dict(problems, orient="index")
            df_problems.to_excel(writer, sheet_name="Problem_Discovery")

        # ============================
        # 2️⃣ PREDICTIONS + RECOMMENDATIONS
        # ============================
        predictions = output_dict.get("predictions", {})
        recommendations = output_dict.get("recommendations", {})

        for target, pred_data in predictions.items():

            if not isinstance(pred_data, dict):
                continue

            preds = pred_data.get("sample_predictions", [])

            recs_list = recommendations.get(target, [])

            rows = []

            for i, pred in enumerate(preds):

                rec = recs_list[i] if i < len(recs_list) else {}

                if isinstance(rec, dict):
                    category = rec.get("category", "N/A")
                    recommendation = rec.get("recommendation", "N/A")
                else:
                    category = "N/A"
                    recommendation = "N/A"

                rows.append({
                    "prediction_value": pred,
                    "risk_category": category,
                    "clinical_recommendation": recommendation
                })

            df_pred = pd.DataFrame(rows)

            sheet_name = f"Pred_{target}"[:31]
            df_pred.to_excel(writer, sheet_name=sheet_name, index=False)

        # ============================
        # 3️⃣ FULL RECOMMENDATIONS
        # ============================
        for target, rec_list in recommendations.items():

            if isinstance(rec_list, list) and rec_list:

                df_recs = pd.DataFrame(rec_list)
                sheet_name = f"Recs_{target}"[:31]
                df_recs.to_excel(writer, sheet_name=sheet_name, index=False)

        # ============================
        # 4️⃣ CLINICAL / BUSINESS INTELLIGENCE
        # ============================
        bi = output_dict.get("business_intelligence", {})

        if isinstance(bi, dict):

            for key, value in bi.items():

                try:
                    if isinstance(value, dict):
                        df_bi = pd.DataFrame.from_dict(value, orient="index")
                    elif isinstance(value, list):
                        df_bi = pd.DataFrame(value)
                    else:
                        df_bi = pd.DataFrame({key: [value]})

                    sheet_name = f"BI_{key}"[:31]
                    df_bi.to_excel(writer, sheet_name=sheet_name, index=True)

                except Exception:
                    df_fallback = pd.DataFrame({"value": [str(value)]})
                    df_fallback.to_excel(writer, sheet_name=f"BI_{key}"[:31], index=False)

        # ============================
        # 5️⃣ META SUMMARY SHEET
        # ============================
        summary = {
            "total_targets": len(predictions),
            "total_recommendation_groups": len(recommendations),
            "has_problem_discovery": bool(problems),
        }

        df_summary = pd.DataFrame([summary])
        df_summary.to_excel(writer, sheet_name="SUMMARY", index=False)

    return excel_file