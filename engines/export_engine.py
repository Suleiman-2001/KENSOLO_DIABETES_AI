# engines/export_engine.py
import pandas as pd
import os

def save_to_excel(output_dict, folder_path="outputs/excel_exports"):
    """
    Converts AI outputs (problem discovery, predictions, recommendations,
    business intelligence) into Excel sheets for business users.
    """
    os.makedirs(folder_path, exist_ok=True)
    excel_file = os.path.join(folder_path, "KENSOLO_AI_Report.xlsx")

    with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:

        # -------------------------------
        # 1️⃣ Problem Discovery
        # -------------------------------
        problems = output_dict.get("problem_discovery", {})
        if problems:
            df_problems = pd.DataFrame.from_dict(problems, orient="index")
            df_problems.to_excel(writer, sheet_name="Problem_Discovery", index=True)

        # -------------------------------
        # 2️⃣ Predictions + Recommendations
        # -------------------------------
        predictions = output_dict.get("predictions", {})
        recommendations = output_dict.get("recommendations", {})

        for target, pred_data in predictions.items():
            # Get sample predictions
            preds = pred_data.get("sample_predictions", [])
            
            # Get corresponding recommendations if available
            recs_list = recommendations.get(target, [])
            
            # Align categories and recommendations
            categories = []
            rec_texts = []
            for i, p in enumerate(preds):
                if i < len(recs_list) and isinstance(recs_list[i], dict):
                    categories.append(recs_list[i].get("category", "N/A"))
                    rec_texts.append(recs_list[i].get("recommendation", "N/A"))
                else:
                    categories.append("N/A")
                    rec_texts.append("N/A")

            df_pred = pd.DataFrame({
                "prediction": preds,
                "category": categories,
                "recommendation": rec_texts
            })

            # Excel sheet name max 31 chars
            sheet_name = f"Pred_{target}"[:31]
            df_pred.to_excel(writer, sheet_name=sheet_name, index=False)

        # -------------------------------
        # 3️⃣ Recommendations (full details)
        # -------------------------------
        for target, rec_list in recommendations.items():
            if rec_list:
                df_recs = pd.DataFrame(rec_list)
                sheet_name = f"Recs_{target}"[:31]
                df_recs.to_excel(writer, sheet_name=sheet_name, index=False)

        # -------------------------------
        # 4️⃣ Business Intelligence
        # -------------------------------
        bi = output_dict.get("business_intelligence", {})
        for key, value in bi.items():
            # Convert dict of dicts to dataframe
            if isinstance(value, dict):
                try:
                    df_bi = pd.DataFrame.from_dict(value, orient="index")
                except:
                    df_bi = pd.DataFrame({key: value})
                sheet_name = f"BI_{key}"[:31]
                df_bi.to_excel(writer, sheet_name=sheet_name, index=True)

    return excel_file
