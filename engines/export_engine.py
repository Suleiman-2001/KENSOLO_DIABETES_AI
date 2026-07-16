import pandas as pd
import os
import json


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _prediction_rows(predictions_dict):
    rows = []

    for target, pred_data in (predictions_dict or {}).items():
        if isinstance(pred_data, dict):
            sample_predictions = pred_data.get("sample_predictions")
            sample_probabilities = pred_data.get("sample_probabilities") or []
            sample_risk_scores = pred_data.get("sample_risk_scores") or []

            if isinstance(sample_predictions, list) and sample_predictions:
                for index, prediction in enumerate(sample_predictions):
                    row = {
                        "target": target,
                        "prediction_value": prediction,
                    }
                    if index < len(sample_probabilities):
                        row["probability"] = sample_probabilities[index]
                    if index < len(sample_risk_scores):
                        row["risk_score"] = sample_risk_scores[index]
                    rows.append(row)
            else:
                row = {str(key): _json_safe(value) for key, value in pred_data.items()}
                row["target"] = target
                rows.append(row)

        elif isinstance(pred_data, list):
            for item in pred_data:
                row = item.copy() if isinstance(item, dict) else {"prediction_value": item}
                row["target"] = target
                rows.append(row)

        else:
            rows.append({"target": target, "prediction_value": pred_data})

    return rows


def _prediction_summary_rows(predictions_dict):
    summary_rows = []

    for target, pred_data in (predictions_dict or {}).items():
        if not isinstance(pred_data, dict):
            summary_rows.append({"target": target, "prediction_value": _json_safe(pred_data)})
            continue

        summary_rows.append(
            {
                "target": target,
                "task": pred_data.get("task"),
                "target_source": pred_data.get("target_source"),
                "best_model": pred_data.get("best_model"),
                "confidence": pred_data.get("confidence"),
                "cv_primary_metric": pred_data.get("cv_primary_metric"),
                "cv_primary_score": pred_data.get("cv_primary_score"),
                "accuracy": pred_data.get("accuracy"),
                "balanced_accuracy": pred_data.get("balanced_accuracy"),
                "f1_score": pred_data.get("f1_score"),
                "precision": pred_data.get("precision"),
                "recall": pred_data.get("recall"),
                "roc_auc": pred_data.get("roc_auc"),
            }
        )

    return summary_rows


def _prediction_sample_rows(predictions_dict):
    rows = []

    for target, pred_data in (predictions_dict or {}).items():
        if not isinstance(pred_data, dict):
            continue

        sample_predictions = pred_data.get("sample_predictions") or []
        sample_probabilities = pred_data.get("sample_probabilities") or []
        sample_risk_scores = pred_data.get("sample_risk_scores") or []

        for index, prediction in enumerate(sample_predictions):
            row = {
                "target": target,
                "sample_index": index,
                "prediction_value": prediction,
            }
            if index < len(sample_probabilities):
                row["probability"] = sample_probabilities[index]
            if index < len(sample_risk_scores):
                row["risk_score"] = sample_risk_scores[index]
            rows.append(row)

    return rows


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

        summary_rows = _prediction_summary_rows(predictions)
        if summary_rows:
            df_pred_summary = pd.DataFrame(summary_rows)
            df_pred_summary.to_excel(writer, sheet_name="Pred_Summary", index=False)

        sample_rows = _prediction_sample_rows(predictions)
        if sample_rows:
            df_pred_samples = pd.DataFrame(sample_rows)
            df_pred_samples.to_excel(writer, sheet_name="Pred_Samples", index=False)

        for target, pred_data in predictions.items():
            recs_list = recommendations.get(target, [])
            if not (isinstance(pred_data, dict) and isinstance(pred_data.get("sample_predictions"), list)):
                continue

            target_rows = []
            sample_predictions = pred_data.get("sample_predictions") or []
            sample_probabilities = pred_data.get("sample_probabilities") or []
            sample_risk_scores = pred_data.get("sample_risk_scores") or []

            for index, prediction in enumerate(sample_predictions):
                rec = recs_list[index] if index < len(recs_list) else {}
                row = {
                    "prediction_value": prediction,
                    "sample_index": index,
                    "risk_category": rec.get("category", "N/A") if isinstance(rec, dict) else "N/A",
                    "clinical_recommendation": rec.get("recommendation", "N/A") if isinstance(rec, dict) else "N/A",
                }
                if index < len(sample_probabilities):
                    row["probability"] = sample_probabilities[index]
                if index < len(sample_risk_scores):
                    row["risk_score"] = sample_risk_scores[index]
                target_rows.append(row)

            if target_rows:
                pd.DataFrame(target_rows).to_excel(writer, sheet_name=f"Pred_{target}"[:31], index=False)

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
        # 4b️⃣ AI MODEL GOVERNANCE
        # ============================
        extra_sections = {
            "feature_engineering": output_dict.get("feature_engineering", {}),
            "risk_scoring": output_dict.get("risk_scoring", {}),
            "model_monitoring": output_dict.get("model_monitoring", {}),
            "diabetes_detection": output_dict.get("diabetes_detection", {}),
            "task_detection": output_dict.get("task_detection", {}),
            "unsupervised_learning": output_dict.get("unsupervised_learning", {}),
            "model_leaderboard": output_dict.get("model_leaderboard", []),
            "shap_explanations": output_dict.get("shap_explanations", {}),
        }

        for key, value in extra_sections.items():
            if not value:
                continue

            try:
                if isinstance(value, dict):
                    if any(isinstance(v, (dict, list)) for v in value.values()):
                        df_extra = pd.json_normalize(value, sep="_")
                    else:
                        df_extra = pd.DataFrame([_json_safe(value)])
                elif isinstance(value, list):
                    df_extra = pd.json_normalize(value) if value and isinstance(value[0], dict) else pd.DataFrame({"value": value})
                else:
                    df_extra = pd.DataFrame({"value": [str(value)]})

                sheet_name = f"AI_{key}"[:31]
                df_extra.to_excel(writer, sheet_name=sheet_name, index=False)

            except Exception:
                df_fallback = pd.DataFrame({"value": [str(value)]})
                df_fallback.to_excel(writer, sheet_name=f"AI_{key}"[:31], index=False)

        # ============================
        # 5️⃣ META SUMMARY SHEET
        # ============================
        summary = {
            "total_targets": len(predictions),
            "total_recommendation_groups": len(recommendations),
            "has_problem_discovery": bool(problems),
            "has_model_monitoring": bool(output_dict.get("model_monitoring")),
            "has_risk_scoring": bool(output_dict.get("risk_scoring")),
            "detected_task": output_dict.get("task_detection", {}).get("task", "unknown"),
            "has_unsupervised_learning": bool(output_dict.get("unsupervised_learning")),
        }

        df_summary = pd.DataFrame([summary])
        df_summary.to_excel(writer, sheet_name="SUMMARY", index=False)

    return excel_file