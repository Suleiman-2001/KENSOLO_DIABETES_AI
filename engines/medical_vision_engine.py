import os
import matplotlib.pyplot as plt
import json
import pandas as pd


# ============================
# 1️⃣ CLINICAL GRAPH ENGINE
# ============================
def generate_graphs(df, targets_dict, folder="outputs/graphs"):
    """
    Diabetes Clinical Vision Engine

    Generates:
    - Distributions
    - Risk threshold overlays (glucose, BMI, BP, age)
    """

    os.makedirs(folder, exist_ok=True)
    graph_paths = []

    clinical_thresholds = {
        "glucose": 140,
        "bmi": 30,
        "bloodpressure": 140,
        "bp": 140,
        "age": 50
    }

    for t in targets_dict.get("numerical", []) + targets_dict.get("categorical", []):

        if t not in df.columns:
            continue

        try:
            data = pd.to_numeric(df[t], errors="coerce").dropna()
            if data.empty:
                continue

            plt.figure(figsize=(6, 4))
            plt.hist(data, bins=25, alpha=0.7)

            # Add clinical risk line if applicable
            t_lower = t.lower()

            for key, threshold in clinical_thresholds.items():
                if key in t_lower:
                    plt.axvline(threshold, color="red", linestyle="--", linewidth=2)

            plt.title(f"{t} Distribution (Clinical View)")
            plt.xlabel(t)
            plt.ylabel("Frequency")

            path = os.path.join(folder, f"{t}_clinical.png")
            plt.savefig(path, bbox_inches="tight")
            plt.close()

            graph_paths.append(path)

        except Exception:
            continue

    print(f"📊 Clinical graphs saved: {folder}")
    return graph_paths


# ============================
# 2️⃣ EXPORT ENGINE (MEDICAL SAFE FORMAT)
# ============================
def save_predictions_and_recommendations(predictions, recommendations, folder="outputs"):
    """
    Clinical export engine for diabetes AI system
    """

    os.makedirs(folder, exist_ok=True)

    # ----------------------------
    # JSON EXPORTS
    # ----------------------------
    with open(os.path.join(folder, "predictions.json"), "w") as f:
        json.dump(predictions, f, indent=4)

    with open(os.path.join(folder, "recommendations.json"), "w") as f:
        json.dump(recommendations, f, indent=4)

    # ----------------------------
    # FLATTENED PREDICTIONS CSV
    # ----------------------------
    pred_rows = []

    for target, info in predictions.items():

        if not isinstance(info, dict):
            continue

        preds = info.get("sample_predictions", [])

        for i, val in enumerate(preds):
            pred_rows.append({
                "target": target,
                "index": i,
                "prediction_value": val,
                "task_type": info.get("task", "unknown"),
                "model": info.get("best_model", "unknown")
            })

    pd.DataFrame(pred_rows).to_csv(
        os.path.join(folder, "predictions.csv"),
        index=False
    )

    # ----------------------------
    # CLINICAL RECOMMENDATIONS CSV
    # ----------------------------
    rec_rows = []

    for target, rec_list in recommendations.items():

        if not isinstance(rec_list, list):
            continue

        for i, rec in enumerate(rec_list):

            rec_rows.append({
                "target": target,
                "index": i,
                "prediction": rec.get("prediction"),
                "risk_category": rec.get("category"),
                "clinical_action": rec.get("recommendation")
            })

    pd.DataFrame(rec_rows).to_csv(
        os.path.join(folder, "recommendations.csv"),
        index=False
    )

    # ----------------------------
    # SUMMARY FILE (IMPORTANT FOR MEDICAL USERS)
    # ----------------------------
    summary = {
        "total_models": len(predictions),
        "total_targets": len(recommendations),
        "clinical_mode": True
    }

    pd.DataFrame([summary]).to_csv(
        os.path.join(folder, "summary.csv"),
        index=False
    )

    print(f"✅ Clinical exports saved in: {folder}")

    return {
        "predictions_json": os.path.join(folder, "predictions.json"),
        "recommendations_json": os.path.join(folder, "recommendations.json"),
        "predictions_csv": os.path.join(folder, "predictions.csv"),
        "recommendations_csv": os.path.join(folder, "recommendations.csv"),
        "summary_csv": os.path.join(folder, "summary.csv"),
    }