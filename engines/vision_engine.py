import os
import matplotlib.pyplot as plt
import json
import pandas as pd

# ----------------------------
# 1️⃣ Graph generation
# ----------------------------
def generate_graphs(df, targets_dict, folder="outputs/graphs"):
    os.makedirs(folder, exist_ok=True)
    graph_paths = []

    for t in targets_dict.get("numerical", []) + targets_dict.get("categorical", []):
        if t in df.columns:
            plt.figure()
            df[t].hist(bins=20)
            plt.title(f"Distribution of {t}")
            plt.xlabel(t)
            plt.ylabel("Count")
            path = os.path.join(folder, f"{t}.png")
            plt.savefig(path)
            plt.close()
            graph_paths.append(path)

    print(f"✅ Graphs saved in folder: {folder}")
    return graph_paths

# ----------------------------
# 2️⃣ Save predictions & recommendations
# ----------------------------
def save_predictions_and_recommendations(predictions, recommendations, folder="outputs"):
    os.makedirs(folder, exist_ok=True)

    # Save JSON files
    with open(os.path.join(folder, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=4)

    with open(os.path.join(folder, "recommendations.json"), "w", encoding="utf-8") as f:
        json.dump(recommendations, f, indent=4)

    # Save predictions to CSV
    pred_rows = []
    for target, info in predictions.items():
        if "sample_predictions" in info:
            for i, val in enumerate(info["sample_predictions"]):
                pred_rows.append({"target": target, "row": i, "prediction": val})
        else:
            pred_rows.append({"target": target, "row": None, "prediction": str(info)})
    pd.DataFrame(pred_rows).to_csv(os.path.join(folder, "predictions.csv"), index=False)

    # Save recommendations to CSV
    rec_rows = []
    for target, rec_list in recommendations.items():
        for i, rec in enumerate(rec_list):
            rec_rows.append({
                "target": target,
                "row": i,
                "prediction": rec.get("prediction"),
                "category": rec.get("category"),
                "recommendation": rec.get("recommendation"),
            })
    pd.DataFrame(rec_rows).to_csv(os.path.join(folder, "recommendations.csv"), index=False)

    print(f"✅ Predictions and recommendations saved in folder: {folder}")
    return {
        "predictions_json": os.path.join(folder, "predictions.json"),
        "recommendations_json": os.path.join(folder, "recommendations.json"),
        "predictions_csv": os.path.join(folder, "predictions.csv"),
        "recommendations_csv": os.path.join(folder, "recommendations.csv"),
    }
