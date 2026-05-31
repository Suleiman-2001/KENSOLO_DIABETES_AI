import os
import json
import pandas as pd
from datetime import datetime

MEMORY_FOLDER = r"F:\ARTIFICIAL INTELLIGENCE\AI_Data_Analytics\memory"
os.makedirs(MEMORY_FOLDER, exist_ok=True)

CONFIDENCE_FILE = os.path.join(MEMORY_FOLDER, "confidence_history.json")
DRIFT_FILE = os.path.join(MEMORY_FOLDER, "data_drift_history.json")


# ============================
# 1️⃣ LOAD / SAVE HELPERS
# ============================
def _load_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ============================
# 2️⃣ MEMORY ENGINE CORE
# ============================
def track_dataset_history(df, predictions):
    """
    Diabetes AI Memory Engine

    Tracks:
    - Dataset snapshots
    - Prediction confidence history
    - Data drift indicators (clinical safety)
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ----------------------------
    # 1️⃣ Save dataset snapshot
    # ----------------------------
    snapshot_file = os.path.join(MEMORY_FOLDER, f"dataset_{timestamp}.csv")
    df.to_csv(snapshot_file, index=False)

    # ----------------------------
    # 2️⃣ Load histories
    # ----------------------------
    confidence_history = _load_json(CONFIDENCE_FILE)
    drift_history = _load_json(DRIFT_FILE)

    adjusted_predictions = {}
    confidence_summary = {}

    # ----------------------------
    # 3️⃣ Process predictions
    # ----------------------------
    for target, info in predictions.items():

        current_conf = info.get("confidence", 1.0)

        history = confidence_history.get(target, [])
        history.append(current_conf)

        # keep last 50
        history = history[-50:]
        confidence_history[target] = history

        avg_conf = sum(history) / len(history)

        adjusted_predictions[target] = info.copy()
        adjusted_predictions[target]["confidence"] = round(avg_conf, 4)

        confidence_summary[target] = round(avg_conf, 4)

    # ----------------------------
    # 4️⃣ DATA DRIFT DETECTION (IMPORTANT FOR DIABETES AI)
    # ----------------------------
    numeric_cols = df.select_dtypes(include="number").columns

    drift_report = {}

    for col in numeric_cols:

        col_mean = float(df[col].mean())
        col_std = float(df[col].std())

        previous = drift_history.get(col, {})

        prev_mean = previous.get("mean", col_mean)
        prev_std = previous.get("std", col_std)

        drift_score = abs(col_mean - prev_mean) / (abs(prev_std) + 1e-6)

        drift_report[col] = {
            "mean": col_mean,
            "std": col_std,
            "drift_score": round(drift_score, 4),
            "risk_level": (
                "High" if drift_score > 1.5
                else "Medium" if drift_score > 0.8
                else "Low"
            )
        }

        # update drift memory
        drift_history[col] = {
            "mean": col_mean,
            "std": col_std
        }

    # ----------------------------
    # 5️⃣ SAVE UPDATED MEMORY
    # ----------------------------
    _save_json(CONFIDENCE_FILE, confidence_history)
    _save_json(DRIFT_FILE, drift_history)

    summary = {
        "snapshot_file": snapshot_file,
        "row_count": len(df),
        "columns": list(df.columns),
        "confidence_summary": confidence_summary,
        "data_drift": drift_report
    }

    return adjusted_predictions, summary