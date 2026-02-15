# engines/memory_engine.py
import os
import pandas as pd
import json
from datetime import datetime

MEMORY_FOLDER = r"F:\ARTIFICIAL INTELLIGENCE\AI_Data_Analytics\memory"
os.makedirs(MEMORY_FOLDER, exist_ok=True)

CONFIDENCE_TRACK_FILE = os.path.join(MEMORY_FOLDER, "confidence_history.json")

def _load_confidence_history():
    """Load previous prediction confidences."""
    if os.path.exists(CONFIDENCE_TRACK_FILE):
        with open(CONFIDENCE_TRACK_FILE, "r") as f:
            return json.load(f)
    return {}

def _save_confidence_history(history):
    """Save updated confidence history."""
    with open(CONFIDENCE_TRACK_FILE, "w") as f:
        json.dump(history, f, indent=2)

def track_dataset_history(df, predictions):
    """
    Stores dataset history and prediction confidence for future reference.
    Adjusts confidence based on historical trends.
    Returns updated predictions with adjusted confidence and a summary dict.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_file = os.path.join(MEMORY_FOLDER, f"dataset_history_{timestamp}.csv")
    df.to_csv(history_file, index=False)

    # Load historical confidence
    history = _load_confidence_history()

    adjusted_predictions = {}
    confidence_summary = {}

    for target, info in predictions.items():
        # Current confidence (default 1.0 if missing)
        current_conf = info.get("confidence", 1.0)

        # Historical confidence
        hist_conf_list = history.get(target, [])
        if hist_conf_list:
            avg_hist_conf = sum(hist_conf_list)/len(hist_conf_list)
            # Adjust current confidence: simple weighted average (50% history, 50% new)
            adjusted_conf = 0.5 * current_conf + 0.5 * avg_hist_conf
        else:
            adjusted_conf = current_conf

        # Update history
        hist_conf_list.append(current_conf)
        if len(hist_conf_list) > 50:  # Keep last 50 entries
            hist_conf_list = hist_conf_list[-50:]
        history[target] = hist_conf_list

        # Save adjusted confidence in predictions
        adjusted_predictions[target] = info.copy()
        adjusted_predictions[target]["confidence"] = adjusted_conf
        confidence_summary[target] = adjusted_conf

    # Save updated confidence history
    _save_confidence_history(history)

    summary = {
        "history_file": history_file,
        "confidence_summary": confidence_summary,
        "row_count": len(df),
        "columns": list(df.columns)
    }

    return adjusted_predictions, summary
