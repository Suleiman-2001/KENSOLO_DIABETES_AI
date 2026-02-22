# engines/recommendation_engine.py
from joblib import Parallel, delayed

def _process_regression_prediction(p, lower_threshold=50, upper_threshold=75):
    """
    Process a single regression prediction.
    Adds configurable thresholds for alerts.
    """
    try:
        value = float(p)
        if value < lower_threshold:
            return {"prediction": value, "recommendation": "Immediate action required"}
        elif lower_threshold <= value < upper_threshold:
            return {"prediction": value, "recommendation": "Monitor closely"}
        else:
            return {"prediction": value, "recommendation": "Maintain current strategy"}
    except Exception:
        return {"prediction": p, "recommendation": "Invalid prediction value"}


def _process_classification_prediction(p):
    """
    Process a single classification prediction.
    """
    try:
        return {"prediction": str(p), "recommendation": f"Predicted class = {p}"}
    except Exception:
        return {"prediction": p, "recommendation": "Invalid classification value"}


def run_recommendations(predictions, parallel_threshold=10, max_jobs=2, max_flags=1000):
    """
    Safe and optimized recommendation engine:
    - Sequential for small prediction arrays (fast, avoids joblib overhead)
    - Parallel for large prediction arrays
    - Limits memory usage for huge datasets
    - Streamlit-friendly: avoids freezing on tiny or large datasets
    - Automatically handles invalid or empty predictions

    Parameters:
    - predictions: dict of target predictions from predictive engine
    - parallel_threshold: min number of predictions to trigger parallel processing
    - max_jobs: max parallel jobs for large arrays
    - max_flags: maximum number of predictions to process per target (default 1000)
    """
    recommendations = {}

    for target, info in predictions.items():
        if "error" in info:
            recommendations[target] = [{"prediction": None, "recommendation": "Skipped due to error"}]
            continue  # skip targets with errors

        task_type = info.get("task")
        sample_preds = info.get("sample_predictions", [])

        # Safety: if empty predictions, skip
        if not sample_preds:
            recommendations[target] = [{"prediction": None, "recommendation": "No predictions available"}]
            continue

        # Limit number of predictions processed to avoid memory spikes
        if max_flags is not None:
            sample_preds = sample_preds[:max_flags]

        # Determine whether to run in parallel
        use_parallel = len(sample_preds) >= parallel_threshold

        # Process predictions safely
        try:
            if task_type == "regression":
                if use_parallel:
                    recs = Parallel(n_jobs=max_jobs)(
                        delayed(_process_regression_prediction)(p) for p in sample_preds
                    )
                else:
                    recs = [_process_regression_prediction(p) for p in sample_preds]

            elif task_type == "classification":
                if use_parallel:
                    recs = Parallel(n_jobs=max_jobs)(
                        delayed(_process_classification_prediction)(p) for p in sample_preds
                    )
                else:
                    recs = [_process_classification_prediction(p) for p in sample_preds]

            else:
                # Unknown task type
                recs = [{"prediction": p, "recommendation": "Unknown task type"} for p in sample_preds]

        except Exception as e:
            # Catch any unexpected errors to prevent hanging
            recs = [{"prediction": p, "recommendation": f"Error processing prediction: {e}"} for p in sample_preds]

        recommendations[target] = recs

    return recommendations