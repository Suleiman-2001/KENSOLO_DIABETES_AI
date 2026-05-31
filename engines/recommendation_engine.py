from joblib import Parallel, delayed


def _process_regression_prediction(p, lower_threshold=0.3, upper_threshold=0.7):
    """
    Clinical regression interpretation (risk probability style)
    """

    try:
        value = float(p)

        if value < lower_threshold:
            return {
                "prediction": value,
                "risk_level": "Low",
                "category": "Low",
                "recommendation": "No immediate diabetes risk detected"
            }

        elif lower_threshold <= value < upper_threshold:
            return {
                "prediction": value,
                "risk_level": "Medium",
                "category": "Medium",
                "recommendation": "Monitor metabolic indicators regularly"
            }

        else:
            return {
                "prediction": value,
                "risk_level": "High",
                "category": "High",
                "recommendation": "High diabetes risk - recommend clinical assessment"
            }

    except Exception:
        return {
            "prediction": p,
            "risk_level": "Unknown",
            "recommendation": "Invalid risk score"
        }


def _process_classification_prediction(p):
    """
    Diabetes classification interpretation
    """

    try:
        label = str(p).lower()

        if "diabetes" in label or label in ["1", "yes", "positive"]:
            return {
                "prediction": p,
                "risk_level": "High",
                "category": "High",
                "recommendation": "Diabetes detected - clinical follow-up required"
            }

        elif label in ["0", "no", "negative"]:
            return {
                "prediction": p,
                "risk_level": "Low",
                "category": "Low",
                "recommendation": "No diabetes detected - maintain healthy lifestyle"
            }

        else:
            return {
                "prediction": p,
                "risk_level": "Unknown",
                "recommendation": "Unclear classification output"
            }

    except Exception:
        return {
            "prediction": p,
            "risk_level": "Unknown",
            "recommendation": "Invalid classification value"
        }


def run_recommendations(predictions, parallel_threshold=10, max_jobs=2, max_flags=1000):
    """
    Diabetes Clinical Recommendation Engine

    Converts predictions into:
    - Risk levels (Low / Medium / High)
    - Clinical recommendations
    - Population-level insights
    """

    recommendations = {}

    for target, info in predictions.items():

        if "error" in info:
            recommendations[target] = [{
                "prediction": None,
                "risk_level": "Unknown",
                "recommendation": "Model error - cannot generate clinical advice"
            }]
            continue

        task_type = info.get("task")
        sample_preds = info.get("sample_predictions", [])

        if not sample_preds:
            recommendations[target] = [{
                "prediction": None,
                "risk_level": "Unknown",
                "recommendation": "No prediction data available"
            }]
            continue

        sample_preds = sample_preds[:max_flags]
        use_parallel = len(sample_preds) >= parallel_threshold

        try:
            if task_type == "regression":

                if use_parallel:
                    recs = Parallel(n_jobs=max_jobs)(
                        delayed(_process_regression_prediction)(p)
                        for p in sample_preds
                    )
                else:
                    recs = [_process_regression_prediction(p) for p in sample_preds]

            elif task_type == "classification":

                if use_parallel:
                    recs = Parallel(n_jobs=max_jobs)(
                        delayed(_process_classification_prediction)(p)
                        for p in sample_preds
                    )
                else:
                    recs = [_process_classification_prediction(p) for p in sample_preds]

            else:
                recs = [{
                    "prediction": p,
                    "risk_level": "Unknown",
                    "recommendation": "Unsupported model type for clinical interpretation"
                } for p in sample_preds]

        except Exception as e:
            recs = [{
                "prediction": p,
                "risk_level": "Error",
                "recommendation": f"Processing failure: {str(e)}"
            } for p in sample_preds]

        recommendations[target] = recs

    return recommendations