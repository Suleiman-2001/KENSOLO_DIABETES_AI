# engines/why_engine.py
import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except Exception:
    LIME_AVAILABLE = False


def _prepare_explanation_inputs(model_pipeline, X):
    """Return a numeric feature matrix suitable for SHAP/LIME."""
    candidate = X.copy()

    if hasattr(model_pipeline, "named_steps") and len(model_pipeline.named_steps) > 1:
        preprocessor = model_pipeline[:-1]
        expected_columns = getattr(preprocessor, "feature_names_in_", None)
        if expected_columns is not None:
            candidate = candidate.reindex(columns=list(expected_columns), fill_value=np.nan)

    # Prefer pipeline preprocessing if the model is a Pipeline.
    if hasattr(model_pipeline, "named_steps") and len(model_pipeline.named_steps) > 1:
        try:
            preprocessor = model_pipeline[:-1]
            transformed = preprocessor.transform(candidate)
            if hasattr(transformed, "toarray"):
                transformed = transformed.toarray()
            transformed_array = np.asarray(transformed, dtype=float)

            if hasattr(preprocessor, "get_feature_names_out"):
                feature_names = list(preprocessor.get_feature_names_out())
            else:
                feature_names = [f"feature_{i}" for i in range(transformed_array.shape[1])]

            transformed_df = pd.DataFrame(
                transformed_array,
                columns=feature_names,
                index=candidate.index,
            )
            return transformed_df, transformed_array, model_pipeline.named_steps.get("model", model_pipeline)
        except Exception:
            pass

    # Fallback: one-hot encode categoricals for safe explainability.
    try:
        safe_df = pd.get_dummies(candidate, dummy_na=True)
        safe_df = safe_df.astype(float)
        return safe_df, np.asarray(safe_df, dtype=float), model_pipeline
    except Exception:
        numeric_df = candidate.select_dtypes(include=[np.number]).copy()
        if numeric_df.shape[1] == 0:
            numeric_df = candidate.copy()
            for col in numeric_df.columns:
                numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce")
        numeric_df = numeric_df.fillna(0).astype(float)
        return numeric_df, np.asarray(numeric_df, dtype=float), model_pipeline


def explain_predictions(model_pipeline, df, target_column, top_n=5):
    """
    Explain model outputs with SHAP and, when available, LIME.

    Parameters:
        model_pipeline : trained sklearn Pipeline
        df             : DataFrame with features
        target_column  : str, column name of target
        top_n          : number of top features to show

    Returns:
        explanation_dict : dict containing SHAP/LIME explanation details
    """
    explanation_dict = {
        "target": target_column,
        "feature_importance": [],
        "sample_explanations": [],
        "lime_explanations": [],
        "explainability_methods": {
            "shap": False,
            "lime": False,
        },
    }

    try:
        X = df.drop(columns=[target_column])
        if X.empty:
            return {**explanation_dict, "error": "No feature columns available"}

        explanation_df, explanation_array, estimator = _prepare_explanation_inputs(model_pipeline, X)
        if explanation_array.size == 0:
            return {**explanation_dict, "error": "Unable to build numeric explanation matrix"}

        # --- SHAP explanation ---
        if SHAP_AVAILABLE and explanation_array.ndim == 2:
            try:
                if hasattr(estimator, "predict_proba"):
                    explainer = shap.Explainer(estimator.predict_proba, explanation_df)
                    shap_values = explainer(explanation_df)
                    values = np.asarray(shap_values.values)
                    if values.ndim == 3:
                        values = values[..., -1]
                    if values.ndim == 2 and values.shape[1] == 2:
                        values = values[:, 1]
                else:
                    explainer = shap.Explainer(estimator.predict, explanation_df)
                    shap_values = explainer(explanation_df)
                    values = np.asarray(shap_values.values)
                    if values.ndim == 3:
                        values = values[..., -1]

                if values.ndim == 1:
                    values = values.reshape(-1, 1)

                mean_abs_shap = np.abs(values).mean(axis=0)
                feature_importance = pd.DataFrame(
                    {
                        "feature": explanation_df.columns,
                        "mean_abs_shap": mean_abs_shap,
                    }
                ).sort_values("mean_abs_shap", ascending=False).head(top_n)

                explanation_dict["feature_importance"] = feature_importance.to_dict(orient="records")
                explanation_dict["explainability_methods"]["shap"] = True

                sample_count = min(5, explanation_df.shape[0])
                for i in range(sample_count):
                    row_values = np.asarray(values[i]).tolist()
                    if not isinstance(row_values, list):
                        row_values = [row_values]
                    explanation_dict["sample_explanations"].append(
                        {
                            "row": int(i),
                            "feature_contributions": dict(zip(explanation_df.columns, row_values)),
                        }
                    )
            except Exception as shap_error:
                explanation_dict["shap_error"] = str(shap_error)

        # --- LIME explanation ---
        if LIME_AVAILABLE and explanation_df.shape[0] > 0 and explanation_df.shape[1] > 0:
            try:
                feature_names = explanation_df.columns.tolist()
                explanation_values = np.asarray(explanation_df, dtype=float)

                def predict_fn(batch):
                    batch_array = np.asarray(batch, dtype=float)
                    if hasattr(estimator, "predict_proba"):
                        return estimator.predict_proba(batch_array)
                    preds = estimator.predict(batch_array)
                    return np.asarray(preds).reshape(-1, 1)

                mode = "classification" if hasattr(estimator, "predict_proba") else "regression"
                explainer = LimeTabularExplainer(
                    training_data=explanation_values,
                    feature_names=feature_names,
                    class_names=["class_0", "class_1"] if mode == "classification" else ["value"],
                    mode=mode,
                    random_state=42,
                    discretize_continuous=True,
                )

                for i in range(min(3, explanation_df.shape[0])):
                    instance = explanation_values[i]
                    pred_label = 1 if hasattr(estimator, "predict_proba") else 0
                    if hasattr(estimator, "predict_proba"):
                        try:
                            pred_probs = estimator.predict_proba(instance.reshape(1, -1))
                            pred_label = int(np.argmax(pred_probs[0]))
                        except Exception:
                            pred_label = 1
                    exp = explainer.explain_instance(
                        instance,
                        predict_fn,
                        labels=[pred_label],
                        num_features=min(top_n, len(feature_names)),
                        num_samples=200,
                    )
                    lime_top = [
                        {
                            "feature": feature,
                            "weight": round(float(weight), 4),
                        }
                        for feature, weight in exp.as_list(label=pred_label)
                    ]
                    explanation_dict["lime_explanations"].append(
                        {
                            "row": int(i),
                            "predicted_label": pred_label,
                            "top_features": lime_top,
                        }
                    )

                explanation_dict["explainability_methods"]["lime"] = True
            except Exception as lime_error:
                explanation_dict["lime_error"] = str(lime_error)

        if not explanation_dict["feature_importance"] and not explanation_dict["lime_explanations"]:
            explanation_dict["error"] = "No explainability output could be generated"

    except Exception as e:
        explanation_dict = {
            **explanation_dict,
            "error": str(e),
        }

    return explanation_dict
