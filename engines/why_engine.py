# engines/why_engine.py
import pandas as pd
import shap
import numpy as np

def explain_predictions(model_pipeline, df, target_column, top_n=5):
    """
    Explains the predictions for a numeric or categorical target using SHAP.
    
    Parameters:
        model_pipeline : trained sklearn Pipeline
        df             : DataFrame with features
        target_column  : str, column name of target
        top_n          : number of top features to show
    
    Returns:
        explanation_dict : dict containing SHAP feature importance & sample explanations
    """
    explanation_dict = {}
    try:
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Create explainer
        explainer = shap.Explainer(model_pipeline.predict, X, feature_names=X.columns)
        shap_values = explainer(X)

        # Global feature importance
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        feature_importance = pd.DataFrame({
            "feature": X.columns,
            "mean_abs_shap": mean_abs_shap
        }).sort_values("mean_abs_shap", ascending=False).head(top_n)

        # Sample explanation for first 5 rows
        sample_expl = []
        for i in range(min(5, X.shape[0])):
            row_dict = dict(zip(X.columns, shap_values[i].values))
            sample_expl.append({"row": i, "feature_contributions": row_dict})

        explanation_dict = {
            "feature_importance": feature_importance.to_dict(orient="records"),
            "sample_explanations": sample_expl
        }

    except Exception as e:
        explanation_dict = {"error": str(e)}

    return explanation_dict
