import pandas as pd


def _as_dataframe(current_features):
    if isinstance(current_features, pd.DataFrame):
        return current_features.copy()
    if isinstance(current_features, pd.Series):
        return current_features.to_frame().T
    if isinstance(current_features, dict):
        return pd.DataFrame([current_features])
    raise TypeError("current_features must be a DataFrame, Series, or dict")


def simulated_progression(current_features, years=5):
    """
    Create a simple longitudinal simulation by nudging clinical variables over time.
    This is a heuristic scenario generator, not a causal clinical model.
    """
    df = _as_dataframe(current_features)
    years = max(0, int(years))

    annual_deltas = {
        "age": 1.0,
        "glucose": 2.0,
        "hba1c": 0.08,
        "bmi": 0.15,
        "insulin": -0.5,
        "pressure": 0.4,
    }

    for col in df.columns:
        col_lower = str(col).lower()
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        for key, delta in annual_deltas.items():
            if key in col_lower:
                df[col] = pd.to_numeric(df[col], errors="coerce") + (delta * years)
                break

    return df


def predict_future_risk(model, current_features, years=5):
    """
    Predict risk probability from simulated future features.

    Example usage:
        predict_future_risk(model, current_features, years=5)
    """
    progressed = simulated_progression(current_features, years=years)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(progressed)

        if getattr(proba, "ndim", 1) == 2 and proba.shape[1] > 1:
            risk_values = proba[:, 1]
        else:
            risk_values = proba.ravel()

        return {
            "years": years,
            "risk_probability": [float(x) for x in risk_values],
            "progressed_features": progressed.to_dict(orient="records"),
        }

    preds = model.predict(progressed)
    return {
        "years": years,
        "risk_prediction": [float(x) for x in preds],
        "progressed_features": progressed.to_dict(orient="records"),
    }
