# engines/predictive_engine.py
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def run_predictive_model(df, tfidf_features, targets):
    """
    Predictive engine that automatically decides regression vs classification.
    """
    results = {}
    if not targets:
        return results

    for target in targets:
        try:
            y = df[target]

            X = df.drop(columns=[target])
            X = pd.get_dummies(X, drop_first=True)
            X = X.select_dtypes(include=["number"])
            if X.empty:
                continue

            # Decide model type
            if y.dtype == "O" or y.nunique() <= 10:
                # Classification
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                clf = RandomForestClassifier(n_estimators=100, random_state=42)
                clf.fit(X_train, y_train)
                preds = clf.predict(X_test)
                results[target] = {
                    "model": "RandomForestClassifier",
                    "sample_predictions": preds[:5].tolist(),
                    "accuracy": float(accuracy_score(y_test, preds))
                }
            else:
                # Regression
                model = LinearRegression()
                model.fit(X, y)
                preds = model.predict(X)
                results[target] = {
                    "model": "LinearRegression",
                    "sample_predictions": preds[:5].tolist(),
                    "mean_prediction": float(preds.mean())
                }
        except Exception as e:
            results[target] = {"error": str(e)}

    return results
