from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd


def _build_preprocessor(X):
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if np.issubdtype(X[c].dtype, np.number)]

    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ]
    )


def run_ml(X, y):
    """
    Simple ML engine (robust version)
    Returns structured output for pipeline integration.
    """

    print("🤖 Running ML engine...")

    # Basic validation
    if len(X) < 10:
        return {"error": "Not enough data to train model"}

    try:
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Preprocessing
        preprocessor = _build_preprocessor(X_train)

        # Model pipeline
        model = Pipeline([
            ("prep", preprocessor),
            ("model", LogisticRegression(max_iter=1000))
        ])

        # Train
        model.fit(X_train, y_train)

        # Predict
        preds = model.predict(X_test)

        # Accuracy
        acc = accuracy_score(y_test, preds)

        # Sample predictions
        sample_preds = model.predict(X.head(5))

        print(f"📊 Model accuracy: {acc:.2f}")

        return {
            "task": "classification",
            "model_type": "LogisticRegression",
            "accuracy": round(float(acc), 4),
            "sample_predictions": [str(x) for x in sample_preds],
            "status": "success"
        }

    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }