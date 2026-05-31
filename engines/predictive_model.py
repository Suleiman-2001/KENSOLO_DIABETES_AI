# engines/predictive_engine.py

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack, csr_matrix


def run_predictive_model(df, tfidf_matrix, column_types):
    """
    Hybrid Predictive Engine:
    - TF-IDF features (text intelligence)
    - Numeric features (clinical/business signals)
    - Classification performance labelling
    """

    results = {}

    numeric_cols = [
        col for col, t in column_types.items()
        if t == "numerical" and col in df.columns
    ]

    if not numeric_cols:
        return results

    # Ensure TF-IDF is valid sparse matrix
    if tfidf_matrix is None:
        return results

    if not hasattr(tfidf_matrix, "shape"):
        return {"error": "Invalid TF-IDF matrix"}

    tfidf_matrix = csr_matrix(tfidf_matrix)

    for target_col in numeric_cols:
        try:
            if target_col not in df.columns:
                continue

            col_data = df[target_col].dropna()
            if len(col_data) < 10:
                results[target_col] = {"error": "Insufficient data"}
                continue

            mean_val = col_data.mean()

            # ----------------------------
            # 1. Create performance labels
            # ----------------------------
            def label(val):
                if val < mean_val * 0.8:
                    return "Needs Support"
                elif val > mean_val * 1.2:
                    return "Excelling"
                else:
                    return "Average"

            df = df.copy()
            df["performance"] = df[target_col].apply(label)

            y = df["performance"]

            # ----------------------------
            # 2. Numeric features (safe handling)
            # ----------------------------
            feature_cols = [
                c for c in numeric_cols
                if c != target_col and c in df.columns
            ]

            if feature_cols:
                X_numeric = df[feature_cols].fillna(0).values
                X_numeric = csr_matrix(X_numeric)
                X = hstack([tfidf_matrix, X_numeric])
            else:
                X = tfidf_matrix

            # ----------------------------
            # 3. Train/test split safety
            # ----------------------------
            if len(df) < 5:
                results[target_col] = {"error": "Too few rows for training"}
                continue

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42
            )

            # ----------------------------
            # 4. Model
            # ----------------------------
            clf = RandomForestClassifier(
                n_estimators=50,
                random_state=42,
                n_jobs=-1
            )

            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)

            # ----------------------------
            # 5. Output structure (IMPROVED)
            # ----------------------------
            results[target_col] = {
                "task": "classification",
                "model": "RandomForestClassifier",
                "sample_predictions": preds.tolist(),
                "classes": clf.classes_.tolist(),
                "train_size": len(X_train),
                "test_size": len(X_test)
            }

        except Exception as e:
            results[target_col] = {
                "error": str(e)
            }

    print("✅ Predictive engine completed")
    return results