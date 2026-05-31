import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# ----------------------------
# Clinical Graph Engine
# ----------------------------
def generate_clinical_graphs(
    df,
    predictions,
    explanations=None,
    adaptive_insights=None,
    folder="outputs/clinical_graphs"
):
    """
    Clinical-grade visualization engine for healthcare / diabetes AI.

    Generates:
    - Biomarker distributions (glucose, BMI, BP, etc.)
    - Risk distribution from predictions
    - Feature importance (clinical explainability)
    - Adaptive risk alerts visualization
    """

    os.makedirs(folder, exist_ok=True)
    saved = []

    # =====================================================
    # 1. Biomarker Distributions (Core Clinical Signals)
    # =====================================================
    biomarkers = ["glucose", "bmi", "bloodpressure", "age", "insulin"]

    for col in df.columns:
        if any(b in col.lower() for b in biomarkers) and pd.api.types.is_numeric_dtype(df[col]):

            plt.figure(figsize=(6, 4))
            df[col].dropna().hist(bins=30)
            plt.title(f"Clinical Distribution: {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")

            path = os.path.join(folder, f"biomarker_{col}.png")
            plt.savefig(path, bbox_inches="tight")
            plt.close()

            saved.append(path)

    # =====================================================
    # 2. Prediction Risk Distribution
    # =====================================================
    for target, info in predictions.items():
        vals = np.array(info.get("sample_predictions", []), dtype=float)

        if vals.size == 0:
            continue

        plt.figure(figsize=(6, 4))
        plt.hist(vals, bins=20)
        plt.title(f"Clinical Risk Distribution: {target}")
        plt.xlabel("Risk Score / Prediction")
        plt.ylabel("Frequency")

        path = os.path.join(folder, f"risk_dist_{target}.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()

        saved.append(path)

    # =====================================================
    # 3. Clinical Feature Importance (Explainability)
    # =====================================================
    if explanations:
        for target, exp in explanations.items():

            ranking = exp.get("feature_ranking", [])
            if not ranking:
                continue

            features = [r[0] for r in ranking[:10]]
            scores = [r[1] for r in ranking[:10]]

            plt.figure(figsize=(7, 4))
            plt.barh(features[::-1], scores[::-1])
            plt.title(f"Clinical Feature Drivers: {target}")
            plt.xlabel("Importance")

            path = os.path.join(folder, f"clinical_features_{target}.png")
            plt.savefig(path, bbox_inches="tight")
            plt.close()

            saved.append(path)

    # =====================================================
    # 4. Adaptive Risk Visualization (Patient Safety View)
    # =====================================================
    if adaptive_insights:
        for target, info in adaptive_insights.items():

            flags = info.get("adaptive_flags", [])
            if not flags:
                continue

            low = sum(1 for f in flags if "Below" in f.get("alert", ""))
            high = sum(1 for f in flags if "Above" in f.get("alert", ""))

            plt.figure(figsize=(5, 4))
            plt.bar(["Low Risk Anomaly", "High Risk Anomaly"], [low, high])
            plt.title(f"Clinical Risk Alerts: {target}")
            plt.ylabel("Count")

            path = os.path.join(folder, f"clinical_risk_{target}.png")
            plt.savefig(path, bbox_inches="tight")
            plt.close()

            saved.append(path)

    # =====================================================
    # Summary
    # =====================================================
    print(f"Clinical graphs generated: {len(saved)} files")
    print(f"Saved in: {folder}")

    return saved