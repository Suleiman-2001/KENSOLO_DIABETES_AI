# engines/self_visualizer.py

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

OUTPUT_DIR = "outputs/visuals"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_dataset_overview(df: pd.DataFrame):
    """
    Visualizes dataset structure + missing values
    """
    fig_path = os.path.join(OUTPUT_DIR, "dataset_overview.png")

    plt.figure(figsize=(10, 5))

    missing = df.isnull().mean().sort_values(ascending=False)

    plt.bar(missing.index[:15], missing.values[:15])
    plt.title("Top Missing Value Rates")
    plt.xticks(rotation=45)
    plt.ylabel("Missing Rate")

    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    return fig_path


def plot_model_performance(predictions: dict):
    """
    Visualizes model performance (accuracy / R²)
    """
    fig_path = os.path.join(OUTPUT_DIR, "model_performance.png")

    targets = []
    scores = []

    for target, info in predictions.items():
        if "error" in info:
            continue

        targets.append(target)

        if info.get("task") == "classification":
            scores.append(info.get("accuracy", 0))
        else:
            scores.append(info.get("r2_score", 0))

    plt.figure(figsize=(8, 5))
    plt.bar(targets, scores)
    plt.title("Model Performance by Target")
    plt.ylabel("Score (Accuracy / R²)")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    return fig_path


def plot_risk_profile(self_critic_output: dict):
    """
    Visualizes system risk level
    """
    fig_path = os.path.join(OUTPUT_DIR, "risk_profile.png")

    risk_score = self_critic_output.get("risk_score", 0)
    trust_level = self_critic_output.get("trust_level", "unknown")

    labels = ["Risk Score", "Max Score"]
    values = [risk_score, 10]

    plt.figure(figsize=(6, 5))
    plt.bar(labels, values)
    plt.title(f"System Risk Profile ({trust_level.upper()})")
    plt.ylabel("Risk Level")

    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    return fig_path


def plot_predictions_distribution(predictions: dict):
    """
    Shows distribution of sample predictions
    """
    fig_paths = []

    for target, info in predictions.items():
        if "sample_predictions" not in info:
            continue

        preds = info["sample_predictions"]

        if not preds:
            continue

        fig_path = os.path.join(OUTPUT_DIR, f"{target}_distribution.png")

        plt.figure(figsize=(6, 4))
        plt.hist(preds, bins=10, alpha=0.7)
        plt.title(f"Prediction Distribution: {target}")
        plt.xlabel("Prediction Value")
        plt.ylabel("Frequency")

        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()

        fig_paths.append(fig_path)

    return fig_paths


def run_self_visualization(df, predictions, self_critic_output):
    """
    Master function: runs all system visualizations
    """

    outputs = {
        "dataset_overview": plot_dataset_overview(df),
        "model_performance": plot_model_performance(predictions),
        "risk_profile": plot_risk_profile(self_critic_output),
        "prediction_distributions": plot_predictions_distribution(predictions)
    }

    return outputs