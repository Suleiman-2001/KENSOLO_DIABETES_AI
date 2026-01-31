# app.py
import sys, os
sys.path.append(os.path.abspath(os.getcwd()))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import numpy as np

warnings.filterwarnings("ignore")  # suppress warnings

from core.router import route_to_engines

st.set_page_config(page_title="KENSOLO AI", layout="wide")
st.title("🤖 KENSOLO — AI Analytics Dashboard")

# ----------------------------
# File Upload
# ----------------------------
uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file:
    # ----------------------------
    # Load CSV/Excel safely
    # ----------------------------
    try:
        if uploaded_file.name.endswith(".csv"):
            try:
                df = pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
        else:
            df = pd.read_excel(uploaded_file)
        st.success(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")
    except Exception as e:
        st.error(f"Failed to load file: {e}")
        st.stop()

    # ----------------------------
    # Autofix toggle
    # ----------------------------
    autofix = st.checkbox("Enable Autofix Mode (auto-fill missing / remove constant columns)")

    if autofix:
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].fillna("")
            else:
                df[col] = df[col].fillna(df[col].median())
        constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
        if constant_cols:
            df = df.drop(columns=constant_cols)
            st.info(f"Removed constant columns: {constant_cols}")
        st.success("Autofix applied successfully!")

    # ----------------------------
    # Column types detection
    # ----------------------------
    column_types = {col: ("text" if df[col].dtype == "object" else "numerical") for col in df.columns}

    # ----------------------------
    # Run AI Analysis
    # ----------------------------
    if st.button("Run AI Analysis"):
        with st.spinner("Running AI analysis..."):
            output = route_to_engines(df, column_types, autofix=autofix)

        # ----------------------------
        # Problem Discovery
        # ----------------------------
        st.subheader("🛠 Problem Discovery")
        st.json(output.get("problem_discovery", {}))

        # ----------------------------
        # Business Intelligence
        # ----------------------------
        st.subheader("📌 Business Intelligence")
        st.json(output.get("business_insights", {}))

        # ----------------------------
        # Predictions
        # ----------------------------
        st.subheader("📊 Predictions")
        predictions = output.get("predictions", {})
        st.json(predictions)

        # ----------------------------
        # Why Engine (SHAP Explanations)
        # ----------------------------
        st.subheader("💡 Why Predictions (Feature Importance)")

        try:
            from shap import Explainer, summary_plot
        except ImportError:
            st.warning("SHAP library is not installed. Skipping feature explanations.")
            Explainer = None

        if Explainer is not None:
            for target, info in predictions.items():
                if "best_model_pipeline" not in info:
                    continue

                st.markdown(f"**Target:** {target} — Best Model: {info['best_model']}")
                try:
                    model_pipe = info["best_model_pipeline"]
                    X_raw = df.drop(columns=[target])

                    # ----------------------------
                    # Select relevant features
                    # ----------------------------
                    if info["task"] == "regression":
                        X_raw = X_raw.select_dtypes(include=[np.number])
                        if X_raw.empty:
                            st.info(f"No numeric columns to explain for {target}. Skipping SHAP.")
                            continue

                    # Transform using pipeline preprocessor if available
                    if "prep" in model_pipe.named_steps:
                        X_transformed = model_pipe.named_steps["prep"].transform(X_raw)
                    else:
                        X_transformed = X_raw.values

                    # Use small background sample to avoid errors
                    background = X_transformed[:min(100, X_transformed.shape[0])]
                    model = model_pipe.named_steps["model"]

                    explainer = Explainer(model.predict, background)
                    shap_values = explainer(X_transformed)

                    # Determine feature names
                    feature_names = X_raw.columns.tolist() if hasattr(X_raw, 'columns') else None

                    # Plot SHAP summary
                    fig, ax = plt.subplots(figsize=(10, 5))
                    summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False)
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"Failed to generate SHAP for {target}: {e}")

        # ----------------------------
        # Recommendations
        # ----------------------------
        st.subheader("🎯 Recommendations")
        st.json(output.get("recommendations", {}))

        # ----------------------------
        # Self Critic
        # ----------------------------
        st.subheader("🧪 Self-Critic")
        st.json(output.get("self_critic", {}))

        # ----------------------------
        # Graphs
        # ----------------------------
        st.subheader("📈 Graphs")
        graph_folder = output.get("graph_folder", "outputs/graphs")

        if os.path.exists(graph_folder):
            graphs = [f for f in os.listdir(graph_folder) if f.endswith(".png")]
            if graphs:
                for g in sorted(graphs):
                    st.image(os.path.join(graph_folder, g), caption=g, use_container_width=True)
            else:
                st.warning("No graphs found in graph folder.")
        else:
            st.warning("Graph folder does not exist.")

        # ----------------------------
        # Downloads
        # ----------------------------
        st.subheader("💾 Download Outputs")

        downloadable_files = [
            "outputs/predictions.json",
            "outputs/recommendations.json",
            "outputs/predictions.csv",
            "outputs/recommendations.csv",
            "outputs/report.pdf",
        ]

        for file_name in downloadable_files:
            if os.path.exists(file_name):
                with open(file_name, "rb") as f:
                    st.download_button(
                        f"Download {os.path.basename(file_name)}",
                        f,
                        file_name=os.path.basename(file_name)
                    )
            else:
                st.info(f"Not generated yet: {file_name}")
