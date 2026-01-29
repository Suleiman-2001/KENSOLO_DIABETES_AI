import sys, os
sys.path.append(os.path.abspath(os.getcwd()))

import streamlit as st
import pandas as pd
from core.router import route_to_engines

st.set_page_config(page_title="🤖 KENSOLO AI", layout="wide")
st.title("🤖 KENSOLO — AI Analytics Dashboard")

# Folder paths
GRAPH_FOLDER = "outputs/graphs"
PREDICTIONS_FILE = "outputs/predictions.json"
RECOMMENDATIONS_FILE = "outputs/recommendations.json"

uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    st.success(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")

    # Detect column types
    column_types = {col: "text" if df[col].dtype == "object" else "numerical" for col in df.columns}

    if st.button("Run AI Analysis"):
        with st.spinner("Running AI analysis..."):
            output = route_to_engines(df, column_types)

        st.subheader("🛠 Problem Discovery")
        st.json(output["problem_discovery"])

        st.subheader("🧠 Self-Critic")
        st.json(output["self_critic"])

        # Display predictions & recommendations per target
        st.subheader("📊 Predictions & Recommendations")
        for target, pred_info in output["predictions"].items():
            st.markdown(f"### {target.capitalize()}")

            # Show graph
            graph_path = os.path.join(GRAPH_FOLDER, f"{target}.png")
            if os.path.exists(graph_path):
                st.image(graph_path, caption=f"{target} distribution", use_column_width=True)
            else:
                st.info("Graph not available.")

            # Display sample predictions table
            if "sample_predictions" in pred_info:
                sample_df = pd.DataFrame({
                    "Prediction": pred_info["sample_predictions"]
                })
                # Add category & recommendation if available
                rec_list = output["recommendations"].get(target, [])
                if rec_list:
                    sample_df["Category"] = [r["category"] for r in rec_list]
                    sample_df["Recommendation"] = [r["recommendation"] for r in rec_list]
                st.dataframe(sample_df)

        # Download buttons for predictions & recommendations
        st.subheader("💾 Download Outputs")
        if os.path.exists(PREDICTIONS_FILE):
            with open(PREDICTIONS_FILE, "rb") as f:
                st.download_button("Download Predictions JSON", f, file_name="predictions.json")
        if os.path.exists(RECOMMENDATIONS_FILE):
            with open(RECOMMENDATIONS_FILE, "rb") as f:
                st.download_button("Download Recommendations JSON", f, file_name="recommendations.json")
