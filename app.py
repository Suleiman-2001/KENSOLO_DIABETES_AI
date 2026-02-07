import sys, os
sys.path.append(os.path.abspath(os.getcwd()))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import numpy as np
import base64

warnings.filterwarnings("ignore")  # suppress warnings

from core.router import route_to_engines

# ----------------------------
# Background Image with dark overlay
# ----------------------------
def set_background(image_path, darkness=0.6):
    """
    Set background image with dark overlay in Streamlit.
    darkness: float (0 to 1), higher = darker.
    """
    image_path = image_path.replace("\\", "/")  # fix slashes for Windows
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            data = f.read()
        encoded = base64.b64encode(data).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background: linear-gradient(rgba(0,0,0,{darkness}), rgba(0,0,0,{darkness})),
                            url("data:image/jpg;base64,{encoded}") no-repeat center center fixed;
                background-size: cover;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning(f"Background image not found at {image_path}")

# Absolute path to your image
image_path = "images/1687972706509..jpg"
set_background(image_path, darkness=0.9)  # increase darkness for better readability

# ----------------------------
# ----------------------------
# Enhance visibility of analytics content
# ----------------------------
st.markdown(
    """
    <style>
    /* Make headers bold, larger, and white */
    h1, h2, h3, h4, h5, h6, .stHeader {
        font-weight: 900 !important;
        color: #FFFFFF !important;  /* White headers */
        text-shadow: 1px 1px 3px rgba(0,0,0,0.7);  /* subtle shadow for contrast */
    }

    /* JSON and info boxes */
    .stJson, .stText, .stMarkdown {
        font-weight: 700 !important;
        color: #FFFFFF !important;  /* White text */
        background-color: rgba(0,0,0,0.5) !important; /* semi-transparent dark background */
        padding: 5px;
        border-radius: 5px;
    }

    /* Buttons */
    .stButton>button {
        font-weight: 700 !important;
        background-color: #1E90FF !important;  /* Blue buttons */
        color: #FFFFFF !important;  /* White text */
    }

    /* Selectbox, file uploader, checkbox labels */
    div[role="listbox"], .stFileUploader, label, span {
        color: #FFFFFF !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(page_title="KENSOLO AI", layout="wide")
st.title("🤖 KENSOLO — AI Analytics Dashboard")

# ----------------------------
# Industry Selection
# ----------------------------
industry_options = ["None", "Finance", "Healthcare", "Retail", "Manufacturing"]
selected_industry = st.selectbox("Select Industry for Smart Insights (Industry Mode)", industry_options)
industry_value = None if selected_industry == "None" else selected_industry

# ----------------------------
# File Upload
# ----------------------------
uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file:
    # Load CSV/Excel safely
    try:
        if uploaded_file.name.endswith(".csv"):
            try:
                df = pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
        else:
            df = pd.read_excel(uploaded_file)
        st.success(f"✅ Loaded {df.shape[0]} rows × {df.shape[1]} columns")
    except Exception as e:
        st.error(f"Failed to load file: {e}")
        st.stop()

    # Autofix toggle
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

    # Column types detection
    column_types = {col: ("text" if df[col].dtype == "object" else "numerical") for col in df.columns}

    # Run AI Analysis (Manual)
    if st.button("Run AI Analysis"):
        with st.spinner("🚀 Running AI analysis..."):
            output = route_to_engines(df, column_types, autofix=autofix, industry=industry_value)

        # Problem Discovery + Top Outliers
        st.subheader("🛠 Problem Discovery")
        problem_discovery = output.get("problem_discovery") or {}
        st.json(problem_discovery)

        top_outliers = {}
        for col, desc in problem_discovery.items():
            if col in df.columns and np.issubdtype(df[col].dtype, np.number):
                outliers_df = df[col].sort_values(key=lambda x: abs(x - x.mean()), ascending=False).head(10)
                top_outliers[col] = outliers_df.tolist()
        if top_outliers:
            st.markdown("**Top 10 outliers per column:**")
            for col, values in top_outliers.items():
                st.write(f"{col}: {values}")

        # Business Intelligence
        st.subheader("📌 Business Intelligence")
        business_insights = output.get("business_insights") or {}
        st.json(business_insights)

        # Industry Smart Insights
        if industry_value:
            st.subheader(f"💡 Industry Smart Insights ({industry_value})")
            industry_insights = output.get("industry_insights") or {}
            if industry_insights:
                for key, val in industry_insights.items():
                    if "missing" in key.lower():
                        st.warning(f"⚠️ {key}: {val}")
                    elif "outliers" in key.lower() or "anomalies" in key.lower() or "invalid" in key.lower():
                        st.error(f"❌ {key}: {val}")
                    else:
                        st.info(f"ℹ️ {key}: {val}")
            else:
                st.success("No industry-specific issues detected.")

        # Predictions
        st.subheader("📊 Predictions")
        predictions = output.get("predictions") or {}
        st.json(predictions)

        # Why Engine (Top SHAP Features)
        st.subheader("💡 Why Predictions (Top SHAP Features)")
        try:
            from shap import Explainer, values_to_matrix, summary_plot
        except ImportError:
            st.warning("SHAP library is not installed. Skipping feature explanations.")
            Explainer = None

        if Explainer is not None:
            for target, info in predictions.items():
                model_pipe = info.get("best_model_pipeline")
                if not model_pipe:
                    continue

                st.markdown(f"**Target:** {target} — Best Model: {info.get('best_model', 'N/A')}")

                try:
                    X_raw = df.drop(columns=[target])
                    if info.get("task") == "regression":
                        X_raw = X_raw.select_dtypes(include=[np.number])
                        if X_raw.empty:
                            st.info(f"No numeric columns to explain for {target}. Skipping SHAP.")
                            continue

                    if "prep" in model_pipe.named_steps:
                        X_transformed = model_pipe.named_steps["prep"].transform(X_raw)
                    else:
                        X_transformed = X_raw.values

                    background = X_transformed[:min(100, X_transformed.shape[0])]
                    model = model_pipe.named_steps["model"]

                    explainer = Explainer(model.predict, background)
                    shap_values = explainer(X_transformed)

                    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
                    feature_names = X_raw.columns.tolist()
                    top_indices = np.argsort(mean_abs_shap)[-5:][::-1]
                    top_features = [(feature_names[i], mean_abs_shap[i]) for i in top_indices]

                    feat_names, feat_vals = zip(*top_features)
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.barh(feat_names, feat_vals, color="skyblue")
                    ax.invert_yaxis()
                    ax.set_xlabel("Mean |SHAP Value|")
                    ax.set_title(f"Top 5 Features for {target}")
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"Failed to generate SHAP for {target}: {e}")

        # Recommendations
        st.subheader("🎯 Recommendations")
        recommendations = output.get("recommendations") or {}
        st.json(recommendations)

        # Self Critic
        st.subheader("🧪 Self-Critic")
        self_critic = output.get("self_critic") or {}
        st.json(self_critic)

        # Decision Intelligence
        st.subheader("🧠 Decision Intelligence")
        decision_intelligence = output.get("decision_intelligence") or {}
        st.json(decision_intelligence)

        # Graphs
        st.subheader("📈 Graphs")
        graph_folder = output.get("graph_folder") or "outputs/graphs"
        if os.path.exists(graph_folder):
            graphs = [f for f in os.listdir(graph_folder) if f.endswith(".png")]
            if graphs:
                for g in sorted(graphs):
                    st.image(os.path.join(graph_folder, g), caption=g, use_container_width=True)
            else:
                st.warning("No graphs found in graph folder.")
        else:
            st.warning("Graph folder does not exist.")

        # Downloads
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

        # Adaptive / Self-Learning Insights
        st.subheader("💡 Adaptive / Self-Learning Insights")
        adaptive_insights = output.get("self_critic") or {}
        if adaptive_insights:
            st.json(adaptive_insights)
            st.download_button(
                "Download Adaptive Insights JSON",
                data=pd.Series(adaptive_insights).to_json(),
                file_name="adaptive_insights.json"
            )
        else:
            st.info("No adaptive insights generated.")

# ----------------------------
# AI Analytics Autopilot
# ----------------------------
st.subheader("🤖 AI Analytics Autopilot")
autopilot_mode = st.checkbox("Enable AI Analytics Autopilot (Run full analysis automatically)")

if uploaded_file and autopilot_mode:
    if st.button("Run Autopilot"):
        with st.spinner("🚀 Running AI Analytics Autopilot..."):
            auto_column_types = {}
            for col in df.columns:
                if np.issubdtype(df[col].dtype, np.number):
                    auto_column_types[col] = "numerical"
                elif np.issubdtype(df[col].dtype, np.datetime64):
                    auto_column_types[col] = "datetime"
                else:
                    auto_column_types[col] = "text"

            output = route_to_engines(
                df=df,
                column_types=auto_column_types,
                autofix=True,
                industry=industry_value
            )

        st.success("✅ AI Analytics Autopilot Complete!")

        # Display outputs
        sections = [
            ("🛠 Problem Discovery", "problem_discovery"),
            ("📌 Business Intelligence", "business_insights"),
            (f"💡 Industry Smart Insights ({industry_value})", "industry_insights"),
            ("📊 Predictions", "predictions"),
            ("🎯 Recommendations", "recommendations"),
            ("🧪 Self-Critic", "self_critic"),
            ("🧠 Decision Intelligence", "decision_intelligence")
        ]

        for title, key in sections:
            st.subheader(title)
            st.json(output.get(key) or {})

        # Graphs
        st.subheader("📈 Graphs")
        graph_folder = output.get("graph_folder") or "outputs/graphs"
        if os.path.exists(graph_folder):
            graphs = [f for f in os.listdir(graph_folder) if f.endswith(".png")]
            if graphs:
                for g in sorted(graphs):
                    st.image(os.path.join(graph_folder, g), caption=g, use_container_width=True)
            else:
                st.warning("No graphs found in graph folder.")
        else:
            st.warning("Graph folder does not exist.")

        # Downloads
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

        # Adaptive Insights
        st.subheader("💡 Adaptive / Self-Learning Insights")
        adaptive_insights = output.get("self_critic") or {}
        if adaptive_insights:
            st.json(adaptive_insights)
            st.download_button(
                "Download Adaptive Insights JSON",
                data=pd.Series(adaptive_insights).to_json(),
                file_name="adaptive_insights.json"
            )
        else:
            st.info("No adaptive insights generated.")
