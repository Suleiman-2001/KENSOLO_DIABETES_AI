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
# Dark Blue Background (No Image)
# ----------------------------
def set_background(bg_color="#0B3D91"):  # dark blue
    """
    Set a solid dark blue background in Streamlit.
    """
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {bg_color};
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Apply dark blue background
set_background()

# ----------------------------
# Enhance visibility of analytics content for dark blue background
# ----------------------------
st.markdown(
    """
    <style>
    h1, h2, h3, h4, h5, h6, .stHeader {
        font-weight: 900 !important;
        color: #FFFFFF !important;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.7);
    }
    .stJson, .stText, .stMarkdown, .stInfo {
        font-weight: 700 !important;
        color: #000000 !important;  /* black text for better visibility */
        background-color: rgba(255,255,255,0.8) !important;  /* light background */
        padding: 8px;
        border-radius: 6px;
        overflow-x: auto;
    }
    .stButton>button {
        font-weight: 700 !important;
        background-color: #1E90FF !important;
        color: #FFFFFF !important;
    }
    div[role="listbox"], .stFileUploader, label, span {
        color: #FFFFFF !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# ----------------------------
# Custom style for Talk-to-Your-Data AI response
# ----------------------------
st.markdown(
    """
    <style>
    /* Force AI response text to be black */
    .stInfo, .stText, .stMarkdown {
        color: #000000 !important;  /* black */
        background-color: rgba(255, 255, 255, 0.2) !important;  /* optional light background for contrast */
        padding: 5px;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
def display_ai_answer(answer):
    st.markdown(
        f"""
        <div style="
            background-color: rgba(255,255,255,0.9);
            color: black;
            padding: 12px;
            border-radius: 8px;
            font-weight: 600;
        ">
            {answer}
        </div>
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

    column_types = {col: ("text" if df[col].dtype == "object" else "numerical") for col in df.columns}

    # ----------------------------
    # Power BI Export Function
    # ----------------------------
    def export_for_powerbi(df, output, industry_value):
        os.makedirs("outputs/powerbi", exist_ok=True)
        df.to_csv("outputs/powerbi/df_for_powerbi.csv", index=False)

        predictions = output.get("predictions") or {}
        if predictions:
            all_preds = {}
            for target, info in predictions.items():
                all_preds[target] = info.get("sample_predictions", [])
            pd.DataFrame(all_preds).to_csv("outputs/powerbi/predictions_for_powerbi.csv", index=False)

        recommendations = output.get("recommendations") or {}
        if recommendations:
            pd.DataFrame(recommendations).to_csv("outputs/powerbi/recommendations_for_powerbi.csv", index=False)

        st.success("✅ Power BI export completed in outputs/powerbi/")

  # ----------------------------
# Talk-to-Your-Data AI
# ----------------------------
try:
    from engines.talk_to_data import talk_to_data_ai
    TALK_ENABLED = True
except ModuleNotFoundError:
    TALK_ENABLED = False

if TALK_ENABLED:
    st.subheader("💬 Talk to Your Data AI")
    user_question = st.text_input("Ask a question about your data (e.g., 'Top 5 Amount outliers')")
    if st.button("Ask AI") and user_question:
        with st.spinner("🤖 Generating answer..."):
            answer = talk_to_data_ai(df, query=user_question)  # ✅ updated 'question' -> 'query'
            st.info(answer)
else:
    display_ai_answer("Talk-to-Your-Data AI module not installed. Skipping.")

    # ----------------------------
    # Run AI Analysis (Manual)
    # ----------------------------
    if st.button("Run AI Analysis"):
        with st.spinner("🚀 Running AI analysis..."):
            output = route_to_engines(df, column_types, autofix=autofix, industry=industry_value)

        # ----------------------------
        # Problem Discovery + Top Outliers Table
        # ----------------------------
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
                st.dataframe(pd.DataFrame({col: values}))

        # ----------------------------
        # Business Intelligence
        # ----------------------------
        st.subheader("📌 Business Intelligence")
        business_insights = output.get("business_insights") or {}
        st.json(business_insights)

        # ----------------------------
        # Industry Smart Insights
        # ----------------------------
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

        # ----------------------------
        # Predictions + Filtering
        # ----------------------------
        st.subheader("📊 Predictions")
        predictions = output.get("predictions") or {}
        st.json(predictions)

        st.subheader("📊 Filtered Predictions")
        recommendations = output.get("recommendations") or {}
        category_filter = st.selectbox("Filter predictions by category", ["All", "High", "Medium", "Low"])
        filtered_recs = {}
        for key, rec_list in recommendations.items():
            if category_filter != "All":
                filtered_recs[key] = [r for r in rec_list if r.get("category") == category_filter]
            else:
                filtered_recs[key] = rec_list
        st.json(filtered_recs)

        # ----------------------------
        # Why Engine (Top SHAP Features)
        # ----------------------------
        st.subheader("💡 Why Predictions (Top SHAP Features)")
        try:
            from shap import Explainer
        except ImportError:
            st.warning("SHAP library is not installed. Skipping feature explanations.")
            Explainer = None

        if Explainer:
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

                    X_transformed = model_pipe.named_steps.get("prep", None)
                    if X_transformed:
                        X_transformed = X_transformed.transform(X_raw)
                    else:
                        X_transformed = X_raw.values

                    model = model_pipe.named_steps["model"]
                    background = X_transformed[:min(100, X_transformed.shape[0])]

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

        # ----------------------------
        # Recommendations
        # ----------------------------
        st.subheader("🎯 Recommendations")
        st.json(recommendations)

        # ----------------------------
        # Self Critic
        # ----------------------------
        st.subheader("🧪 Self-Critic")
        self_critic = output.get("self_critic") or {}
        st.json(self_critic)

        # ----------------------------
        # Decision Intelligence
        # ----------------------------
        st.subheader("🧠 Decision Intelligence")
        decision_intelligence = output.get("decision_intelligence") or {}
        st.json(decision_intelligence)

        # ----------------------------
        # Graphs
        # ----------------------------
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

        # ----------------------------
        # Power BI Export
        # ----------------------------
        st.button("Export for Power BI", on_click=export_for_powerbi, args=(df, output, industry_value))

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

        # ----------------------------
        # Adaptive / Self-Learning Insights
        # ----------------------------
        st.subheader("💡 Adaptive / Self-Learning Insights")
        adaptive_insights = output.get("adaptive_insights") or {}
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

        # ----------------------------
        # Display outputs (Autopilot)
        # ----------------------------
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
        adaptive_insights = output.get("adaptive_insights") or {}
        if adaptive_insights:
            st.json(adaptive_insights)
            st.download_button(
                "Download Adaptive Insights JSON",
                data=pd.Series(adaptive_insights).to_json(),
                file_name="adaptive_insights.json"
            )
        else:
            st.info("No adaptive insights generated.")
