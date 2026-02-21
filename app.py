import sys, os
sys.path.append(os.path.abspath(os.getcwd()))

import streamlit as st
# ----------------------------
# SAFE DEFAULTS (to avoid NameError)
# ----------------------------
df = None
column_types = None
autofix = False
# ----------------------------
# MUST BE FIRST STREAMLIT COMMAND
# ----------------------------
st.set_page_config(page_title="KENSOLO AI", layout="wide")
st.title("🤖 KENSOLO — AI Analytics Dashboard")

import pandas as pd
import matplotlib.pyplot as plt
import warnings
import numpy as np
import base64

warnings.filterwarnings("ignore")  # suppress warnings

from core.router import route_to_engines

# ----------------------------
# Session State for Outputs
# ----------------------------
if "output" not in st.session_state:
    st.session_state.output = None

if "autopilot_ran" not in st.session_state:
    st.session_state.autopilot_ran = False

# ----------------------------
# Dark Blue Background (No Image)
# ----------------------------
def set_background(bg_color="#0B3D91"):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {bg_color};
        }}
        h1, h2, h3, h4, h5, h6, .stHeader {{
            font-weight: 900 !important;
            color: #FFFFFF !important;
            text-shadow: 1px 1px 3px rgba(0,0,0,0.7);
        }}
        .stText, .stInfo, .stMarkdown, .stJson {{
            font-weight: 700 !important;
            color: #000000 !important;
            background-color: rgba(255,255,255,0.2) !important;
            padding: 8px;
            border-radius: 6px;
            overflow-x: auto;
        }}
        .stButton>button {{
            font-weight: 700 !important;
            background-color: #1E90FF !important;
            color: #FFFFFF !important;
        }}
        div[role="listbox"], .stFileUploader, label, span {{
            color: #FFFFFF !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background()

# ----------------------------
# Display AI Answers
# ----------------------------
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
# Industry Selection
# ----------------------------
industry_options = ["None", "Finance", "Healthcare", "Retail", "Manufacturing"]
selected_industry = st.selectbox("Select Industry for Smart Insights (Industry Mode)", industry_options)
industry_value = None if selected_industry == "None" else selected_industry

# ----------------------------
# File Upload
# ----------------------------
uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
df = None

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

    # Reset autopilot flag when a new file is uploaded
    st.session_state.autopilot_ran = False

    # ----------------------------
    # Autofix
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

    column_types = {col: ("text" if df[col].dtype == "object" else "numerical") for col in df.columns}

    # ----------------------------
    # Power BI Export
    # ----------------------------
    def export_for_powerbi(df, output, industry_value):
        powerbi_folder = "outputs/powerbi"
        os.makedirs(powerbi_folder, exist_ok=True)

        df.to_csv(os.path.join(powerbi_folder, "df_for_powerbi.csv"), index=False)

        predictions = output.get("predictions") or {}
        if predictions:
            all_preds = {}
            for target, info in predictions.items():
                sample_preds = info.get("sample_predictions")
                if sample_preds:
                    all_preds[target] = pd.DataFrame(sample_preds)
            if all_preds:
                for target, df_preds in all_preds.items():
                    df_preds.to_csv(os.path.join(powerbi_folder, f"predictions_{target}.csv"), index=False)

        recommendations = output.get("recommendations") or {}
        if recommendations:
            for key, rec_list in recommendations.items():
                if rec_list:
                    pd.DataFrame(rec_list).to_csv(os.path.join(powerbi_folder, f"recommendations_{key}.csv"), index=False)

        adaptive_insights = output.get("adaptive_insights") or {}
        if adaptive_insights:
            pd.DataFrame.from_dict(adaptive_insights, orient="index").to_csv(
                os.path.join(powerbi_folder, "adaptive_insights.csv")
            )

        st.success(f"✅ Power BI export completed! CSVs saved in '{powerbi_folder}'")


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
                answer = talk_to_data_ai(df, query=user_question)
                display_ai_answer(answer)
    else:
        display_ai_answer("Talk-to-Your-Data AI module not installed. Skipping.")

# ----------------------------
# Run AI Analysis (Manual)
# ----------------------------
if df is not None and st.button("Run AI Analysis"):
    with st.spinner("🚀 Running AI analysis..."):
        # Fallback in case column_types is not defined
        if column_types is None:
            column_types = {col: ("text" if df[col].dtype == "object" else "numerical") for col in df.columns}
        st.session_state.output = route_to_engines(df, column_types, autofix=autofix, industry=industry_value)
        output = st.session_state.output

        # ----------------------------
        # SAVE FILES FOR DOWNLOAD
        # ----------------------------
        import json
        import os
        import pandas as pd
        os.makedirs("outputs", exist_ok=True)

        # Predictions JSON
        with open("outputs/predictions.json", "w") as f:
            json.dump(output.get("predictions", {}), f, indent=4)

        # Recommendations JSON
        with open("outputs/recommendations.json", "w") as f:
            json.dump(output.get("recommendations", {}), f, indent=4)

        # Predictions CSV
        pred_rows = []
        for target, items in output.get("predictions", {}).items():
            for item in items:
                row = item.copy() if isinstance(item, dict) else {"value": item}
                row["target"] = target
                pred_rows.append(row)
        pd.DataFrame(pred_rows).to_csv("outputs/predictions.csv", index=False)

        # Recommendations CSV
        rec_rows = []
        for target, rec_list in output.get("recommendations", {}).items():
            for rec in rec_list:
                row = rec.copy()
                row["target"] = target
                rec_rows.append(row)
        pd.DataFrame(rec_rows).to_csv("outputs/recommendations.csv", index=False)

        # Simple PDF Report
        try:
            from fpdf import FPDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "KENSOLO AI Report", ln=True, align="C")
            pdf.output("outputs/report.pdf")
        except Exception as e:
            st.warning(f"PDF report generation failed: {e}")

    st.success("✅ AI Analysis Complete! Files saved in outputs/")

    # ----------------------------
    # Display Outputs
    # ----------------------------
    output = st.session_state.output
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

    # Adaptive / Self-Learning Insights
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

# Run autopilot automatically only once per file upload
if uploaded_file and autopilot_mode and not st.session_state.autopilot_ran:
    st.session_state.autopilot_ran = True
    with st.spinner("🚀 Running AI Analytics Autopilot..."):
        auto_column_types = {}
        for col in df.columns:
            if np.issubdtype(df[col].dtype, np.number):
                auto_column_types[col] = "numerical"
            elif np.issubdtype(df[col].dtype, np.datetime64):
                auto_column_types[col] = "datetime"
            else:
                auto_column_types[col] = "text"

        st.session_state.output = route_to_engines(
            df=df,
            column_types=auto_column_types,
            autofix=True,
            industry=industry_value
        )

    st.success("✅ AI Analytics Autopilot Complete!")

# Display output if exists
if st.session_state.output:
    output = st.session_state.output

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

    # Adaptive / Self-Learning Insights
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
