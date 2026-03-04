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
# Function to save predictions, recommendations, and report
# ----------------------------
def save_outputs(output):
    import os
    import json
    import pandas as pd
    from fpdf import FPDF

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
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "KENSOLO AI Report", ln=True, align="C")
        pdf.output("outputs/report.pdf")
    except Exception as e:
        st.warning(f"PDF report generation failed: {e}")
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
import hashlib
import uuid

warnings.filterwarnings("ignore")  # suppress warnings

from core.router import route_to_engines
from engines.autofix_engine import apply_autofix

# ----------------------------
# Session State for Outputs
# ----------------------------
if "output" not in st.session_state:
    st.session_state.output = None

if "autopilot_ran" not in st.session_state:
    st.session_state.autopilot_ran = False

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

def set_background():
    st.markdown("""
    <style>
    /* =========================
       MAIN SOFT SKY-CLOUD BACKGROUND
    ========================== */
    .stApp {
        background: linear-gradient(
            to bottom right, 
            rgba(245, 245, 245, 0.95),  /* Soft Cloud / Sky color 95% */
            rgba(15, 32, 39, 0.05)       /* Tiny dark overlay 5% */
        );
        background-attachment: fixed;
    }

    /* Remove Streamlit white block background */
    .block-container {
        background: transparent !important;
    }

    /* Section containers (cards) */
    div[data-testid="stVerticalBlock"] > div {
        background-color: rgba(255,255,255,0.92) !important; /* slightly transparent white */
        padding: 20px !important;
        border-radius: 15px !important;
        margin-bottom: 20px !important;
        color: #0F2027 !important; /* dark text */
        box-shadow: 0 8px 20px rgba(0,0,0,0.08); /* subtle shadow */
    }

    /* JSON Viewer */
    div[data-testid="stJson"] {
        background-color: rgba(255,255,255,0.95) !important;
        border-radius: 12px;
        padding: 12px;
        color: #0F2027 !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    }

    /* File uploader */
    section[data-testid="stFileUploader"] {
        background-color: rgba(255,255,255,0.95) !important;
        padding: 25px;
        border-radius: 15px;
        border: 2px dashed rgba(15, 32, 39, 0.2);
        box-shadow: 0 6px 15px rgba(0,0,0,0.06);
    }

    /* File uploader label */
    section[data-testid="stFileUploader"] label {
        font-size: 20px !important;  /* Larger font */
        font-weight: 700 !important; /* Bold */
        color: #0F2027 !important;
    }

    /* Drag & Drop placeholder */
    section[data-testid="stFileUploader"] div[data-testid="stForm"] span {
        font-size: 18px !important;  /* Slightly larger text */
        color: #0F2027 !important;
        font-weight: 500;
    }

    /* Selectbox label */
    label[for^="Select"] {
        font-size: 20px !important; 
        font-weight: 700 !important;
        color: #0F2027 !important;
    }

    /* Selectbox options container */
    div[data-baseweb="select"] > div {
        background-color: rgba(255,255,255,0.95) !important;
        color: #0F2027 !important;
        font-size: 16px !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(to right, #00C6FF, #0072FF);
        color: white !important;
        font-weight: 700;
        border-radius: 10px;
        border: none;
        font-size: 16px !important;
        padding: 8px 20px !important;
    }

    /* Headings */
    h1, h2, h3, h4 {
        color: #0F2027 !important;
    }

    </style>
    """, unsafe_allow_html=True)

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
# KPI Cards Dashboard
# ----------------------------
def display_kpi_cards(df, output):
    """Display key KPI metrics in interactive cards"""
    if output is None:
        return
    
    st.markdown("## Key Performance Indicators (KPIs)")
    
    # Create tabs for different metric categories
    tab1, tab2, tab3, tab4 = st.tabs(["Dataset Overview", "Data Quality", "Statistics", "Predictions"])
    
    with tab1:
        # Dataset Overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Rows",
                value=f"{len(df):,}",
                delta=None,
                help="Number of records in dataset"
            )
        
        with col2:
            st.metric(
                label="Total Columns",
                value=df.shape[1],
                delta=None,
                help="Number of fields/features"
            )
        
        with col3:
            size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            st.metric(
                label="Dataset Size",
                value=f"{size_mb:.2f} MB",
                delta=None,
                help="Total memory usage"
            )
        
        with col4:
            numeric_cols = df.select_dtypes(include=[np.number]).shape[1]
            st.metric(
                label="Numeric Columns",
                value=numeric_cols,
                delta=None,
                help="Columns with numeric data"
            )
        
        # Overview from output
        overview = output.get("overview_summary", {})
        if overview:
            st.divider()
            st.write("### Additional Overview")
            col_overview = st.columns(min(3, len(overview)))
            for idx, (key, val) in enumerate(list(overview.items())[:3]):
                with col_overview[idx % len(col_overview)]:
                    if isinstance(val, (int, float)):
                        st.metric(label=str(key).replace("_", " ").title(), value=f"{val:.2f}" if isinstance(val, float) else val)
                    else:
                        st.write(f"**{str(key).replace('_', ' ').title()}**: {str(val)[:50]}")
    
    with tab2:
        # Data Quality
        quality = output.get("quality_summary", {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Calculate overall completeness
            missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            completeness = 100 - missing_pct
            st.metric(
                label="Data Completeness",
                value=f"{completeness:.1f}%",
                delta=f"{missing_pct:.1f}% missing",
                help="Percentage of non-null values"
            )
        
        with col2:
            if isinstance(quality, dict) and "overall_quality_score" in quality:
                quality_score = quality["overall_quality_score"]
                st.metric(
                    label="Quality Score",
                    value=f"{quality_score:.1f}%",
                    delta=None,
                    help="Overall data quality rating"
                )
            else:
                st.metric(label="Quality Score", value="N/A", help="Data quality metrics not available")
        
        with col3:
            unique_ratio = df.nunique().sum() / (len(df) * df.shape[1]) * 100
            st.metric(
                label="Uniqueness Index",
                value=f"{unique_ratio:.1f}%",
                delta=None,
                help="Average unique value percentage"
            )
        
        with col4:
            duplicate_rows = len(df) - len(df.drop_duplicates())
            st.metric(
                label="Duplicate Rows",
                value=duplicate_rows,
                delta=None,
                help="Number of completely duplicate records"
            )
        
        # Show missing values per column
        st.divider()
        st.write("### Missing Values by Column")
        missing_data = df.isnull().sum().sort_values(ascending=False)
        missing_data = missing_data[missing_data > 0]
        
        if len(missing_data) > 0:
            fig, ax = plt.subplots(figsize=(10, 4))
            missing_data.plot(kind='barh', ax=ax, color='coral')
            ax.set_xlabel("Count of Missing Values")
            ax.set_title("Missing Data Distribution")
            st.pyplot(fig)
        else:
            st.success("No missing values detected!")
    
    with tab3:
        # Statistics
        st.write("### Numeric Column Statistics")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            # Allow user to select columns for detailed stats
            selected_cols = st.multiselect(
                "Select columns to analyze",
                numeric_cols,
                default=list(numeric_cols)[:3],
                key="kpi_numeric_cols_select"
            )
            
            if selected_cols:
                stats_df = df[selected_cols].describe().T
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.dataframe(stats_df[['mean', 'std', 'min', 'max']], use_container_width=True)
                
                with col2:
                    # Distribution charts
                    for col in selected_cols[:2]:  # Show distributions for first 2 columns
                        fig, ax = plt.subplots(figsize=(8, 3))
                        df[col].hist(bins=30, ax=ax, color='skyblue', edgecolor='black')
                        ax.set_title(f"Distribution of {col}")
                        ax.set_xlabel(col)
                        ax.set_ylabel("Frequency")
                        st.pyplot(fig)
        else:
            st.info("No numeric columns available for statistical analysis")
    
    with tab4:
        # Predictions Summary
        predictions = output.get("predictions", {})
        
        if predictions:
            st.write("### Prediction Model Summary")
            
            pred_data = []
            for target, info in predictions.items():
                if "error" not in info:
                    pred_data.append({
                        "Target": target,
                        "Task": info.get("task", "N/A"),
                        "Model": info.get("best_model", "N/A"),
                        "Score": f"{info.get('r2_score', info.get('accuracy', 'N/A')):.4f}" if isinstance(info.get('r2_score', info.get('accuracy')), (int, float)) else "N/A"
                    })
                else:
                    pred_data.append({
                        "Target": target,
                        "Task": "Error",
                        "Model": "N/A",
                        "Score": info.get("error", "Unknown error")
                    })
            
            if pred_data:
                pred_df = pd.DataFrame(pred_data)
                st.dataframe(pred_df, use_container_width=True)
                
                # Model performance visualization
                st.divider()
                st.write("### Model Performance Overview")
                
                scores = []
                targets = []
                for data in pred_data:
                    if data["Score"] != "N/A" and data["Task"] != "Error":
                        try:
                            score = float(data["Score"])
                            scores.append(score)
                            targets.append(data["Target"][:20])  # Truncate long names
                        except:
                            pass
                
                if scores:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.barh(targets, scores, color='lightgreen')
                    ax.set_xlabel("Score (R² or Accuracy)")
                    ax.set_title("Model Performance Comparison")
                    ax.set_xlim([0, 1])
                    st.pyplot(fig)
        else:
            st.info("No predictions available. Run AI Analysis to generate predictions.")

# ----------------------------
# Industry Selection
# ----------------------------
industry_options = ["None", "Finance", "Healthcare", "Retail", "Manufacturing"]
selected_industry = st.selectbox("Select Industry for Smart Insights (Industry Mode)", industry_options)
industry_value = None if selected_industry == "None" else selected_industry

# ----------------------------
# File Upload - Multi-Format Support
# ----------------------------
def load_file(uploaded_file):
    """
    Load data from various file formats.
    Supports: CSV, XLSX, JSON, Parquet, TSV, ODS, SQLite, HDF5
    """
    filename = uploaded_file.name.lower()
    
    try:
        if filename.endswith(".csv"):
            try:
                return pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                return pd.read_csv(uploaded_file, encoding="ISO-8859-1")
        
        elif filename.endswith(".xlsx"):
            return pd.read_excel(uploaded_file, engine="openpyxl")
        
        elif filename.endswith(".xls"):
            return pd.read_excel(uploaded_file, engine="xlrd")
        
        elif filename.endswith(".ods"):
            return pd.read_excel(uploaded_file, engine="odf")
        
        elif filename.endswith(".tsv"):
            try:
                return pd.read_csv(uploaded_file, sep="\t")
            except UnicodeDecodeError:
                return pd.read_csv(uploaded_file, sep="\t", encoding="ISO-8859-1")
        
        elif filename.endswith(".json"):
            return pd.read_json(uploaded_file)
        
        elif filename.endswith(".parquet"):
            return pd.read_parquet(uploaded_file)
        
        elif filename.endswith(".db") or filename.endswith(".sqlite"):
            import sqlite3
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            try:
                conn = sqlite3.connect(tmp_path)
                tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
                if len(tables) > 0:
                    table_name = tables.iloc[0, 0]
                    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                    conn.close()
                    return df
                else:
                    conn.close()
                    raise ValueError("No tables found in SQLite database")
            finally:
                os.unlink(tmp_path)
        
        elif filename.endswith(".h5") or filename.endswith(".hdf5"):
            return pd.read_hdf(uploaded_file)
        
        else:
            raise ValueError(f"Unsupported file format: {filename}")
    
    except Exception as e:
        raise Exception(f"Error loading {filename}: {str(e)}")


def parse_pasted_data(data_string, format_type="csv"):
    """Parse pasted data with intelligent format detection for ANY dataset"""
    import json
    from io import StringIO
    import re
    
    try:
        if format_type == "json":
            return pd.read_json(StringIO(data_string))
        
        # For CSV-like formats, intelligently detect and parse any format
        if format_type == "csv" or format_type == "tsv":
            lines = data_string.strip().split('\n')
            if not lines:
                raise ValueError("No data provided")
            
            first_line = lines[0]
            second_line = lines[1] if len(lines) > 1 else ""
            
            # Strategy: Analyze first 2 lines to determine the actual delimiter
            potential_delimiters = ['\t', ',', ';', '|', ' ']
            delimiter_scores = {}
            
            for delim in potential_delimiters:
                # Count delimiter occurrences in first line
                count_line1 = first_line.count(delim)
                count_line2 = second_line.count(delim) if second_line else count_line1
                
                # Consistency is important - same delimiter should appear same # of times in both lines
                consistency = 0 if count_line1 != count_line2 else abs(count_line1 - count_line2)
                
                # Score: prioritize consistency and frequency
                score = (count_line1 + count_line2) * 2 - consistency
                delimiter_scores[delim] = score
            
            # Handle TSV explicitly - use tab if selected
            if format_type == "tsv":
                delimiter = '\t'
            else:
                # Choose delimiter with highest score (but must have at least 1 occurrence)
                valid_delimiters = {d: s for d, s in delimiter_scores.items() if delimiter_scores[d] > 0}
                
                if valid_delimiters:
                    delimiter = max(valid_delimiters.items(), key=lambda x: x[1])[0]
                else:
                    # Fallback to whitespace if no clear delimiter
                    delimiter = r'\s+'
            
            # Try parsing with detected delimiter
            try:
                if delimiter == r'\s+':
                    # Whitespace: use Python engine for regex support
                    df = pd.read_csv(
                        StringIO(data_string), 
                        sep=delimiter, 
                        engine='python',
                        skipinitialspace=True
                    )
                else:
                    # Regular delimiter
                    df = pd.read_csv(
                        StringIO(data_string), 
                        sep=delimiter,
                        skipinitialspace=True
                    )
            except Exception as e1:
                # If detected delimiter fails, try alternatives in order
                for fallback_delim in ['\t', ',', ';', '|', r'\s+']:
                    try:
                        if fallback_delim == r'\s+':
                            df = pd.read_csv(
                                StringIO(data_string), 
                                sep=fallback_delim, 
                                engine='python',
                                skipinitialspace=True
                            )
                        else:
                            df = pd.read_csv(
                                StringIO(data_string), 
                                sep=fallback_delim,
                                skipinitialspace=True
                            )
                        break
                    except Exception:
                        continue
                else:
                    # If all fail, raise error
                    raise e1
            
            # Clean up column names
            df.columns = df.columns.str.strip()
            
            # Remove completely empty columns
            df = df.dropna(axis=1, how='all')
            
            # Remove rows that are completely NaN
            df = df.dropna(how='all')
            
            return df
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    except Exception as e:
        raise Exception(f"Error parsing {format_type} data: {str(e)}")


def coerce_types(df):
    """Intelligently convert string columns to numeric/datetime where possible"""
    df = df.copy()
    
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try numeric conversion
            try:
                numeric_col = pd.to_numeric(df[col], errors='coerce')
                # If more than 80% converted successfully, use numeric
                if numeric_col.notna().sum() / len(numeric_col) > 0.8:
                    df[col] = numeric_col
                    continue
            except:
                pass
            
            # Try datetime conversion
            try:
                datetime_col = pd.to_datetime(df[col], errors='coerce')
                # If more than 80% converted successfully, use datetime
                if datetime_col.notna().sum() / len(datetime_col) > 0.8:
                    df[col] = datetime_col
                    continue
            except:
                pass
    
    return df


# ----------------------------
# Upload or Paste Data
# ----------------------------
st.subheader("Data Source")
upload_tab, paste_tab = st.tabs(["Upload File", "Paste Data"])

df = None
uploaded_file = None
pasted_data = None

with upload_tab:
    st.write("Upload a dataset file from your computer")
    uploaded_file = st.file_uploader(
        "Upload Dataset",
        type=["csv", "xlsx", "xls", "ods", "tsv", "json", "parquet", "db", "sqlite", "h5", "hdf5"],
        key="file_uploader"
    )
    
    if uploaded_file:
        try:
            df = load_file(uploaded_file)
            df = coerce_types(df)  # Convert string columns to numeric/datetime
            st.success(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns from {uploaded_file.name}")
        except Exception as e:
            st.error(f"Failed to load file: {e}")
            st.stop()

        # Reset autopilot flag when a new file is uploaded
        st.session_state.autopilot_ran = False

        # Show file format info
        with st.expander("Supported File Formats"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("""
                **Spreadsheets:**
                - CSV
                - XLSX
                - XLS
                - ODS
                - TSV
                """)
            with col2:
                st.write("""
                **Data Formats:**
                - JSON
                - Parquet
                - HDF5
                - SQLite
                """)

with paste_tab:
    st.write("Paste your dataset directly")
    
    col_format1, col_format2 = st.columns([1, 3])
    with col_format1:
        paste_format = st.selectbox("Data Format", ["CSV", "JSON", "TSV"], key="paste_format_select")
    
    if paste_format == "CSV":
        st.caption("Format: Each row on a new line, columns separated by commas")
        example = """Name,Age,City
John,25,NYC
Jane,30,LA
Bob,35,Chicago"""
    elif paste_format == "JSON":
        st.caption("Format: JSON array of objects")
        example = """[
  {"Name": "John", "Age": 25, "City": "NYC"},
  {"Name": "Jane", "Age": 30, "City": "LA"},
  {"Name": "Bob", "Age": 35, "City": "Chicago"}
]"""
    else:  # TSV
        st.caption("Format: Each row on a new line, columns separated by tabs")
        example = """Name\tAge\tCity
John\t25\tNYC
Jane\t30\tLA
Bob\t35\tChicago"""
    
    pasted_data = st.text_area(
        "Paste your data below:",
        height=200,
        placeholder=example,
        key="data_paste_area"
    )
    
    if st.button("Load Pasted Data", key="load_paste_btn"):
        if pasted_data.strip():
            try:
                st.session_state.pasted_df = parse_pasted_data(pasted_data, paste_format.lower())
                st.session_state.pasted_df = coerce_types(st.session_state.pasted_df)  # Convert string columns to numeric/datetime
                st.success(f"Loaded {st.session_state.pasted_df.shape[0]} rows x {st.session_state.pasted_df.shape[1]} columns from pasted data")
                st.session_state.autopilot_ran = False
            except Exception as e:
                st.error(f"Failed to parse data: {e}")
        else:
            st.warning("Please paste some data first")

# Handle both upload and paste data
if uploaded_file:
    df = df if df is not None else load_file(uploaded_file)
elif "pasted_df" in st.session_state:
    df = st.session_state.pasted_df

if df is not None:

    # ----------------------------
    # Autofix
    # ----------------------------
    autofix = st.checkbox("Enable Autofix Mode (auto-fill missing / remove constant columns)")
    if autofix:
        try:
            df, autofix_summary = apply_autofix(df)
            st.success("Autofix applied successfully!")
            
            # Display Autofix Summary
            with st.expander("Autofix Details", expanded=False):
                if autofix_summary.get("missing", {}).get("filled"):
                    st.write("**Filled Missing Values (≤20% nulls):**")
                    for item in autofix_summary["missing"]["filled"]:
                        st.write(f"  • {item['column']}: {item['null_pct']} nulls → {item['action']}")
                
                if autofix_summary.get("missing", {}).get("dropped"):
                    st.write("**Dropped Rows with High Nulls (>20%):**")
                    for item in autofix_summary["missing"]["dropped"]:
                        st.write(f"  • {item['column']}: {item['null_pct']} nulls → Dropped {item['rows_before'] - item['rows_after']} rows")
        except Exception as e:
            st.error(f"Autofix failed: {e}")

    column_types = {}
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_datetime64_any_dtype(dtype) or pd.api.types.is_datetime64tz_dtype(dtype):
            column_types[col] = "datetime"
        elif dtype == "object":
            column_types[col] = "text"
        else:
            column_types[col] = "numerical"

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
        if st.button("Ask AI", key="talk_to_data_btn") and user_question:
            with st.spinner("🤖 Generating answer..."):
                answer = talk_to_data_ai(df, query=user_question)
                display_ai_answer(answer)
    else:
        display_ai_answer("Talk-to-Your-Data AI module not installed. Skipping.")

    # ----------------------------
    # AI Analytics Autopilot
    # ----------------------------
    st.subheader("🤖 AI Analytics Autopilot")
    autopilot_mode = st.checkbox("Enable AI Analytics Autopilot (Run full analysis automatically)", key="autopilot_check")
    
    # Run autopilot automatically only once per file/paste load
    if autopilot_mode and not st.session_state.autopilot_ran:
        st.session_state.autopilot_ran = True
        with st.spinner("🚀 Running AI Analytics Autopilot..."):
            auto_column_types = {}
            for col in df.columns:
                dtype = df[col].dtype
                if pd.api.types.is_datetime64_any_dtype(dtype) or pd.api.types.is_datetime64tz_dtype(dtype):
                    auto_column_types[col] = "datetime"
                elif pd.api.types.is_numeric_dtype(dtype):
                    auto_column_types[col] = "numerical"
                else:
                    auto_column_types[col] = "text"

            st.session_state.output = route_to_engines(
                df=df,
                column_types=auto_column_types,
                autofix=True,
                industry=industry_value
            )
        save_outputs(st.session_state.output)
        st.success("✅ AI Analytics Autopilot Complete!")
        st.rerun()

# ----------------------------
# Run AI Analysis (Manual)
# ----------------------------
if df is not None and st.button("Run AI Analysis", key="run_analysis_btn"):
    st.session_state.analysis_done = False  # Reset flag during analysis
    with st.spinner("🚀 Running AI analysis..."):
        # Fallback in case column_types is not defined
        if column_types is None:
            column_types = {}
            for col in df.columns:
                dtype = df[col].dtype
                if pd.api.types.is_datetime64_any_dtype(dtype) or pd.api.types.is_datetime64tz_dtype(dtype):
                    column_types[col] = "datetime"
                elif dtype == "object":
                    column_types[col] = "text"
                else:
                    column_types[col] = "numerical"
        st.session_state.output = route_to_engines(df, column_types, autofix=autofix, industry=industry_value)
        st.session_state.analysis_done = True  # Set flag after analysis completes

        # ----------------------------
        # SAVE FILES FOR DOWNLOAD
        # ----------------------------
        import json
        import os
        import pandas as pd
        os.makedirs("outputs", exist_ok=True)

        output = st.session_state.output

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
    for i, file_name in enumerate(downloadable_files):
        if os.path.exists(file_name):
            with open(file_name, "rb") as f:
                unique_id = uuid.uuid4().hex
                st.download_button(
                    f"Download {os.path.basename(file_name)}",
                    f,
                    file_name=os.path.basename(file_name),
                    key=f"download_btn_{i}_{unique_id}"
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
            file_name="adaptive_insights.json",
            key="adaptive_insights_download"
        )
    else:
        st.info("No adaptive insights generated.")
