
import seaborn as sns
import plotly.express as px

import sys, os
sys.path.append(os.path.abspath(os.getcwd()))
import os
os.makedirs("outputs", exist_ok=True)
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
# Display & Download Generated Files
# ----------------------------
def display_generated_files():
    import os, json, pandas as pd, streamlit as st

    output_folder = "outputs"
    files = {
        "Predictions CSV": os.path.join(output_folder, "predictions.csv"),
        "Recommendations CSV": os.path.join(output_folder, "recommendations.csv"),
        "Recommendations JSON": os.path.join(output_folder, "recommendations.json"),
        "Report PDF": os.path.join(output_folder, "report.pdf")
    }

    for name, path in files.items():
        if os.path.exists(path):
            st.write(f"**{name}:**")
            if path.endswith(".csv"):
                df_file = pd.read_csv(path)
                st.dataframe(df_file.head(), use_container_width=True)
            elif path.endswith(".json"):
                with open(path) as f:
                    st.json(json.load(f))
            st.download_button(f"Download {name}", open(path, "rb"), file_name=os.path.basename(path))
        else:
            st.warning(f"{name} not generated yet. Run analysis to create it.")
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
# P&L Dashboard
# ----------------------------
def display_pl_dashboard(df):
    st.subheader("💰 Profit & Loss Dashboard")

    # Try detecting financial columns automatically
    revenue_cols = [c for c in df.columns if "revenue" in c.lower() or "sales" in c.lower()]
    cost_cols = [c for c in df.columns if "cost" in c.lower() or "expense" in c.lower()]
    profit_cols = [c for c in df.columns if "profit" in c.lower()]

    if not revenue_cols or not cost_cols:
        st.info("Dataset does not appear to contain financial columns (revenue/cost).")
        return

    revenue_col = revenue_cols[0]
    cost_col = cost_cols[0]

    if profit_cols:
        profit_col = profit_cols[0]
    else:
        df["Profit"] = df[revenue_col] - df[cost_col]
        profit_col = "Profit"

    # ----------------------------
    # KPI Metrics
    # ----------------------------
    total_revenue = df[revenue_col].sum()
    total_cost = df[cost_col].sum()
    total_profit = df[profit_col].sum()
    margin = (total_profit / total_revenue) * 100 if total_revenue != 0 else 0

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Revenue", f"${total_revenue:,.2f}")
    col2.metric("Total Cost", f"${total_cost:,.2f}")
    col3.metric("Total Profit", f"${total_profit:,.2f}")
    col4.metric("Profit Margin", f"{margin:.2f}%")

    st.divider()

    # ----------------------------
    # Revenue vs Cost Chart
    # ----------------------------
    st.write("### Revenue vs Cost")

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(df[revenue_col].values, label="Revenue")
    ax.plot(df[cost_col].values, label="Cost")
    ax.legend()
    ax.set_title("Revenue vs Cost Trend")
    st.pyplot(fig)

    # ----------------------------
    # Profit Distribution
    # ----------------------------
    st.write("### Profit Distribution")

    fig2, ax2 = plt.subplots(figsize=(8,4))
    df[profit_col].hist(bins=30, ax=ax2)
    ax2.set_title("Profit Distribution")
    st.pyplot(fig2)

    # ----------------------------
    # Top Profit Records
    # ----------------------------
    st.write("### Top Profit Records")
    top_profit = df.sort_values(profit_col, ascending=False).head(10)
    st.dataframe(top_profit)
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
     # show KPI cards
    display_kpi_cards(df, st.session_state.output)

    # show P&L dashboard
    display_pl_dashboard(df)
# Create optimized dataset for analysis
# ----------------------------
if df is None or df.empty:
    st.error("❌ No dataset loaded. Please upload a valid file first.")
    st.stop()

df_analysis = df.sample(
    n=50000 if len(df) > 50000 else len(df),
    random_state=42
)
# -----------------------------
# 3️⃣ Feature Engineering & Dynamic Dashboard (Fully Dynamic)
# -----------------------------
if df is not None and not df.empty:
    df_filtered = df.copy()

    # -----------------------------
    # Sidebar Filters (Dynamic for ALL Categorical Columns)
    # -----------------------------
    st.sidebar.title("Filter & Search")
    categorical_cols = df_filtered.select_dtypes(include=['object', 'category']).columns.tolist()

    for col in categorical_cols:
        unique_vals = df_filtered[col].dropna().unique().tolist()
        if unique_vals:
            # Use ALL unique values as default (truly dynamic)
            selected = st.sidebar.multiselect(f"Filter {col}", options=unique_vals, default=unique_vals)
            df_filtered = df_filtered[df_filtered[col].isin(selected)]

    # -----------------------------
    # Feature Engineering (Optional)
    # -----------------------------
    numeric_cols = df_filtered.select_dtypes(include=np.number).columns.tolist()
    
    # Dynamic numeric feature if stock-like columns exist
    if 'Open' in df_filtered.columns and 'Close' in df_filtered.columns:
        df_filtered['Daily_Return'] = (df_filtered['Close'] - df_filtered['Open']) / df_filtered['Open']
        df_filtered['High_Risk'] = df_filtered['Daily_Return'].apply(lambda x: 'Yes' if abs(x) > 0.05 else 'No')
        if 'High_Risk' not in categorical_cols:
            categorical_cols.append('High_Risk')

    # -----------------------------
    # Interactive Plots (Dynamic)
    # -----------------------------
    st.title("📈 Data Dashboard")

    # X-axis: prefer 'Date', else first non-numeric, else first numeric
    x_candidates = df_filtered.select_dtypes(exclude=np.number).columns.tolist()
    if 'Date' in df_filtered.columns:
        x_col = 'Date'
    elif x_candidates:
        x_col = x_candidates[0]
    elif numeric_cols:
        x_col = numeric_cols[0]
    else:
        x_col = None

    # Y-axis: any numeric column
    y_col = st.selectbox("Select Column to Plot", numeric_cols) if numeric_cols else None

    if x_col and y_col:
        fig = px.line(df_filtered, x=x_col, y=y_col, title=f"{y_col} Trend vs {x_col}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough columns to generate a plot.")

    # -----------------------------
    # Summary Statistics
    # -----------------------------
    st.subheader("Summary Statistics")
    if numeric_cols:
        st.write(df_filtered[numeric_cols].describe())
    else:
        st.info("No numeric columns to summarize.")

    # -----------------------------
    # Column Management (Dynamic)
    # -----------------------------
    st.subheader("📊 Column Management")
    cols_to_remove = st.multiselect("Remove Columns", df_filtered.columns.tolist())
    cols_to_add = st.text_input("Add New Calculated Column (example: Col1 - Col2)", placeholder="e.g., Col1 - Col2")

    df_manage = df_filtered.copy()
    if cols_to_remove:
        df_manage = df_manage.drop(columns=cols_to_remove)
    if cols_to_add:
        try:
            df_manage[cols_to_add] = df_manage.eval(cols_to_add)
            st.success(f"Column '{cols_to_add}' added!")
        except Exception as e:
            st.error(f"Failed to add column: {e}")

    st.write(df_manage.head(10))

    # -----------------------------
    # Search in Dataset (Dynamic)
    # -----------------------------
    st.subheader("🔍 Search in Dataset")
    if not df_manage.empty:
        search_col = st.selectbox("Column to Search", df_manage.columns)
        search_val = st.text_input("Value to Search For")
        if search_val:
            search_results = df_manage[
                df_manage[search_col].astype(str).str.strip().str.upper().str.contains(search_val.strip().upper())
            ]
            if not search_results.empty:
                st.write(search_results)
            else:
                st.info(f"No results found for '{search_val}' in column '{search_col}'")
else:
    st.warning("No dataset loaded or dataset is empty.")
# -----------------------------
# -----------------------------
# Safe correlation heatmap function
# -----------------------------
def safe_corr_heatmap(df, numeric_cols):
    if numeric_cols and len(numeric_cols) > 1:

        # 👉 STEP 1: sample data for speed (IMPORTANT)
        df_sample = df[numeric_cols].sample(
            n=min(5000, len(df)),
            random_state=42
        )

        # 👉 STEP 2: limit columns (VERY IMPORTANT for 200 cols)
        limited_cols = numeric_cols[:20]

        corr = df_sample[limited_cols].corr()

        # 👉 STEP 3: make figure lighter
        fig, ax = plt.subplots(figsize=(8, 5))

        # ❌ remove annot=True (this causes lag)
        sns.heatmap(corr, cmap="coolwarm", ax=ax)

        st.pyplot(fig)

    else:
        st.info("Not enough numeric columns for correlation heatmap.")
# -----------------------------
# Check if dataframe is loaded
# -----------------------------
if df is not None and not df.empty:
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
else:
    numeric_cols, cat_cols, date_cols = [], [], []

# -----------------------------
# Dashboard Tabs
# -----------------------------
if numeric_cols:
    st.subheader("📊 AI Data Analytics Dashboard Tabs")
    tab1, tab2, tab3 = st.tabs(["Correlation Heatmap", "Scatter Plot", "Other Charts"])

    # ---------------------
    # Tab 1: Correlation Heatmap
    # ---------------------
    with tab1:
        st.write("### Correlation Heatmap")
        safe_corr_heatmap(df, numeric_cols)

    # ---------------------
    # Tab 2: Scatter Plot Explorer
    # ---------------------
    with tab2:
        st.write("### Scatter Plot Explorer")
        if len(numeric_cols) >= 2:
            x_axis = st.selectbox(
                "X-axis",
                numeric_cols,
                key=f"scatter_x_tab2_{'_'.join(numeric_cols)}"
            )
            y_axis = st.selectbox(
                "Y-axis",
                numeric_cols,
                key=f"scatter_y_tab2_{'_'.join(numeric_cols)}"
            )
            fig_scatter = px.scatter(
                df,
                x=x_axis,
                y=y_axis,
                title=f"{x_axis} vs {y_axis}",
                color=None
            )
            st.plotly_chart(
                fig_scatter,
                use_container_width=True,
                key=f"scatter_chart_tab2_{x_axis}_{y_axis}"
            )
        else:
            st.info("Need at least 2 numeric columns to create scatter plots.")

# ---------------------
# Tab 3: Other Charts Example
# ---------------------

if numeric_cols:

    with tab3:
        st.write("### Line Chart Example")

        col = st.selectbox(
            "Select Column",
            numeric_cols,
            key=f"line_col_tab3_{'_'.join(numeric_cols)}"
        )

        # ✅ Prepare data safely
        df_plot = df.copy()

        if isinstance(df_plot.index, pd.MultiIndex):
            df_plot = df_plot.reset_index()

        # ✅ Safe X-axis selection
        x_col = "Date" if "Date" in df_plot.columns else df_plot.columns[0]

        # ✅ Plot
        fig_line = px.line(
            df_plot,
            x=x_col,
            y=col,
            title=f"Line Chart of {col}"
        )

        st.plotly_chart(
            fig_line,
            width='stretch',
            key=f"line_chart_tab3_{col}"
        )

else:
    st.warning("No numeric columns found in your dataset for visualization.")
#----------------------------
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.subheader("📊 Automatic Data Visualizations")

# ----------------------------
# AUTOMATIC DATA VISUALIZATIONS (WITH UNIQUE KEYS)
# ----------------------------
st.subheader("📊 Automatic Data Visualizations")

if df is not None:
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
else:
    numeric_cols, cat_cols, date_cols = [], [], []

tab1, tab2, tab3, tab4 = st.tabs([
    "Distributions",
    "Relationships",
    "Categories",
    "Time Series"
])

# ----------------------------
# 1️⃣ Numeric Distributions
# ----------------------------
with tab1:
    if numeric_cols:
        st.write("### Numeric Distributions")
        for col in numeric_cols[:4]:
            fig = px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}")
            st.plotly_chart(fig, use_container_width=True, key=f"hist_{col}")

        st.write("### Boxplots (Outlier Detection)")
        col_choice = st.selectbox("Select numeric column for boxplot", numeric_cols)
        fig = px.box(df, y=col_choice, title=f"Outliers in {col_choice}")
        st.plotly_chart(fig, use_container_width=True, key=f"box_{col_choice}")
    else:
        st.info("No numeric columns available for visualization.")

# ----------------------------
# 2️⃣ Relationships
# ----------------------------
with tab2:
    st.write("### Scatter Plot Explorer")

    # Check if there are at least 2 numeric columns
    if len(numeric_cols) >= 2:
        # Unique keys for Streamlit widgets
        x_axis = st.selectbox("Select X-axis", numeric_cols, key="scatter_x_tab2")
        y_axis = st.selectbox("Select Y-axis", numeric_cols, key="scatter_y_tab2")

        # Create scatter plot using Plotly
        fig_scatter = px.scatter(
            df,
            x=x_axis,
            y=y_axis,
            title=f"{x_axis} vs {y_axis}",
            labels={x_axis: x_axis, y_axis: y_axis},
        )
        st.plotly_chart(fig_scatter, use_container_width=True, key=f"scatter_chart_tab2")

        # Optional: show correlation heatmap below
        st.write("### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    else:
        st.info("Need at least two numeric columns to create scatter plots.")
# ----------------------------
# 3️⃣ Categorical Analysis
# ----------------------------
with tab3:
    if cat_cols:
        st.write("### Category Frequency")
        cat_col = st.selectbox("Select categorical column", cat_cols)
        counts = df[cat_col].value_counts().reset_index()
        counts.columns = [cat_col, "count"]
        fig = px.bar(counts, x=cat_col, y="count", title=f"{cat_col} Distribution")
        st.plotly_chart(fig, use_container_width=True, key=f"bar_{cat_col}")
    else:
        st.info("No categorical columns detected.")

# ----------------------------
# 4️⃣ Time Series Analysis
# ----------------------------
with tab4:
    if date_cols and numeric_cols:
        st.write("### Time Series Trends")
        date_col = st.selectbox("Select date column", date_cols)
        value_col = st.selectbox("Select value column", numeric_cols)
        df_sorted = df.sort_values(date_col)
        fig = px.line(df_sorted, x=date_col, y=value_col, title=f"{value_col} over time")
        st.plotly_chart(fig, use_container_width=True, key=f"line_{value_col}_{date_col}")
    else:
        st.info("Requires at least one datetime column and one numeric column.")
# ----------------------------
# END OF AUTOMATIC VISUALIZATIONS
# ----------------------------

# Tabs for cleaner UI
tab1, tab2, tab3, tab4 = st.tabs([
    "Distributions",
    "Relationships",
    "Categories",
    "Time Series"
])

# ----------------------------
# NUMERIC DISTRIBUTIONS
# ----------------------------
with tab1:

    if len(numeric_cols) > 0:

        st.write("### Numeric Distributions")

        for col in numeric_cols[:4]:

            fig = px.histogram(
                df,
                x=col,
                nbins=30,
                title=f"Distribution of {col}"
            )

            st.plotly_chart(fig, use_container_width=True)

        st.write("### Boxplots (Outlier Detection)")

        col_choice = st.selectbox(
            "Select numeric column",
            numeric_cols,
            key="boxplot_column"
        )

        fig = px.box(
            df,
            y=col_choice,
            title=f"Outliers in {col_choice}"
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No numeric columns available.")

# ----------------------------
# RELATIONSHIPS
# ----------------------------
with tab2:

    if len(numeric_cols) >= 2:

        st.write("### Scatter Plot Explorer")

        col1, col2 = st.columns(2)

        with col1:
            x_axis = st.selectbox(
                "Select X-axis",
                numeric_cols,
                key="scatter_x"
            )

        with col2:
            y_axis = st.selectbox(
                "Select Y-axis",
                numeric_cols,
                key="scatter_y"
            )

        fig = px.scatter(
            df,
            x=x_axis,
            y=y_axis,
            title=f"{x_axis} vs {y_axis}"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.write("### Correlation Heatmap")

        fig, ax = plt.subplots(figsize=(10,6))

        sns.heatmap(
            df[numeric_cols].corr(),
            annot=True,
            cmap="coolwarm",
            ax=ax
        )

        st.pyplot(fig)

    else:
        st.info("Need at least two numeric columns for relationship analysis.")

# ----------------------------
# CATEGORICAL ANALYSIS
# ----------------------------
with tab3:

    if len(cat_cols) > 0:

        st.write("### Category Frequency")

        cat_col = st.selectbox(
            "Select category column",
            cat_cols,
            key="category_column"
        )

        counts = df[cat_col].value_counts().reset_index()

        counts.columns = [cat_col, "count"]

        fig = px.bar(
            counts,
            x=cat_col,
            y="count",
            title=f"{cat_col} Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No categorical columns detected.")

# ----------------------------
# TIME SERIES ANALYSIS
# ----------------------------
with tab4:

    if len(date_cols) > 0 and len(numeric_cols) > 0:

        st.write("### Time Series Trends")

        date_col = st.selectbox(
            "Select date column",
            date_cols,
            key="time_date"
        )

        value_col = st.selectbox(
            "Select value column",
            numeric_cols,
            key="time_value"
        )

        df_sorted = df.sort_values(date_col)

        fig = px.line(
            df_sorted,
            x=date_col,
            y=value_col,
            title=f"{value_col} over time"
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Dataset does not contain datetime columns.")

    # ----------------------------
# Autofix + Column Types (Safe)
# ----------------------------
if df is not None and not df.empty:
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

    # Detect column types safely
    column_types = {}
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_datetime64_any_dtype(dtype) or pd.api.types.is_datetime64tz_dtype(dtype):
            column_types[col] = "datetime"
        elif dtype == "object":
            column_types[col] = "text"
        else:
            column_types[col] = "numerical"
else:
    column_types = {}
    st.info("No dataset loaded. Upload or paste data to enable Autofix and column analysis.")
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
            output = talk_to_data_ai(df, query=user_question)
            st.session_state.output = output  # save to session

            # Safely display answer
            answer_text = output.get("answer", "No answer returned") if output else "No output generated"
            display_ai_answer(answer_text)

            # Save outputs
            save_outputs(st.session_state.output)
            st.success("✅ Predictions and recommendations saved to 'outputs/' folder!")
    else:
        display_ai_answer("Talk-to-Your-Data AI module not installed. Skipping.") # ----------------------------
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

    #st.success("✅ AI Analysis Complete! Files saved in outputs/")

    # ----------------------------
    # Display Outputs
    # ----------------------------
    output = st.session_state.output
    save_outputs(output)
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
