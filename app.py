
import seaborn as sns
import plotly.express as px

import sys, os
sys.path.append(os.path.abspath(os.getcwd()))
os.makedirs("outputs", exist_ok=True)
GRAPH_FOLDER = os.path.join("outputs", "graphs")
os.makedirs(GRAPH_FOLDER, exist_ok=True)
DRIFT_BASELINE_PATH = os.path.join("outputs", "drift_baseline_profile.json")
import streamlit as st
# ----------------------------
# SAFE DEFAULTS (to avoid NameError)
# ----------------------------
df = None
column_types = None
autofix = False


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)
# ----------------------------
# Function to save predictions, recommendations, and report
# ----------------------------
def save_outputs(output):
    import os
    import json
    import pandas as pd
    from fpdf import FPDF

    os.makedirs("outputs", exist_ok=True)

    def _prediction_summary_rows(predictions_dict):
        summary_rows = []
        for target, pred_data in (predictions_dict or {}).items():
            if not isinstance(pred_data, dict):
                summary_rows.append({"target": target, "prediction_value": _json_safe(pred_data)})
                continue

            summary_rows.append(
                {
                    "target": target,
                    "task": pred_data.get("task"),
                    "target_source": pred_data.get("target_source"),
                    "best_model": pred_data.get("best_model"),
                    "confidence": pred_data.get("confidence"),
                    "cv_primary_metric": pred_data.get("cv_primary_metric"),
                    "cv_primary_score": pred_data.get("cv_primary_score"),
                    "accuracy": pred_data.get("accuracy"),
                    "balanced_accuracy": pred_data.get("balanced_accuracy"),
                    "f1_score": pred_data.get("f1_score"),
                    "precision": pred_data.get("precision"),
                    "recall": pred_data.get("recall"),
                    "roc_auc": pred_data.get("roc_auc"),
                }
            )
        return summary_rows

    def _prediction_sample_rows(predictions_dict):
        rows = []

        for target, pred_data in (predictions_dict or {}).items():
            if not isinstance(pred_data, dict):
                continue

            sample_predictions = pred_data.get("sample_predictions") or []
            sample_probabilities = pred_data.get("sample_probabilities") or []
            sample_risk_scores = pred_data.get("sample_risk_scores") or []

            for index, prediction in enumerate(sample_predictions):
                row = {
                    "target": target,
                    "sample_index": index,
                    "prediction_value": prediction,
                }
                if index < len(sample_probabilities):
                    row["probability"] = sample_probabilities[index]
                if index < len(sample_risk_scores):
                    row["risk_score"] = sample_risk_scores[index]
                rows.append(row)

        return rows

    def _pdf_safe(text):
        return str(text).encode("latin-1", errors="ignore").decode("latin-1")

    def _break_long_tokens(text, max_token_len=40):
        """Insert spaces into very long tokens so PDF wrapping never fails."""
        tokens = str(text).split(" ")
        normalized = []
        for token in tokens:
            if len(token) <= max_token_len:
                normalized.append(token)
                continue

            parts = [token[i:i + max_token_len] for i in range(0, len(token), max_token_len)]
            normalized.append(" ".join(parts))

        return " ".join(normalized)

    def _pdf_safe_multicell(pdf_obj, text, line_height=6):
        # Guard against cursor drift causing zero/negative available width in fpdf2.
        pdf_obj.set_x(pdf_obj.l_margin)
        available_width = pdf_obj.w - pdf_obj.l_margin - pdf_obj.r_margin
        if available_width <= 10:
            available_width = 180

        safe_text = _pdf_safe(text)
        if not safe_text:
            safe_text = "-"
        safe_text = _break_long_tokens(safe_text, max_token_len=32)

        # Split by explicit new lines to preserve section formatting.
        for chunk in safe_text.splitlines() or [safe_text]:
            try:
                pdf_obj.multi_cell(available_width, line_height, chunk)
            except Exception:
                fallback = chunk[:120] + " ...[truncated]" if len(chunk) > 120 else chunk
                pdf_obj.set_x(pdf_obj.l_margin)
                pdf_obj.cell(0, line_height, fallback, ln=True)

    def _pdf_add_section_title(pdf_obj, title):
        pdf_obj.set_font("Arial", "B", 13)
        pdf_obj.ln(4)
        pdf_obj.set_x(pdf_obj.l_margin)
        pdf_obj.cell(0, 8, _pdf_safe(title), ln=True)

    def _pdf_add_bullet(pdf_obj, text):
        pdf_obj.set_font("Arial", "", 11)
        _pdf_safe_multicell(pdf_obj, f"- {text}", line_height=6)

    def _build_prediction_bullets(predictions_dict):
        lines = []
        for target, info in (predictions_dict or {}).items():
            if not isinstance(info, dict):
                lines.append(f"Target {target}: prediction={info}")
                continue

            if info.get("error"):
                lines.append(f"Target {target}: error={info.get('error')}")
                continue

            lines.append(
                f"Target {target}: model={info.get('best_model', 'N/A')}, task={info.get('task', 'N/A')}, "
                f"confidence={info.get('confidence', 'N/A')}"
            )

            for metric_key in ["accuracy", "balanced_accuracy", "f1_score", "precision", "recall", "roc_auc", "r2_score"]:
                metric_val = info.get(metric_key)
                if metric_val is not None:
                    lines.append(f"  {metric_key}: {metric_val}")

            sample_preds = info.get("sample_predictions") or []
            if sample_preds:
                joined = ", ".join(str(v) for v in sample_preds[:5])
                lines.append(f"  sample_predictions: {joined}")

        return lines

    def _build_recommendation_bullets(recommendations_dict):
        lines = []
        for target, rec_list in (recommendations_dict or {}).items():
            if not isinstance(rec_list, list) or not rec_list:
                lines.append(f"Target {target}: no recommendations")
                continue

            lines.append(f"Target {target} recommendations:")
            for idx, rec in enumerate(rec_list, start=1):
                if isinstance(rec, dict):
                    text = rec.get("recommendation") or rec.get("text") or rec.get("action") or str(rec)
                    lines.append(f"  {idx}. {text}")
                else:
                    lines.append(f"  {idx}. {rec}")

        return lines

    def _build_explainability_bullets(explanations_dict):
        lines = []
        for target, payload in (explanations_dict or {}).items():
            if not isinstance(payload, dict):
                lines.append(f"Target {target}: explainability payload unavailable")
                continue

            methods = payload.get("explainability_methods", {}) or {}
            shap_status = "enabled" if methods.get("shap") else "unavailable"
            lime_status = "enabled" if methods.get("lime") else "unavailable"
            lines.append(f"Target {target}: SHAP={shap_status}, LIME={lime_status}")

            if payload.get("error"):
                lines.append(f"  explainability_error: {payload.get('error')}")

            feature_importance = payload.get("feature_importance") or []
            if isinstance(feature_importance, list) and feature_importance:
                lines.append("  top_feature_drivers:")
                for idx, item in enumerate(feature_importance[:5], start=1):
                    if isinstance(item, dict):
                        feature_name = item.get("feature", f"feature_{idx}")
                        score = item.get("mean_abs_shap", item.get("importance", "N/A"))
                        lines.append(f"    {idx}. {feature_name}: {score}")

            sample_explanations = payload.get("sample_explanations") or []
            if isinstance(sample_explanations, list) and sample_explanations:
                lines.append(f"  sample_explanations_count: {len(sample_explanations)}")

                first_sample = sample_explanations[0] if sample_explanations else {}
                contributions = first_sample.get("feature_contributions") if isinstance(first_sample, dict) else None
                if isinstance(contributions, dict) and contributions:
                    ranked = sorted(
                        contributions.items(),
                        key=lambda item: abs(float(item[1])) if str(item[1]).replace(".", "", 1).replace("-", "", 1).isdigit() else 0,
                        reverse=True,
                    )[:3]
                    lines.append("  top_local_contributions_row_0:")
                    for idx, (feature_name, value) in enumerate(ranked, start=1):
                        lines.append(f"    {idx}. {feature_name}: {value}")

            if payload.get("shap_error"):
                lines.append(f"  shap_note: {payload.get('shap_error')}")
            if payload.get("lime_error"):
                lines.append(f"  lime_note: {payload.get('lime_error')}")

        return lines

    def _derive_diabetes_status(result_output):
        risk_scoring = result_output.get("risk_scoring", {}) if isinstance(result_output, dict) else {}
        predictions = result_output.get("predictions", {}) if isinstance(result_output, dict) else {}

        mean_risk = risk_scoring.get("mean_risk_score", risk_scoring.get("average_risk"))
        if mean_risk is None:
            mean_risk = risk_scoring.get("high_risk_share")
            if isinstance(mean_risk, (int, float)):
                mean_risk = float(mean_risk) * 100.0

        high_risk_share = risk_scoring.get("high_risk_share")
        if isinstance(high_risk_share, (int, float)) and high_risk_share <= 1:
            high_risk_share_pct = float(high_risk_share) * 100.0
        elif isinstance(high_risk_share, (int, float)):
            high_risk_share_pct = float(high_risk_share)
        else:
            high_risk_share_pct = None

        detected_yes = None
        for _target, info in (predictions or {}).items():
            if isinstance(info, dict):
                preds = info.get("sample_predictions") or []
                if preds:
                    positive_count = sum(1 for value in preds if str(value).strip().lower() in {"1", "true", "yes", "diabetes", "positive"})
                    detected_yes = positive_count > 0
                    break

        if isinstance(mean_risk, (int, float)):
            if mean_risk >= 70:
                chance_label = "High"
            elif mean_risk >= 35:
                chance_label = "Moderate"
            else:
                chance_label = "Low"
        else:
            chance_label = "Unknown"

        if detected_yes is True:
            detected_label = "Diabetes indicators found"
        elif detected_yes is False:
            detected_label = "No diabetes indicators found in prediction samples"
        else:
            detected_label = "Not enough signal to confirm diabetes detection"

        return {
            "detected_label": detected_label,
            "chance_label": chance_label,
            "mean_risk": mean_risk,
            "high_risk_share_pct": high_risk_share_pct,
        }

    # Predictions JSON
    with open("outputs/predictions.json", "w") as f:
        json.dump(_json_safe(output.get("predictions", {})), f, indent=4)

    # Recommendations JSON
    with open("outputs/recommendations.json", "w") as f:
        json.dump(_json_safe(output.get("recommendations", {})), f, indent=4)

    predictions = output.get("predictions", {})
    explanations = output.get("explanations", {})
    calibration = (((output.get("model_monitoring", {}) or {}).get("training", {}) or {}).get("calibration", {}) or {})
    drift_summary = output.get("drift_summary", {}) if isinstance(output, dict) else {}
    what_if_summary = output.get("what_if_summary", {}) if isinstance(output, dict) else {}
    diabetes_status = _derive_diabetes_status(output)

    # Predictions CSVs
    pd.DataFrame(_prediction_summary_rows(predictions)).to_csv("outputs/predictions.csv", index=False)
    pd.DataFrame(_prediction_sample_rows(predictions)).to_csv("outputs/predictions_samples.csv", index=False)

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
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "KENSOLO AI Report", ln=True, align="C")

        _pdf_add_section_title(pdf, "Analysis Snapshot")
        _pdf_add_bullet(pdf, f"Prediction models: {len(predictions or {})}")
        recommendation_count = sum(len(v) for v in output.get("recommendations", {}).values()) if isinstance(output.get("recommendations"), dict) else 0
        _pdf_add_bullet(pdf, f"Recommendations generated: {recommendation_count}")
        _pdf_add_bullet(pdf, f"Selected model: {output.get('model_monitoring', {}).get('training', {}).get('best_model', output.get('model_monitoring', {}).get('selected_model', 'N/A'))}")
        _pdf_add_bullet(pdf, f"Risk score (mean/avg): {output.get('risk_scoring', {}).get('mean_risk_score', output.get('risk_scoring', {}).get('average_risk', 'N/A'))}")
        _pdf_add_bullet(pdf, f"Diabetes detection status: {diabetes_status.get('detected_label')}")
        _pdf_add_bullet(pdf, f"Chance of getting diabetes: {diabetes_status.get('chance_label')}")
        if diabetes_status.get("mean_risk") is not None:
            _pdf_add_bullet(pdf, f"Mean risk score: {diabetes_status.get('mean_risk')}")
        if diabetes_status.get("high_risk_share_pct") is not None:
            _pdf_add_bullet(pdf, f"High risk share: {round(float(diabetes_status.get('high_risk_share_pct')), 2)}%")

        _pdf_add_section_title(pdf, "Predictions (Point Form)")
        for line in _build_prediction_bullets(predictions):
            _pdf_add_bullet(pdf, line)

        _pdf_add_section_title(pdf, "Recommendations (Point Form)")
        for line in _build_recommendation_bullets(output.get("recommendations", {})):
            _pdf_add_bullet(pdf, line)

        _pdf_add_section_title(pdf, "Explainability Summary")
        for line in _build_explainability_bullets(explanations):
            _pdf_add_bullet(pdf, line)

        _pdf_add_section_title(pdf, "Confidence Calibration")
        if calibration.get("status") == "available":
            _pdf_add_bullet(pdf, f"ECE: {calibration.get('ece')}")
            _pdf_add_bullet(pdf, f"Brier score: {calibration.get('brier_score')}")
            _pdf_add_bullet(pdf, f"Samples: {calibration.get('n_samples')}")
            calibration_bins = calibration.get("bins") or []
            for bin_item in calibration_bins[:5]:
                _pdf_add_bullet(
                    pdf,
                    "Bin {bin}: predicted={pred}, observed={obs}, gap={gap}, n={n}".format(
                        bin=bin_item.get("bin", "N/A"),
                        pred=bin_item.get("predicted_rate", "N/A"),
                        obs=bin_item.get("observed_rate", "N/A"),
                        gap=bin_item.get("gap", "N/A"),
                        n=bin_item.get("count", "N/A"),
                    ),
                )
        else:
            _pdf_add_bullet(pdf, "Calibration metrics unavailable for this run.")

        _pdf_add_section_title(pdf, "Data Drift Snapshot")
        if drift_summary:
            _pdf_add_bullet(pdf, f"Drift status: {drift_summary.get('status', 'N/A')}")
            _pdf_add_bullet(pdf, f"Drift score: {drift_summary.get('drift_score', 'N/A')}")
            top_numeric = drift_summary.get("top_numeric") or []
            if top_numeric:
                _pdf_add_bullet(pdf, "Top numeric shifts:")
                for item in top_numeric[:5]:
                    _pdf_add_bullet(
                        pdf,
                        "{feature}: baseline_mean={base}, current_mean={cur}, z_shift={shift}".format(
                            feature=item.get("feature", "N/A"),
                            base=item.get("baseline_mean", "N/A"),
                            cur=item.get("current_mean", "N/A"),
                            shift=item.get("standardized_shift", "N/A"),
                        ),
                    )
            else:
                _pdf_add_bullet(pdf, "No comparable numeric drift features available.")
        else:
            _pdf_add_bullet(pdf, "Drift baseline not initialized or drift summary unavailable.")

        _pdf_add_section_title(pdf, "What-If Scenario Summary")
        if what_if_summary:
            _pdf_add_bullet(pdf, f"Target: {what_if_summary.get('target', 'N/A')}")
            _pdf_add_bullet(pdf, f"Baseline risk: {what_if_summary.get('baseline_risk_pct', 'N/A')}%")
            _pdf_add_bullet(pdf, f"Scenario risk: {what_if_summary.get('scenario_risk_pct', 'N/A')}%")
            _pdf_add_bullet(pdf, f"Risk delta: {what_if_summary.get('risk_delta_pct', 'N/A')}%")
            changed_features = what_if_summary.get("changed_features") or []
            if changed_features:
                _pdf_add_bullet(pdf, "Adjusted drivers:")
                for feature in changed_features[:6]:
                    _pdf_add_bullet(
                        pdf,
                        "{name}: {base} -> {new}".format(
                            name=feature.get("feature", "N/A"),
                            base=feature.get("baseline", "N/A"),
                            new=feature.get("scenario", "N/A"),
                        ),
                    )
        else:
            _pdf_add_bullet(pdf, "Scenario simulation summary unavailable for this run.")

        pdf.output("outputs/report.pdf")
    except Exception as e:
        st.warning(f"PDF report generation failed: {e}")


def display_issues(issues: dict):
    """Render problem discovery issues in a human-friendly layout."""
    if not issues:
        st.info("No data quality or discovery issues detected.")
        return

    # Ensure consistent ordering
    keys = list(issues.keys())
    cols = st.columns(2)
    i = 0
    for k in keys:
        v = issues.get(k) or {}
        col = cols[i % len(cols)]
        with col:
            sev = (v.get("severity") or v.get("severity", "Medium")).lower()
            color = "#F59E0B" if sev in ("medium",) else ("#DC2626" if sev in ("high",) else "#10B981")
            st.markdown(f"<div style='border-radius:8px;padding:12px;border:1px solid #eee;background:#fff;box-shadow:0 2px 6px rgba(0,0,0,0.03)'>\n<strong>{v.get('column','Unknown')}</strong><br/>\n<small>{v.get('issue_type','')}</small><br/>\n<p style='color:{color};font-weight:700;margin:6px 0'>{v.get('details','')}</p>\n<small>Severity: {v.get('severity','Medium')}</small>\n</div>", unsafe_allow_html=True)
        i += 1


def talk_to_data_fallback(df, query: str):
    """A lightweight local fallback for 'Talk to Your Data' when the AI engine is unavailable.

    Supports simple queries: 'top N outliers', 'describe <col>', 'top values <col>', 'summary'.
    """
    q = (query or "").strip().lower()
    result = {"answer": "", "details": {}}

    try:
        if "outlier" in q:
            # compute IQR-based outliers per numeric column
            numeric = df.select_dtypes(include=[np.number])
            outlier_summary = {}
            for c in numeric.columns:
                s = numeric[c].dropna()
                if s.empty:
                    continue
                q1 = s.quantile(0.25)
                q3 = s.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outliers = s[(s < lower) | (s > upper)]
                pct = 100 * len(outliers) / max(1, len(s))
                outlier_summary[c] = {"count": int(len(outliers)), "pct": f"{pct:.2f}%"}

            # sort by count desc
            sorted_out = sorted(outlier_summary.items(), key=lambda x: x[1]["count"], reverse=True)
            result["answer"] = "Outlier summary computed for numeric columns."
            result["details"] = sorted_out
            return result

        if q.startswith("describe") or q.startswith("summary"):
            # return pandas describe
            cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if cols:
                d = df[cols].describe().to_dict()
                result["answer"] = "Summary statistics for numeric columns"
                result["details"] = d
                return result

        # top values
        if q.startswith("top") and "values" in q:
            # try to parse column
            parts = q.split()
            col = None
            for p in parts:
                if p in df.columns.str.lower().tolist():
                    # find actual column name
                    col = [c for c in df.columns if c.lower() == p][0]
                    break
            if col is None and len(df.columns) > 0:
                col = df.columns[0]
            top = df[col].value_counts().head(10).to_dict()
            result["answer"] = f"Top values for {col}"
            result["details"] = top
            return result

        # fallback: return simple dataset info
        result["answer"] = f"Dataset has {len(df)} rows and {df.shape[1]} columns. Use queries like 'top N outliers' or 'describe <col>'."
        return result

    except Exception as e:
        return {"answer": f"Fallback query failed: {e}", "details": {}}

# ----------------------------
# Display & Download Generated Files
# ----------------------------
def display_generated_files():
    import os, json, pandas as pd, streamlit as st

    output_folder = "outputs"
    files = {
        "Prediction Summary CSV": os.path.join(output_folder, "predictions.csv"),
        "Prediction Samples CSV": os.path.join(output_folder, "predictions_samples.csv"),
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
                    try:
                        pretty_display(json.load(f))
                    except Exception:
                        st.write(f.read())
            st.download_button(f"Download {name}", open(path, "rb"), file_name=os.path.basename(path))
        else:
            st.warning(f"{name} not generated yet. Run analysis to create it.")
# ----------------------------
# MUST BE FIRST STREAMLIT COMMAND
# ----------------------------
st.set_page_config(page_title="IntelliHealth Diabetics Analytics Platform", layout="wide")
st.title("🩺 IntelliHealth — Diabetics Analytics Platform")

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


def pretty_display(data, max_rows=10):
    """Display dicts/lists as readable tables or lists instead of raw JSON."""
    if data is None:
        st.info("No data to display.")
        return

    # Self-critic special formatting
    if isinstance(data, dict) and data.get("risk_flags") is not None:
        # show top-level metrics and list risk flags
        metrics = {k: v for k, v in data.items() if k != "risk_flags"}
        if metrics:
            try:
                st.dataframe(pd.DataFrame([metrics]).T.rename(columns={0: 'value'}), use_container_width=True)
            except Exception:
                st.write(metrics)
        st.markdown("**Risk Flags**")
        for rf in data.get("risk_flags", []):
            st.write(f"- {rf}")
        return

    # Numeric-summary like structures (each value is a dict of stats)
    if isinstance(data, dict):
        first_val = next(iter(data.values())) if data else None
        if isinstance(first_val, dict) and set(first_val.keys()) & {"mean", "std", "min", "max", "25%", "50%", "75%", "count"}:
            try:
                df_stats = pd.DataFrame.from_dict(data, orient="index")
                st.dataframe(df_stats, use_container_width=True)
                return
            except Exception:
                pass

        # generic dict -> table of key/value
        try:
            df = pd.DataFrame([{"key": k, "value": (v if not isinstance(v, (dict, list)) else str(v))} for k, v in data.items()])
            st.dataframe(df.head(max_rows), use_container_width=True)
            return
        except Exception:
            st.write(data)
            return

    if isinstance(data, list):
        try:
            df = pd.DataFrame(data)
            st.dataframe(df.head(max_rows), use_container_width=True)
            return
        except Exception:
            for item in data[:max_rows]:
                st.write(item)
            return

    # fallback
    st.write(data)


def _format_metric_value(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def display_predictions_point_form(predictions):
    if not isinstance(predictions, dict) or not predictions:
        st.info("No predictions available.")
        return

    for target, info in predictions.items():
        st.markdown(f"### Target: {target}")

        if not isinstance(info, dict):
            st.markdown(f"- Prediction: {_format_metric_value(info)}")
            st.divider()
            continue

        if info.get("error"):
            st.error(f"Prediction error: {info.get('error')}")
            st.divider()
            continue

        st.markdown(f"- Task: {info.get('task', 'N/A')}")
        st.markdown(f"- Best model: {info.get('best_model', 'N/A')}")

        if info.get("confidence") is not None:
            st.markdown(f"- Confidence: {_format_metric_value(info.get('confidence'))}")

        score_fields = [
            ("Accuracy", "accuracy"),
            ("Balanced accuracy", "balanced_accuracy"),
            ("F1 score", "f1_score"),
            ("Precision", "precision"),
            ("Recall", "recall"),
            ("ROC AUC", "roc_auc"),
            ("R2 score", "r2_score"),
            ("CV primary score", "cv_primary_score"),
        ]

        available_scores = [
            f"{label}: {_format_metric_value(info.get(key))}"
            for label, key in score_fields
            if info.get(key) is not None
        ]
        if available_scores:
            st.markdown("- Performance metrics:")
            for metric_line in available_scores:
                st.markdown(f"  - {metric_line}")

        sample_predictions = info.get("sample_predictions") or []
        sample_probabilities = info.get("sample_probabilities") or []
        sample_risk_scores = info.get("sample_risk_scores") or []

        if sample_predictions:
            st.markdown("- Sample predictions:")
            for index, prediction in enumerate(sample_predictions[:5]):
                point = f"{index + 1}. prediction={_format_metric_value(prediction)}"
                if index < len(sample_probabilities):
                    point += f", probability={_format_metric_value(sample_probabilities[index])}"
                if index < len(sample_risk_scores):
                    point += f", risk_score={_format_metric_value(sample_risk_scores[index])}"
                st.markdown(f"  - {point}")

        risk_summary = info.get("risk_score_summary") or {}
        if isinstance(risk_summary, dict) and risk_summary:
            st.markdown("- Risk summary:")
            risk_lines = [
                ("Mean risk", risk_summary.get("mean_risk_score", risk_summary.get("average_risk"))),
                ("High risk count", risk_summary.get("high_risk_count")),
                ("Moderate risk count", risk_summary.get("moderate_risk_count")),
                ("Low risk count", risk_summary.get("low_risk_count")),
            ]
            for label, value in risk_lines:
                if value is not None:
                    st.markdown(f"  - {label}: {_format_metric_value(value)}")

        st.divider()


def display_recommendations_point_form(recommendations):
    if not isinstance(recommendations, dict) or not recommendations:
        st.warning("No recommendations were generated.")
        return

    for target, rec_list in recommendations.items():
        st.markdown(f"### Recommendations for {target}")

        if not isinstance(rec_list, list) or not rec_list:
            st.info("No recommendation items for this target.")
            st.divider()
            continue

        for index, rec in enumerate(rec_list, start=1):
            if isinstance(rec, dict):
                text = rec.get("recommendation") or rec.get("text") or rec.get("action") or "Recommendation"
                st.markdown(f"- {index}. {text}")

                priority = rec.get("priority")
                category = rec.get("category")
                reason = rec.get("reason") or rec.get("rationale")
                expected_impact = rec.get("expected_impact")

                if priority is not None:
                    st.markdown(f"  - Priority: {_format_metric_value(priority)}")
                if category is not None:
                    st.markdown(f"  - Category: {_format_metric_value(category)}")
                if reason is not None:
                    st.markdown(f"  - Reason: {_format_metric_value(reason)}")
                if expected_impact is not None:
                    st.markdown(f"  - Expected impact: {_format_metric_value(expected_impact)}")
            else:
                st.markdown(f"- {index}. {_format_metric_value(rec)}")

        st.divider()


def _format_signed_value(value):
    try:
        number = float(value)
        return f"{number:+.4f}"
    except Exception:
        return str(value)


def display_explainability_detailed(explanations):
    if not isinstance(explanations, dict) or not explanations:
        st.info("No explainability output was generated for this run.")
        return

    st.markdown(
        """
        <div style='border:1px solid #dbeafe;background:#eff6ff;padding:12px;border-radius:10px;margin-bottom:10px;'>
            <strong>Model Explainability</strong><br/>
            This section highlights the top features that drive predictions and sample-level contributions.
        </div>
        """,
        unsafe_allow_html=True,
    )

    for target, payload in explanations.items():
        if not isinstance(payload, dict):
            with st.expander(f"Explainability for {target}", expanded=False):
                pretty_display(payload)
            continue

        methods = payload.get("explainability_methods", {}) or {}
        shap_on = bool(methods.get("shap"))
        lime_on = bool(methods.get("lime"))

        with st.expander(f"Explainability for {target}", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("SHAP", "Enabled" if shap_on else "Unavailable")
            with col2:
                st.metric("LIME", "Enabled" if lime_on else "Unavailable")
            with col3:
                st.metric("Sample Explanations", len(payload.get("sample_explanations", []) or []))

            if payload.get("error"):
                st.error(f"Explainability issue: {payload.get('error')}")

            feature_importance = payload.get("feature_importance") or []
            if feature_importance:
                fi_df = pd.DataFrame(feature_importance)
                score_col = "mean_abs_shap" if "mean_abs_shap" in fi_df.columns else ("importance" if "importance" in fi_df.columns else None)

                st.markdown("#### Top Feature Drivers")
                if score_col and "feature" in fi_df.columns:
                    fi_df = fi_df.sort_values(score_col, ascending=False)
                    fig = px.bar(
                        fi_df.head(10),
                        x=score_col,
                        y="feature",
                        orientation="h",
                        title="Top features influencing prediction",
                        color=score_col,
                        color_continuous_scale="Blues",
                    )
                    fig.update_layout(yaxis=dict(autorange="reversed"), height=380, margin=dict(l=40, r=20, t=60, b=40))
                    st.plotly_chart(fig, use_container_width=True)

                    top_rows = fi_df.head(5).to_dict(orient="records")
                    st.markdown("#### Key Driver Points")
                    for idx, row in enumerate(top_rows, start=1):
                        score_val = row.get(score_col)
                        st.markdown(
                            f"- {idx}. {row.get('feature', 'feature')} | contribution_strength={_format_metric_value(score_val)}"
                        )
                else:
                    pretty_display(feature_importance)
            else:
                st.info("No global feature importance found for this target.")

            sample_explanations = payload.get("sample_explanations") or []
            if sample_explanations:
                st.markdown("#### Sample-level Explanation Highlights")
                for sample in sample_explanations[:3]:
                    row_id = sample.get("row", "N/A")
                    contributions = sample.get("feature_contributions") or {}
                    if isinstance(contributions, dict) and contributions:
                        ranked = sorted(contributions.items(), key=lambda item: abs(float(item[1])) if str(item[1]).replace('.', '', 1).replace('-', '', 1).isdigit() else 0, reverse=True)[:5]
                        st.markdown(f"- Row {row_id} top contribution factors:")
                        for feature_name, feature_value in ranked:
                            st.markdown(f"  - {feature_name}: {_format_signed_value(feature_value)}")
                    else:
                        st.markdown(f"- Row {row_id}: contribution details unavailable")

            lime_explanations = payload.get("lime_explanations") or []
            if lime_explanations:
                st.markdown("#### LIME Local Rules")
                for lime_item in lime_explanations[:3]:
                    st.markdown(f"- Row {lime_item.get('row', 'N/A')} predicted_label={lime_item.get('predicted_label', 'N/A')}")
                    for rule in lime_item.get("top_features", [])[:5]:
                        st.markdown(f"  - {rule.get('feature', 'feature')}: weight={_format_signed_value(rule.get('weight'))}")

            if payload.get("shap_error"):
                st.warning(f"SHAP note: {payload.get('shap_error')}")
            if payload.get("lime_error"):
                st.warning(f"LIME note: {payload.get('lime_error')}")


def derive_diabetes_status(output):
    risk_scoring = output.get("risk_scoring", {}) if isinstance(output, dict) else {}
    predictions = output.get("predictions", {}) if isinstance(output, dict) else {}

    mean_risk = risk_scoring.get("mean_risk_score", risk_scoring.get("average_risk"))
    if mean_risk is None:
        mean_risk = risk_scoring.get("high_risk_share")
        if isinstance(mean_risk, (int, float)):
            mean_risk = float(mean_risk) * 100.0

    high_risk_share = risk_scoring.get("high_risk_share")
    if isinstance(high_risk_share, (int, float)) and high_risk_share <= 1:
        high_risk_share_pct = float(high_risk_share) * 100.0
    elif isinstance(high_risk_share, (int, float)):
        high_risk_share_pct = float(high_risk_share)
    else:
        high_risk_share_pct = None

    detected_yes = None
    for _target, info in (predictions or {}).items():
        if isinstance(info, dict):
            preds = info.get("sample_predictions") or []
            if preds:
                positive_count = sum(1 for value in preds if str(value).strip().lower() in {"1", "true", "yes", "diabetes", "positive"})
                detected_yes = positive_count > 0
                break

    if isinstance(mean_risk, (int, float)):
        if mean_risk >= 70:
            chance_label = "High"
        elif mean_risk >= 35:
            chance_label = "Moderate"
        else:
            chance_label = "Low"
    else:
        chance_label = "Unknown"

    if detected_yes is True:
        detected_label = "Diabetes indicators found"
        severity = "high" if chance_label == "High" else "medium"
    elif detected_yes is False:
        detected_label = "No diabetes indicators found in prediction samples"
        severity = "low"
    else:
        detected_label = "Not enough signal to confirm diabetes detection"
        severity = "medium"

    return {
        "detected_label": detected_label,
        "chance_label": chance_label,
        "mean_risk": mean_risk,
        "high_risk_share_pct": high_risk_share_pct,
        "severity": severity,
    }


def _build_dataset_profile(df, max_features=20):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:max_features]
    categorical_cols = [
        col for col in df.columns
        if col not in numeric_cols and not pd.api.types.is_datetime64_any_dtype(df[col])
    ][:max_features]

    profile = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "numeric": {},
        "categorical": {},
    }

    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        non_null = series.dropna()
        if non_null.empty:
            continue
        profile["numeric"][col] = {
            "mean": float(non_null.mean()),
            "std": float(non_null.std(ddof=0)) if len(non_null) > 1 else 0.0,
            "p10": float(non_null.quantile(0.10)),
            "p90": float(non_null.quantile(0.90)),
            "missing_rate": float(series.isna().mean()),
        }

    for col in categorical_cols:
        series = df[col].astype(str).fillna("Unknown")
        if series.empty:
            continue
        counts = series.value_counts(normalize=True)
        top_value = str(counts.index[0]) if not counts.empty else "Unknown"
        top_share = float(counts.iloc[0]) if not counts.empty else 0.0
        profile["categorical"][col] = {
            "top_value": top_value,
            "top_share": top_share,
            "missing_rate": float(df[col].isna().mean()),
            "n_unique": int(df[col].nunique(dropna=True)),
        }

    return profile


def _compute_drift_summary(current_profile, baseline_profile):
    numeric_rows = []
    categorical_rows = []

    baseline_numeric = (baseline_profile or {}).get("numeric", {})
    current_numeric = (current_profile or {}).get("numeric", {})
    for col in sorted(set(baseline_numeric.keys()).intersection(current_numeric.keys())):
        base = baseline_numeric[col]
        cur = current_numeric[col]
        base_std = abs(float(base.get("std", 0.0)))
        denom = max(base_std, 1e-6)
        z_shift = abs(float(cur.get("mean", 0.0)) - float(base.get("mean", 0.0))) / denom
        numeric_rows.append(
            {
                "feature": col,
                "baseline_mean": round(float(base.get("mean", 0.0)), 4),
                "current_mean": round(float(cur.get("mean", 0.0)), 4),
                "standardized_shift": round(float(z_shift), 4),
            }
        )

    baseline_cat = (baseline_profile or {}).get("categorical", {})
    current_cat = (current_profile or {}).get("categorical", {})
    for col in sorted(set(baseline_cat.keys()).intersection(current_cat.keys())):
        base = baseline_cat[col]
        cur = current_cat[col]
        share_delta = abs(float(cur.get("top_share", 0.0)) - float(base.get("top_share", 0.0)))
        top_changed = str(cur.get("top_value", "")) != str(base.get("top_value", ""))
        categorical_rows.append(
            {
                "feature": col,
                "baseline_top": str(base.get("top_value", "")),
                "current_top": str(cur.get("top_value", "")),
                "top_share_delta": round(float(share_delta), 4),
                "top_changed": bool(top_changed),
            }
        )

    numeric_shift = np.mean([row["standardized_shift"] for row in numeric_rows]) if numeric_rows else 0.0
    categorical_shift = np.mean([
        row["top_share_delta"] + (0.25 if row["top_changed"] else 0.0)
        for row in categorical_rows
    ]) if categorical_rows else 0.0

    drift_score = float(numeric_shift * 0.7 + categorical_shift * 0.3)
    if drift_score >= 1.0:
        status = "High"
    elif drift_score >= 0.45:
        status = "Moderate"
    else:
        status = "Low"

    return {
        "status": status,
        "drift_score": round(drift_score, 4),
        "numeric_rows": sorted(numeric_rows, key=lambda item: item["standardized_shift"], reverse=True),
        "categorical_rows": sorted(categorical_rows, key=lambda item: item["top_share_delta"], reverse=True),
    }


def _build_report_drift_summary(df):
    current_profile = _build_dataset_profile(df)
    if not os.path.exists(DRIFT_BASELINE_PATH):
        return {}

    try:
        import json
        with open(DRIFT_BASELINE_PATH, "r", encoding="utf-8") as f:
            baseline_profile = json.load(f)
    except Exception:
        return {}

    drift = _compute_drift_summary(current_profile, baseline_profile)
    return {
        "status": drift.get("status"),
        "drift_score": drift.get("drift_score"),
        "top_numeric": (drift.get("numeric_rows") or [])[:10],
        "top_categorical": (drift.get("categorical_rows") or [])[:10],
    }


def display_confidence_calibration(output):
    st.subheader("📏 Confidence Calibration")

    training = (output.get("model_monitoring", {}) or {}).get("training", {}) or {}
    calibration = training.get("calibration") or {}

    if calibration.get("status") != "available":
        st.info("Calibration metrics are not available for this run.")
        return

    st.markdown(
        """
        <div style='border:1px solid #d1d5db;background:#f9fafb;padding:12px;border-radius:10px;margin-bottom:8px;'>
            <strong>Classic Reliability View</strong><br/>
            Lower ECE and Brier score indicate better-calibrated confidence estimates.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("ECE", calibration.get("ece", "N/A"))
    with col2:
        st.metric("Brier Score", calibration.get("brier_score", "N/A"))
    with col3:
        st.metric("Samples", calibration.get("n_samples", "N/A"))

    bin_rows = calibration.get("bins") or []
    if not bin_rows:
        return

    bin_df = pd.DataFrame(bin_rows)
    st.markdown("#### Reliability by Probability Bin")
    st.dataframe(bin_df, use_container_width=True)

    chart_df = bin_df.melt(
        id_vars=["bin"],
        value_vars=["predicted_rate", "observed_rate"],
        var_name="series",
        value_name="rate",
    )
    fig = px.line(
        chart_df,
        x="bin",
        y="rate",
        color="series",
        markers=True,
        title="Predicted vs Observed Event Rate",
        color_discrete_map={"predicted_rate": "#1d4ed8", "observed_rate": "#b91c1c"},
    )
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)


def display_drift_dashboard(df):
    st.subheader("🧭 Data Drift Dashboard")

    current_profile = _build_dataset_profile(df)
    baseline_profile = None

    if os.path.exists(DRIFT_BASELINE_PATH):
        try:
            import json
            with open(DRIFT_BASELINE_PATH, "r", encoding="utf-8") as f:
                baseline_profile = json.load(f)
        except Exception:
            baseline_profile = None

    if baseline_profile is None:
        try:
            import json
            with open(DRIFT_BASELINE_PATH, "w", encoding="utf-8") as f:
                json.dump(_json_safe(current_profile), f, indent=2)
            st.info("Drift baseline initialized from the current dataset. Run analysis again to compare drift.")
        except Exception as error:
            st.warning(f"Could not initialize drift baseline: {error}")
        return

    drift = _compute_drift_summary(current_profile, baseline_profile)
    status_color = {"Low": "#16a34a", "Moderate": "#d97706", "High": "#dc2626"}.get(drift.get("status"), "#374151")

    st.markdown(
        f"""
        <div style='border:1px solid #d1d5db;background:#ffffff;padding:12px;border-radius:10px;margin-bottom:8px;'>
            <strong style='color:{status_color};'>Drift Status: {drift.get('status')}</strong><br/>
            Composite Drift Score: {drift.get('drift_score')}
        </div>
        """,
        unsafe_allow_html=True,
    )

    numeric_rows = drift.get("numeric_rows") or []
    if numeric_rows:
        st.markdown("#### Top Numeric Shifts")
        st.dataframe(pd.DataFrame(numeric_rows[:10]), use_container_width=True)

    categorical_rows = drift.get("categorical_rows") or []
    if categorical_rows:
        st.markdown("#### Top Categorical Shifts")
        st.dataframe(pd.DataFrame(categorical_rows[:10]), use_container_width=True)

    if st.button("Set Current Dataset as Drift Baseline", key="update_drift_baseline"):
        try:
            import json
            with open(DRIFT_BASELINE_PATH, "w", encoding="utf-8") as f:
                json.dump(_json_safe(current_profile), f, indent=2)
            st.success("Drift baseline updated.")
        except Exception as error:
            st.warning(f"Failed to update drift baseline: {error}")


def _predict_probability(model, frame):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(frame)
        if getattr(proba, "ndim", 1) == 2 and proba.shape[1] > 1:
            return float(proba[0, 1])
        return float(np.asarray(proba).reshape(-1)[0])

    prediction = model.predict(frame)
    return float(np.clip(np.asarray(prediction).reshape(-1)[0], 0.0, 1.0))


def _build_report_what_if_summary(df, output):
    predictions = output.get("predictions", {}) if isinstance(output, dict) else {}
    candidates = [
        (target, info) for target, info in predictions.items()
        if isinstance(info, dict) and info.get("best_model_pipeline") is not None
    ]
    if not candidates:
        return {}

    target, info = candidates[0]
    model = info.get("best_model_pipeline")
    if model is None:
        return {}

    feature_df = df.drop(columns=[target], errors="ignore").copy()
    preprocess = model.named_steps.get("preprocess") if hasattr(model, "named_steps") else None
    expected_columns = getattr(preprocess, "feature_names_in_", None) if preprocess is not None else None
    if expected_columns is not None:
        feature_df = feature_df.reindex(columns=list(expected_columns), fill_value=np.nan)

    if feature_df.empty:
        return {}

    baseline_row = {}
    for col in feature_df.columns:
        series = feature_df[col]
        if pd.api.types.is_numeric_dtype(series):
            values = pd.to_numeric(series, errors="coerce").dropna()
            baseline_row[col] = float(values.median()) if not values.empty else 0.0
        else:
            values = series.dropna().astype(str)
            baseline_row[col] = str(values.mode().iloc[0]) if not values.empty else "Unknown"

    numeric_columns = [col for col in feature_df.columns if pd.api.types.is_numeric_dtype(feature_df[col])]
    priority_tokens = ["glucose", "hba1c", "a1c", "bmi", "age", "insulin", "pressure", "pregnan"]
    ordered = [col for col in numeric_columns if any(token in str(col).lower() for token in priority_tokens)]
    ordered += [col for col in numeric_columns if col not in ordered]
    selected = ordered[:4]
    if not selected:
        return {}

    scenario_row = baseline_row.copy()
    changed = []
    for col in selected:
        series = pd.to_numeric(feature_df[col], errors="coerce").dropna()
        if series.empty:
            continue
        upper = float(series.quantile(0.95))
        baseline_value = float(scenario_row[col])
        scenario_value = min(upper, baseline_value * 1.1)
        scenario_row[col] = float(scenario_value)
        changed.append(
            {
                "feature": col,
                "baseline": round(baseline_value, 4),
                "scenario": round(float(scenario_value), 4),
            }
        )

    try:
        baseline_frame = pd.DataFrame([baseline_row]).reindex(columns=feature_df.columns)
        scenario_frame = pd.DataFrame([scenario_row]).reindex(columns=feature_df.columns)
        baseline_prob = _predict_probability(model, baseline_frame)
        scenario_prob = _predict_probability(model, scenario_frame)
        return {
            "target": target,
            "baseline_risk_pct": round(float(baseline_prob) * 100.0, 2),
            "scenario_risk_pct": round(float(scenario_prob) * 100.0, 2),
            "risk_delta_pct": round((float(scenario_prob) - float(baseline_prob)) * 100.0, 2),
            "changed_features": changed,
        }
    except Exception:
        return {}


def display_what_if_simulator(df, output):
    st.subheader("🧪 What-If Risk Simulator")

    predictions = output.get("predictions", {}) if isinstance(output, dict) else {}
    model_targets = [
        target for target, info in predictions.items()
        if isinstance(info, dict) and info.get("best_model_pipeline") is not None
    ]

    if not model_targets:
        st.info("No reusable trained pipeline found for scenario simulation in this run.")
        return

    selected_target = st.selectbox("Simulation target", options=model_targets, key="what_if_target")
    model = predictions[selected_target].get("best_model_pipeline")
    if model is None:
        st.info("Selected target has no pipeline for simulation.")
        return

    feature_df = df.drop(columns=[selected_target], errors="ignore").copy()
    preprocess = model.named_steps.get("preprocess") if hasattr(model, "named_steps") else None
    expected_columns = getattr(preprocess, "feature_names_in_", None) if preprocess is not None else None
    if expected_columns is not None:
        feature_df = feature_df.reindex(columns=list(expected_columns), fill_value=np.nan)

    if feature_df.empty:
        st.info("No feature columns available for simulation.")
        return

    baseline_row = {}
    for col in feature_df.columns:
        series = feature_df[col]
        if pd.api.types.is_numeric_dtype(series):
            non_null = pd.to_numeric(series, errors="coerce").dropna()
            baseline_row[col] = float(non_null.median()) if not non_null.empty else 0.0
        else:
            mode_series = series.dropna().astype(str)
            baseline_row[col] = str(mode_series.mode().iloc[0]) if not mode_series.empty else "Unknown"

    numeric_columns = [col for col in feature_df.columns if pd.api.types.is_numeric_dtype(feature_df[col])]
    priority_tokens = ["glucose", "hba1c", "a1c", "bmi", "age", "insulin", "pressure", "pregnan"]
    priority_features = [
        col for col in numeric_columns
        if any(token in str(col).lower() for token in priority_tokens)
    ]
    selected_features = (priority_features + [c for c in numeric_columns if c not in priority_features])[:6]

    if not selected_features:
        st.info("No numeric features available for interactive simulation.")
        return

    scenario = baseline_row.copy()
    st.markdown("Adjust key drivers and simulate risk:")
    for col in selected_features:
        series = pd.to_numeric(feature_df[col], errors="coerce").dropna()
        if series.empty:
            continue

        low = float(series.quantile(0.05))
        high = float(series.quantile(0.95))
        default = float(baseline_row[col])

        if not np.isfinite(low) or not np.isfinite(high):
            continue
        if high <= low:
            high = low + 1.0
        default = min(max(default, low), high)

        scenario[col] = st.slider(
            f"{col}",
            min_value=float(low),
            max_value=float(high),
            value=float(default),
            key=f"whatif_{selected_target}_{col}",
        )

    baseline_frame = pd.DataFrame([baseline_row]).reindex(columns=feature_df.columns)
    scenario_frame = pd.DataFrame([scenario]).reindex(columns=feature_df.columns)

    try:
        baseline_prob = _predict_probability(model, baseline_frame)
        scenario_prob = _predict_probability(model, scenario_frame)
        delta = scenario_prob - baseline_prob

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Baseline Risk", f"{baseline_prob * 100:.2f}%")
        with col2:
            st.metric("Scenario Risk", f"{scenario_prob * 100:.2f}%")
        with col3:
            st.metric("Risk Change", f"{delta * 100:+.2f}%")
    except Exception as error:
        st.warning(f"Scenario simulation failed: {error}")

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
        
        missing_total = int(df.isnull().sum().sum())
        missing_pct = (missing_total / max(1, df.size)) * 100
        missing_col_count = int((df.isnull().sum() > 0).sum())

        st.divider()
        st.write("### Missing Value Summary")
        missing_summary = pd.DataFrame([
            {"Metric": "Total missing cells", "Value": missing_total},
            {"Metric": "Columns with missing values", "Value": missing_col_count},
            {"Metric": "Overall missing rate", "Value": f"{missing_pct:.2f}%"}
        ])
        st.dataframe(missing_summary, use_container_width=True)

        handling_explanation = "Missing values are excluded from numeric summary calculations by default (dropna behavior)."
        if st.session_state.get("autofix_summary"):
            handling_explanation = "Autofix mode is enabled; missing values may have been imputed or rows dropped based on null thresholds."
        st.info(handling_explanation)

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
    """Diabetes Overview Dashboard (replaces Profit & Loss view).

    Detects common diabetes-related columns and shows simple clinical KPIs.
    """
    st.subheader("🩺 Diabetes Overview Dashboard")

    # Common diabetes clinical columns to look for
    diabetes_cols = {
        "glucose": [c for c in df.columns if "glucose" in c.lower() or "gluc" in c.lower()],
        "bmi": [c for c in df.columns if "bmi" in c.lower() or "body mass" in c.lower()],
        "age": [c for c in df.columns if c.lower() == "age" or "age" in c.lower()],
        "insulin": [c for c in df.columns if "insulin" in c.lower()],
        "pregnancies": [c for c in df.columns if "preg" in c.lower() or "pregnancies" in c.lower()],
        "outcome": [c for c in df.columns if c.lower() in ("outcome", "diabetes", "diabetes_outcome", "target")]
    }

    found = {k: v for k, v in diabetes_cols.items() if v}

    if not found:
        st.info("Dataset does not appear to contain diabetes-relevant clinical columns (e.g. glucose, BMI, age, insulin). Upload a diabetes clinical dataset or continue with general analysis.")
        return

    # Build simple KPIs
    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    output = st.session_state.get("output") or {}
    risk_scoring = output.get("risk_scoring", {}) if isinstance(output, dict) else {}

    def _first_col(col_list):
        return col_list[0] if col_list else None

    def _to_percent(value):
        try:
            val = float(value)
            return val * 100.0 if val <= 1.0 else val
        except Exception:
            return None

    # Prevalence if an outcome/target exists
    outcome_col = _first_col(diabetes_cols.get("outcome", []))
    prevalence_proxy = None

    if isinstance(risk_scoring, dict):
        if risk_scoring.get("high_risk_share") is not None:
            prevalence_proxy = _to_percent(risk_scoring.get("high_risk_share"))
        elif risk_scoring.get("mean_risk_score") is not None:
            prevalence_proxy = _to_percent(risk_scoring.get("mean_risk_score"))
        elif risk_scoring.get("average_risk") is not None:
            prevalence_proxy = _to_percent(risk_scoring.get("average_risk"))

    if outcome_col and outcome_col in df.columns:
        try:
            prevalence = 100 * pd.to_numeric(df[outcome_col], errors="coerce").dropna().mean()
            col1.metric("Diabetes Prevalence", f"{prevalence:.1f}%")
        except Exception:
            if prevalence_proxy is not None:
                col1.metric("Diabetes Prevalence", f"{prevalence_proxy:.1f}%")
                col1.caption("Proxy estimate from AI risk scoring")
            else:
                col1.metric("Diabetes Prevalence", "N/A")
    else:
        if prevalence_proxy is not None:
            col1.metric("Diabetes Prevalence", f"{prevalence_proxy:.1f}%")
            col1.caption("Proxy estimate from AI risk scoring")
        else:
            col1.metric("Diabetes Prevalence", "N/A")

    # Average glucose
    glucose_col = _first_col(diabetes_cols.get("glucose", []))
    if glucose_col and glucose_col in df.columns:
        col2.metric("Avg Glucose", f"{df[glucose_col].dropna().mean():.1f}")
    else:
        col2.metric("Avg Glucose", "N/A")

    # Average BMI
    bmi_col = _first_col(diabetes_cols.get("bmi", []))
    if bmi_col and bmi_col in df.columns:
        col3.metric("Avg BMI", f"{df[bmi_col].dropna().mean():.1f}")
    else:
        col3.metric("Avg BMI", "N/A")

    # Median age
    age_col = _first_col(diabetes_cols.get("age", []))
    if age_col and age_col in df.columns:
        col4.metric("Median Age", f"{df[age_col].dropna().median():.0f}")
    else:
        col4.metric("Median Age", "N/A")

    st.divider()

    # Distribution charts for found columns
    st.write("### Clinical Distributions")
    for key in ("glucose", "bmi", "age", "insulin"):
        cols = diabetes_cols.get(key, [])
        if cols:
            c = cols[0]
            fig, ax = plt.subplots(figsize=(8, 3))
            try:
                df[c].dropna().hist(bins=30, ax=ax, color='skyblue', edgecolor='black')
                ax.set_title(f"Distribution of {c}")
                st.pyplot(fig)
            except Exception:
                continue

    # Show sample high-risk records if outcome exists
    if outcome_col and outcome_col in df.columns:
        st.write("### High-risk Sample Records")
        # assume higher value==positive class (1)
        try:
            high_risk = df[df[outcome_col].astype(float) == 1].head(10)
            if not high_risk.empty:
                st.dataframe(high_risk)
        except Exception:
            pass
# ----------------------------
# Industry Selection
# ----------------------------
industry_options = ["Healthcare"]
selected_industry = st.selectbox("Select Industry for Smart Insights (Industry Mode)", industry_options)
industry_value = selected_industry

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
    # Sidebar Filters (Dynamic for low-cardinality and categorical columns)
    # -----------------------------
    st.sidebar.title("Filter & Search")

    # Choose filterable columns: categorical OR low-cardinality numeric
    filterable_cols = []
    for c in df_filtered.columns:
        try:
            nunique = int(df_filtered[c].nunique(dropna=True))
        except Exception:
            nunique = 0
        if df_filtered[c].dtype.name in ("object", "category") or nunique <= 100:
            filterable_cols.append(c)

    # Build filters
    for col in filterable_cols:
        unique_vals = df_filtered[col].dropna().unique().tolist()
        # If numeric but many unique values, provide range slider
        if pd.api.types.is_numeric_dtype(df_filtered[col]) and len(unique_vals) > 10:
            try:
                lo = float(df_filtered[col].min())
                hi = float(df_filtered[col].max())
                rng = st.sidebar.slider(f"{col} range", min_value=lo, max_value=hi, value=(lo, hi))
                df_filtered = df_filtered[(df_filtered[col] >= rng[0]) & (df_filtered[col] <= rng[1])]
            except Exception:
                continue
        else:
            # use sorted unique values and display as strings to avoid widget type issues
            opts = sorted([str(x) for x in unique_vals])
            default = opts[:] if opts else []
            selected = st.sidebar.multiselect(f"Filter {col}", options=opts, default=default, key=f"filter_{col}")
            if selected:
                # compare as strings for safety
                df_filtered = df_filtered[df_filtered[col].astype(str).isin(selected)]

    # -----------------------------
    # Feature Engineering (Optional)
    # -----------------------------
    numeric_cols = df_filtered.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df_filtered.select_dtypes(exclude=np.number).columns.tolist()
    
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

# Autofix + Column Types (Safe)
# ----------------------------
if df is not None and not df.empty:
    autofix = st.checkbox("Enable Autofix Mode (auto-fill missing / remove constant columns)")
    
    if autofix:
        try:
            df, autofix_summary = apply_autofix(df)
            st.session_state.autofix_summary = autofix_summary
            st.success("Autofix applied successfully!")
            
            # Display Autofix Summary
            with st.expander("Autofix Details", expanded=False):
                missing = autofix_summary.get("missing", {}) or {}
                filled = missing.get("filled", []) or []
                dropped = missing.get("dropped", []) or []
                duplicates_removed = int((autofix_summary.get("duplicates", {}) or {}).get("duplicates_removed", 0))
                constant_removed = (autofix_summary.get("constant_columns", {}) or {}).get("constant_columns_removed", []) or []
                final_shape = (autofix_summary.get("final_shape", {}) or {})

                if filled:
                    st.write("**Filled Missing Values (<=20% nulls):**")
                    for item in filled:
                        st.write(f"  • {item.get('column')}: {item.get('nulls', 0)} nulls -> {item.get('action')}")

                if dropped:
                    st.write("**Dropped Columns with High Missingness (>20%):**")
                    for item in dropped:
                        st.write(f"  • {item.get('column')}: {item.get('nulls', 0)} nulls -> {item.get('action')}")

                st.write(f"**Duplicate rows removed:** {duplicates_removed}")

                if constant_removed:
                    st.write("**Constant columns removed:**")
                    for col_name in constant_removed:
                        st.write(f"  • {col_name}")
                else:
                    st.write("**Constant columns removed:** none")

                if final_shape:
                    st.write(f"**Final shape after Autofix:** {final_shape.get('rows', 'N/A')} rows x {final_shape.get('columns', 'N/A')} columns")

                if not filled and not dropped and duplicates_removed == 0 and not constant_removed:
                    st.info("Autofix ran, but no cleanup changes were needed for this dataset.")
        except Exception as e:
            st.error(f"Autofix failed: {e}")
    else:
        st.session_state.autofix_summary = None

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
    # Talk-to-Your-Data AI (use engine if available, otherwise use local fallback)
    # ----------------------------
    try:
        from engines.talk_to_data import talk_to_data_ai  # optional advanced engine
        use_talk_engine = True
    except Exception:
        talk_to_data_ai = None
        use_talk_engine = False

    st.subheader("💬 Talk to Your Data AI")
    user_question = st.text_input("Ask a question about your data (e.g., 'Top 5 Amount outliers')", key="talk_input")

    if st.button("Ask AI", key="talk_to_data_btn") and user_question:
        with st.spinner("🤖 Generating answer..."):
            if use_talk_engine and talk_to_data_ai:
                try:
                    output = talk_to_data_ai(df, query=user_question)
                except Exception as e:
                    output = {"answer": f"Engine failed: {e}", "details": {}}
            else:
                output = talk_to_data_fallback(df, user_question)

            st.session_state.output = st.session_state.output or {}
            # Attach the talk result into session output for traceability
            st.session_state.output["talk_to_data"] = output

            # Present answer
            answer_text = output.get("answer", "No answer returned") if isinstance(output, dict) else str(output)
            display_ai_answer(answer_text)

            # If there are details, show them below
            details = output.get("details") if isinstance(output, dict) else None
            if details:
                try:
                    pretty_display(details)
                except Exception:
                    st.write(details)
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
                autofix=True
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
        st.session_state.output = route_to_engines(df, column_types, autofix=autofix)
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
            json.dump(_json_safe(output.get("predictions", {})), f, indent=4)

        # Recommendations JSON
        with open("outputs/recommendations.json", "w") as f:
            json.dump(_json_safe(output.get("recommendations", {})), f, indent=4)

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

        # PDF generation is handled centrally in save_outputs(output).

    #st.success("✅ AI Analysis Complete! Files saved in outputs/")

    # ----------------------------
    # Display Outputs
    # ----------------------------
    output = st.session_state.output
    output["drift_summary"] = _build_report_drift_summary(df)
    output["what_if_summary"] = _build_report_what_if_summary(df, output)
    save_outputs(output)

    st.subheader("🧾 AI Analysis Summary")
    diabetes_targets = output.get("diabetes_targets") or []
    st.markdown(f"**Diabetes target columns detected:** {', '.join(diabetes_targets) if diabetes_targets else 'None explicitly detected.'}")
    st.markdown(f"**Report file:** {output.get('report_path') or 'outputs/report.pdf'}")
    st.markdown(f"**Graph folder:** {output.get('graph_folder') or GRAPH_FOLDER}")
    st.markdown(f"**Prediction models:** {len(output.get('predictions', {}))}")
    recommendation_count = sum(len(v) for v in output.get('recommendations', {}).values()) if isinstance(output.get('recommendations', {}), dict) else 0
    st.markdown(f"**Recommendations generated:** {recommendation_count}")
    decision_count = output.get('decisions', {}).get('decision_count', len(output.get('decisions', {}).get('decisions', [])))
    st.markdown(f"**Clinical decisions generated:** {decision_count}")
    st.markdown(f"**Selected model:** {output.get('model_monitoring', {}).get('training', {}).get('best_model', output.get('model_monitoring', {}).get('selected_model', 'N/A'))}")
    st.markdown(f"**Average risk score:** {output.get('risk_scoring', {}).get('mean_risk_score', output.get('risk_scoring', {}).get('average_risk', 'N/A'))}")
    st.markdown(f"**Target mode:** {output.get('diabetes_detection', {}).get('strategy', 'N/A')}")

    diabetes_status = derive_diabetes_status(output)
    status_color = {
        "high": "#ef4444",
        "medium": "#f59e0b",
        "low": "#10b981",
    }.get(diabetes_status.get("severity"), "#2563eb")
    status_sub = []
    if diabetes_status.get("mean_risk") is not None:
        status_sub.append(f"Mean risk score: {round(float(diabetes_status.get('mean_risk')), 2)}")
    if diabetes_status.get("high_risk_share_pct") is not None:
        status_sub.append(f"High risk share: {round(float(diabetes_status.get('high_risk_share_pct')), 2)}%")
    status_sub_text = " | ".join(status_sub) if status_sub else "Risk details unavailable"

    st.markdown(
        f"""
        <div style='border-radius:12px;padding:14px;border:1px solid {status_color};background:#ffffff;'>
            <div style='font-weight:700;color:{status_color};'>Diabetes Status</div>
            <div style='font-size:16px;font-weight:600;margin-top:4px;'>{diabetes_status.get('detected_label')}</div>
            <div style='margin-top:4px;'>Chance of getting diabetes: <strong>{diabetes_status.get('chance_label')}</strong></div>
            <div style='margin-top:4px;color:#6b7280;font-size:13px;'>{status_sub_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    display_confidence_calibration(output)
    display_drift_dashboard(df)
    display_what_if_simulator(df, output)

    auto_ai_answer = output.get("auto_ai_answer") if isinstance(output, dict) else None
    if isinstance(auto_ai_answer, dict) and auto_ai_answer.get("answer"):
        st.info(f"AI Answer: {auto_ai_answer.get('answer')}")

    sections = [
        ("🛠 Problem Discovery", "problem_discovery"),
        ("📌 Clinical Intelligence", "clinical_insights"),
        ("📊 Predictions", "predictions"),
        ("🧭 Explainability", "explanations"),
        ("🎯 Recommendations", "recommendations"),
        ("🧪 Self-Critic", "self_critic"),
        ("🧬 Feature Engineering", "feature_engineering"),
        ("📈 Risk Scoring", "risk_scoring"),
        ("🛰 Model Monitoring", "model_monitoring"),
        ("🔎 Diabetes Detection", "diabetes_detection"),
        ("🧠 Decision Intelligence", "decisions"),
        ("💡 Adaptive Insights", "adaptive_insights"),
        ("📘 KPI Summary", "kpi_summary"),
        ("📊 Quality Summary", "quality_summary"),
        ("📝 Insight Summary", "insight_summary"),
        ("📋 Overview Summary", "overview_summary")
    ]

    for title, key in sections:
        st.subheader(title)
        if key == "problem_discovery":
            display_issues(output.get(key) or {})
        elif key == "predictions":
            display_predictions_point_form(output.get(key) or {})
        elif key == "explanations":
            display_explainability_detailed(output.get(key) or {})
        elif key == "recommendations":
            display_recommendations_point_form(output.get(key) or {})
        else:
            pretty_display(output.get(key) or {})

    st.subheader("🩺 Diabetes Detection & Recommendations")
    if diabetes_targets:
        st.write("Detected diabetes-related target columns:")
        st.write(diabetes_targets)
    else:
        st.info("No explicit diabetes label target detected; the pipeline still analyzes feature risk signals.")

    recommendations = output.get("recommendations", {}) or {}
    display_recommendations_point_form(recommendations)

    st.subheader("🧠 Clinical Decision Intelligence")
    decisions = output.get("decisions") or {}

    # If the decision engine explicitly blocked decisions, show why
    if decisions.get("status") == "blocked":
        st.error("Clinical decisioning blocked by AI self-critic for safety reasons.")
        st.write("**Block reason:**", decisions.get("reason", "Not provided"))
        st.write("**Self-critic summary:**")
        pretty_display(output.get("self_critic") or {})

        # Also surface any risk flags so the user can act
        sc = output.get("self_critic") or {}
        if sc.get("risk_flags"):
            st.warning("Risk flags detected:")
            for rf in sc.get("risk_flags", []):
                st.write("-", rf)

        # Offer lightweight tentative suggestions derived from recommendations
        recs = output.get("recommendations") or {}
        if recs:
            st.info("Showing tentative (non-actionable) suggestions derived from recommendations:")
            for target, rec_list in recs.items():
                with st.expander(f"Tentative suggestions for {target} ({len(rec_list)})", expanded=False):
                    for r in rec_list:
                        st.write("-", r.get("recommendation") or r.get("text") or r)

    else:
        # Normal active decisioning
        decision_items = decisions.get("decisions") or []
        if decision_items:
            for decision in decision_items:
                with st.expander(f"{decision.get('decision', 'Decision')} — confidence {decision.get('confidence', 'N/A')}", expanded=False):
                    st.write("**Recommended action:**", decision.get("recommended_action", "N/A"))
                    st.write("**Reasoning:**")
                    st.write(decision.get("reasoning", []))
                    st.write("**Expected impact:**")
                    st.write(decision.get("expected_impact", {}))
        else:
            # No decisions present — show diagnostics and a gentle fallback
            st.info("No clinical decisions generated by the engine.")
            st.write("**Self-critic:**")
            pretty_display(output.get("self_critic") or {})

            # Fallback: create simple suggestions from recommendations so user can see actionable ideas
            recs = output.get("recommendations") or {}
            if recs:
                st.info("Fallback suggestions (derived from recommendations):")
                for target, rec_list in recs.items():
                    with st.expander(f"Fallback suggestions for {target} ({len(rec_list)})", expanded=False):
                        for r in rec_list:
                            text = r.get("recommendation") or r.get("text") or str(r)
                            st.write("-", text)
            else:
                st.write("No recommendations available to derive fallback suggestions.")

    # Graphs
    st.subheader("📈 Graphs")
    graph_folder = output.get("graph_folder") or GRAPH_FOLDER
    graph_paths = output.get("graphs") or []

    displayed = False
    if graph_paths:
        for g in graph_paths:
            gpath = g if os.path.isabs(g) else os.path.join(graph_folder, g)
            if os.path.exists(gpath):
                st.image(gpath, caption=os.path.basename(gpath), use_container_width=True)
                displayed = True

    if not displayed and os.path.exists(graph_folder):
        graphs = [f for f in os.listdir(graph_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if graphs:
            for g in sorted(graphs):
                st.image(os.path.join(graph_folder, g), caption=g, use_container_width=True)
            displayed = True

    if not displayed:
        st.warning("No graphs found in the output graph folder. Run analysis to generate visual graph files.")

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
        pretty_display(adaptive_insights)
        st.download_button(
            "Download Adaptive Insights JSON",
            data=pd.Series(adaptive_insights).to_json(),
            file_name="adaptive_insights.json",
            key="adaptive_insights_download"
        )
    else:
        st.info("No adaptive insights generated.")
