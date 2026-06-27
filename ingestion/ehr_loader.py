import os
from typing import Dict, List, Optional, Tuple

import pandas as pd


DEFAULT_EHR_FILE_CANDIDATES = [
    "mimic_iv_diabetes.csv",
    "mimic_iv_cohort.csv",
    "mimic_diabetes.csv",
    "mimiciv_diabetes.csv",
    "mimiciv_cohort.csv",
    "mimic_derived_diabetes.csv",
    "eicu_diabetes.csv",
    "eicu_cohort.csv",
    "eicu_derived_diabetes.csv",
    "ehr_dataset.csv",
]

EHR_COLUMN_ALIASES = {
    "patient_id": [
        "patient_id",
        "subject_id",
        "stay_id",
        "hadm_id",
        "patientunitstayid",
        "icustay_id",
        "unique_pid",
    ],
    "visit_date": [
        "visit_date",
        "charttime",
        "admittime",
        "dischtime",
        "intime",
        "outtime",
        "hospitaladmitoffset",
    ],
    "systolic_bp": [
        "systolic_bp",
        "sbp",
        "systolic",
        "sysbp",
        "nibp_systolic",
        "noninvasivesystolic",
        "systemicsystolic",
    ],
    "diastolic_bp": [
        "diastolic_bp",
        "dbp",
        "diastolic",
        "diasbp",
        "nibp_diastolic",
        "noninvasivediastolic",
        "systemicdiastolic",
    ],
    "hba1c": [
        "hba1c",
        "hb_a1c",
        "a1c",
        "glycohemoglobin",
        "hba1c_percent",
        "glycohemoglobin_a1c",
    ],
    "glucose": ["glucose", "blood_glucose", "glucose_mg_dl", "glucose_serum", "glucose_lab"],
    "medication_flag": [
        "medication_flag",
        "insulin_flag",
        "on_insulin",
        "medication",
        "treated",
        "insulin",
        "insulin_ordered",
        "antidiabetic_med_flag",
    ],
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def _resolve_column(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    columns = set(df.columns)
    for alias in aliases:
        alias_norm = alias.lower().strip()
        if alias_norm in columns:
            return alias_norm
    return None


def _standardize_ehr_columns(df: pd.DataFrame) -> pd.DataFrame:
    standardized = pd.DataFrame(index=df.index)

    for std_col, aliases in EHR_COLUMN_ALIASES.items():
        source_col = _resolve_column(df, aliases)
        if source_col is not None:
            standardized[std_col] = df[source_col]

    for col in standardized.columns:
        if col != "visit_date":
            standardized[col] = pd.to_numeric(standardized[col], errors="ignore")

    if "visit_date" in standardized.columns:
        standardized["visit_date"] = pd.to_datetime(standardized["visit_date"], errors="coerce")

    return standardized


def _discover_ehr_files(data_dir: str, preferred_files: Optional[List[str]] = None) -> List[str]:
    preferred = preferred_files or DEFAULT_EHR_FILE_CANDIDATES
    discovered = []

    for file_name in preferred:
        full_path = os.path.join(data_dir, file_name)
        if os.path.exists(full_path):
            discovered.append(full_path)

    return discovered


def load_ehr_datasets(data_dir: str, preferred_files: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict[str, object]]:
    file_paths = _discover_ehr_files(data_dir, preferred_files)

    if not file_paths:
        summary = {
            "available": False,
            "sources": [],
            "rows": 0,
            "columns": 0,
            "note": "No EHR sources discovered. Supported names include MIMIC-IV/eICU cohort files and ehr_dataset.csv.",
        }
        return pd.DataFrame(), summary

    frames = []
    source_names = []

    for file_path in file_paths:
        try:
            raw = pd.read_csv(file_path)
            raw = _normalize_columns(raw)
            standardized = _standardize_ehr_columns(raw)
            if standardized.empty:
                continue

            standardized["ehr_source_file"] = os.path.basename(file_path)
            frames.append(standardized)
            source_names.append(os.path.basename(file_path))
        except Exception:
            continue

    if not frames:
        summary = {
            "available": False,
            "sources": [],
            "rows": 0,
            "columns": 0,
            "note": "EHR files were found but could not be standardized to expected schema.",
        }
        return pd.DataFrame(), summary

    ehr_df = pd.concat(frames, ignore_index=True)

    summary = {
        "available": True,
        "sources": sorted(set(source_names)),
        "rows": int(len(ehr_df)),
        "columns": int(ehr_df.shape[1]),
        "patients": int(ehr_df["patient_id"].nunique()) if "patient_id" in ehr_df.columns else int(len(ehr_df)),
    }

    return ehr_df, summary


def build_ehr_patient_features(ehr_df: pd.DataFrame) -> pd.DataFrame:
    if ehr_df.empty:
        return pd.DataFrame()

    work = ehr_df.copy()
    numeric_cols = [c for c in work.columns if c not in {"patient_id", "visit_date", "ehr_source_file"}]

    for col in numeric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    if "patient_id" in work.columns:
        grouped = work.groupby("patient_id", dropna=False)[numeric_cols].agg(["mean", "max", "min"]) if numeric_cols else pd.DataFrame(index=work["patient_id"].dropna().unique())
        if isinstance(grouped.columns, pd.MultiIndex):
            grouped.columns = [f"ehr_{a}_{b}" for a, b in grouped.columns]
        grouped = grouped.reset_index()
        return grouped

    agg = {}
    for col in numeric_cols:
        series = work[col].dropna()
        agg[f"ehr_{col}_mean"] = float(series.mean()) if not series.empty else 0.0
        agg[f"ehr_{col}_max"] = float(series.max()) if not series.empty else 0.0
        agg[f"ehr_{col}_min"] = float(series.min()) if not series.empty else 0.0

    return pd.DataFrame([agg])
