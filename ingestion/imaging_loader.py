import os
import re
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image


DEFAULT_IMAGING_METADATA_FILES = [
    "retinopathy_metadata.csv",
    "retinopathy_labels.csv",
    "imaging_dataset.csv",
]


def _resolve_image_path(base_data_dir: str, raw_path: str) -> Optional[str]:
    if not raw_path:
        return None

    candidate = raw_path.strip()

    direct = os.path.join(base_data_dir, candidate)
    if os.path.exists(direct):
        return direct

    image_file = os.path.basename(candidate)
    for folder in ["images", "retina", "retinopathy", "fundus"]:
        joined = os.path.join(base_data_dir, folder, image_file)
        if os.path.exists(joined):
            return joined

    # Fallback: map zero-padded references (img001.png) to local names (img1.png).
    stem, ext = os.path.splitext(image_file)
    digits = re.findall(r"\d+", stem)
    if digits:
        numeric_id = str(int(digits[-1]))
        for folder in ["images", "retina", "retinopathy", "fundus"]:
            folder_path = os.path.join(base_data_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            for name in os.listdir(folder_path):
                name_stem, name_ext = os.path.splitext(name)
                if ext and name_ext.lower() != ext.lower():
                    continue

                name_digits = re.findall(r"\d+", name_stem)
                if name_digits and str(int(name_digits[-1])) == numeric_id:
                    matched = os.path.join(folder_path, name)
                    if os.path.exists(matched):
                        return matched

    return None


def _extract_image_features(image_path: str) -> Dict[str, float]:
    try:
        with Image.open(image_path) as img:
            gray = img.convert("L").resize((224, 224))
            arr = np.asarray(gray, dtype=np.float32) / 255.0

        grad_x = np.diff(arr, axis=1)
        grad_y = np.diff(arr, axis=0)
        edge_energy = float(np.mean(np.abs(grad_x)) + np.mean(np.abs(grad_y)))

        return {
            "img_mean_intensity": float(arr.mean()),
            "img_std_intensity": float(arr.std()),
            "img_p25_intensity": float(np.percentile(arr, 25)),
            "img_p75_intensity": float(np.percentile(arr, 75)),
            "img_edge_energy": edge_energy,
        }
    except Exception:
        return {}


def load_imaging_dataset(data_dir: str) -> Tuple[pd.DataFrame, Dict[str, object]]:
    metadata_path = None
    for name in DEFAULT_IMAGING_METADATA_FILES:
        candidate = os.path.join(data_dir, name)
        if os.path.exists(candidate):
            metadata_path = candidate
            break

    if metadata_path is None:
        return pd.DataFrame(), {
            "available": False,
            "rows": 0,
            "columns": 0,
            "metadata_file": None,
            "note": "No imaging metadata found. Supported names include retinopathy_metadata.csv and imaging_dataset.csv.",
        }

    try:
        meta = pd.read_csv(metadata_path)
    except Exception as e:
        return pd.DataFrame(), {
            "available": False,
            "rows": 0,
            "columns": 0,
            "metadata_file": os.path.basename(metadata_path),
            "error": str(e),
        }

    meta.columns = [str(c).strip().lower() for c in meta.columns]

    path_col = "image_path" if "image_path" in meta.columns else None
    if path_col is None:
        for alt in ["path", "filename", "file", "image"]:
            if alt in meta.columns:
                path_col = alt
                break

    if path_col is None:
        return pd.DataFrame(), {
            "available": False,
            "rows": int(len(meta)),
            "columns": int(meta.shape[1]),
            "metadata_file": os.path.basename(metadata_path),
            "note": "Imaging metadata found but image path column is missing.",
        }

    patient_col = "patient_id" if "patient_id" in meta.columns else None

    rows = []
    metadata_fallback_rows = []
    for _, row in meta.iterrows():
        resolved = _resolve_image_path(data_dir, str(row.get(path_col, "")))
        if not resolved:
            continue

        features = _extract_image_features(resolved)
        item = {"image_path_resolved": resolved}

        if features:
            item.update(features)
        else:
            # Keep metadata-derived signals when image decoding is unavailable.
            item["img_features_unavailable"] = 1.0

        if patient_col:
            item["patient_id"] = row.get(patient_col)

        if "retinopathy_grade" in meta.columns:
            item["retinopathy_grade"] = pd.to_numeric(row.get("retinopathy_grade"), errors="coerce")

        if features:
            rows.append(item)
        else:
            metadata_fallback_rows.append(item)

    if not rows and metadata_fallback_rows:
        rows = metadata_fallback_rows

    if not rows:
        return pd.DataFrame(), {
            "available": False,
            "rows": int(len(meta)),
            "columns": int(meta.shape[1]),
            "metadata_file": os.path.basename(metadata_path),
            "note": "No valid images were resolved from metadata.",
        }

    features_df = pd.DataFrame(rows)

    summary = {
        "available": True,
        "rows": int(len(features_df)),
        "columns": int(features_df.shape[1]),
        "metadata_file": os.path.basename(metadata_path),
        "patients": int(features_df["patient_id"].nunique()) if "patient_id" in features_df.columns else int(len(features_df)),
    }

    return features_df, summary


def build_imaging_patient_features(imaging_df: pd.DataFrame) -> pd.DataFrame:
    if imaging_df.empty:
        return pd.DataFrame()

    if "patient_id" not in imaging_df.columns:
        numeric_cols = imaging_df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return pd.DataFrame()

        agg = {f"imaging_{col}_mean": float(imaging_df[col].mean()) for col in numeric_cols}
        return pd.DataFrame([agg])

    numeric_cols = imaging_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "patient_id"]

    grouped = imaging_df.groupby("patient_id", dropna=False)[numeric_cols].mean().reset_index()
    rename_map = {col: f"imaging_{col}_mean" for col in numeric_cols}
    grouped = grouped.rename(columns=rename_map)
    return grouped
