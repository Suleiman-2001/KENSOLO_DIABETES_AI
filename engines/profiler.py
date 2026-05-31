# engines/profiler_engine.py

import pandas as pd
import numpy as np


def profile_dataset(df: pd.DataFrame):
    """
    Dataset Profiler Engine

    Provides:
    - Structural summary
    - Column-level profiling
    - Data type breakdown
    - Memory usage estimation
    - Distribution signals
    - ML readiness indicators
    """

    if df is None or len(df) == 0:
        return {"error": "Empty dataset"}

    profile = {}

    # ----------------------------
    # 1. Basic structure
    # ----------------------------
    profile["shape"] = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1])
    }

    # ----------------------------
    # 2. Data types summary
    # ----------------------------
    dtype_counts = df.dtypes.value_counts().to_dict()

    profile["data_types"] = {
        str(k): int(v) for k, v in dtype_counts.items()
    }

    # ----------------------------
    # 3. Column-level profiling
    # ----------------------------
    columns_profile = {}

    for col in df.columns:
        col_data = df[col]

        col_info = {
            "dtype": str(col_data.dtype),
            "missing_count": int(col_data.isnull().sum()),
            "missing_percent": float(col_data.isnull().mean()),
            "unique_values": int(col_data.nunique(dropna=True)),
        }

        # Numeric profiling
        if pd.api.types.is_numeric_dtype(col_data):
            col_info.update({
                "mean": float(col_data.mean()),
                "std": float(col_data.std()),
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "zero_variance": bool(col_data.nunique() <= 1)
            })

        # Categorical profiling
        else:
            top_values = col_data.value_counts().head(3).to_dict()
            col_info.update({
                "top_values": top_values,
                "high_cardinality": bool(col_data.nunique() > 50)
            })

        columns_profile[col] = col_info

    profile["columns"] = columns_profile

    # ----------------------------
    # 4. Missing data overview
    # ----------------------------
    total_missing = df.isnull().sum().sum()
    total_cells = df.size

    profile["missing_data"] = {
        "total_missing": int(total_missing),
        "missing_rate": float(total_missing / total_cells)
    }

    # ----------------------------
    # 5. Memory usage
    # ----------------------------
    profile["memory_usage_mb"] = float(df.memory_usage(deep=True).sum() / (1024 * 1024))

    # ----------------------------
    # 6. ML readiness score
    # ----------------------------
    missing_penalty = profile["missing_data"]["missing_rate"] * 40

    constant_cols = sum(1 for c in df.columns if df[c].nunique() <= 1)
    constant_penalty = constant_cols * 5

    high_cardinality_cols = sum(
        1 for c in df.select_dtypes(include=["object"]).columns
        if df[c].nunique() > 50
    )
    cardinality_penalty = high_cardinality_cols * 3

    ml_score = 100 - (missing_penalty + constant_penalty + cardinality_penalty)
    ml_score = max(0, min(100, ml_score))

    profile["ml_readiness"] = {
        "score": round(ml_score, 2),
        "constant_columns": int(constant_cols),
        "high_cardinality_columns": int(high_cardinality_cols),
        "status": (
            "Ready" if ml_score >= 80 else
            "Moderate" if ml_score >= 60 else
            "Poor"
        )
    }

    # ----------------------------
    # 7. Dataset health classification
    # ----------------------------
    if profile["missing_data"]["missing_rate"] < 0.05 and ml_score >= 80:
        health = "Excellent"
    elif ml_score >= 60:
        health = "Good"
    else:
        health = "Poor"

    profile["dataset_health"] = health

    return profile