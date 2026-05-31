# engines/talk_to_data.py

import pandas as pd
import numpy as np
import re


# ----------------------------
# COLUMN MATCHING (IMPROVED)
# ----------------------------
def find_column(df, query):
    query = query.lower()

    # exact match
    for col in df.columns:
        if col.lower() == query:
            return col

    # partial match
    for col in df.columns:
        clean_col = col.lower().replace("_", "").replace(" ", "")
        clean_query = query.replace(" ", "")
        if clean_col in clean_query or clean_query in clean_col:
            return col

    return None


# ----------------------------
# MAIN ENGINE
# ----------------------------
def talk_to_data_ai(df: pd.DataFrame, query: str, output: dict = None):

    q = query.lower().strip()

    # ----------------------------
    # BASIC STRUCTURE
    # ----------------------------
    if any(x in q for x in ["rows", "row count"]):
        return {
            "type": "info",
            "result": {
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1])
            }
        }

    if "columns" in q:
        return {
            "type": "info",
            "result": list(df.columns)
        }

    if "summary" in q or "describe" in q:
        return {
            "type": "analysis",
            "result": df.describe(include="all").to_dict()
        }

    if "head" in q:
        return {
            "type": "preview",
            "result": df.head(5).to_dict()
        }

    # ----------------------------
    # DATA QUALITY
    # ----------------------------
    if "quality" in q or "good" in q:
        missing = df.isnull().mean().mean()
        duplicates = df.duplicated().sum()
        constant = len([c for c in df.columns if df[c].nunique() <= 1])

        return {
            "type": "quality",
            "result": {
                "missing_rate": round(float(missing), 4),
                "duplicates": int(duplicates),
                "constant_columns": int(constant)
            }
        }

    # ----------------------------
    # MISSING VALUES
    # ----------------------------
    if "missing" in q or "null" in q:
        missing = df.isnull().sum()
        missing = missing[missing > 0]

        return {
            "type": "missing",
            "result": missing.to_dict() if not missing.empty else "No missing values"
        }

    # ----------------------------
    # CORRELATION
    # ----------------------------
    if "correlation" in q:
        num = df.select_dtypes(include=np.number)
        if num.shape[1] < 2:
            return {"type": "error", "result": "Not enough numeric columns"}
        return {
            "type": "correlation",
            "result": num.corr().to_dict()
        }

    # ----------------------------
    # OUTLIERS (IQR METHOD - FIXED)
    # ----------------------------
    if "outlier" in q:
        numeric = df.select_dtypes(include=np.number)
        outliers = {}

        for col in numeric.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1

            mask = (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
            if mask.sum() > 0:
                outliers[col] = int(mask.sum())

        return {
            "type": "outliers",
            "result": outliers if outliers else "No outliers detected"
        }

    # ----------------------------
    # COLUMN INTELLIGENCE
    # ----------------------------
    col = find_column(df, q)
    if col:
        info = {
            "dtype": str(df[col].dtype),
            "missing": int(df[col].isnull().sum()),
            "unique": int(df[col].nunique())
        }

        if np.issubdtype(df[col].dtype, np.number):
            info.update({
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max())
            })

        return {
            "type": "column_insight",
            "column": col,
            "result": info
        }

    # ----------------------------
    # FALLBACK
    # ----------------------------
    return {
        "type": "fallback",
        "result": "Query not understood. Try: rows, columns, missing, summary, correlation, outliers, or column name"
    }