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


def _pack_response(response_type, result, answer=None, details=None):
    payload = {
        "type": response_type,
        "result": result,
    }
    if answer is not None:
        payload["answer"] = answer
    if details is not None:
        payload["details"] = details
    return payload


def generate_auto_answer(df: pd.DataFrame):
    rows = int(df.shape[0])
    cols = int(df.shape[1])
    missing_rate = float(df.isnull().mean().mean()) if rows and cols else 0.0
    duplicate_rows = int(df.duplicated().sum())
    numeric_cols = int(df.select_dtypes(include=np.number).shape[1])

    candidate_targets = []
    for col in df.columns:
        series = df[col]
        if np.issubdtype(series.dtype, np.number) and 2 <= series.nunique(dropna=True) <= 10:
            candidate_targets.append(col)

    answer = (
        f"Analysis started. Dataset has {rows} rows and {cols} columns, "
        f"with {numeric_cols} numeric columns and {missing_rate:.2%} average missingness. "
        f"Found {duplicate_rows} duplicate rows. "
        f"Potential target columns: {', '.join(candidate_targets[:3]) if candidate_targets else 'none detected'}."
    )

    details = {
        "rows": rows,
        "columns": cols,
        "numeric_columns": numeric_cols,
        "missing_rate": round(missing_rate, 4),
        "duplicate_rows": duplicate_rows,
        "candidate_targets": candidate_targets,
    }

    return _pack_response("auto_summary", details, answer=answer, details=details)


# ----------------------------
# MAIN ENGINE
# ----------------------------
def talk_to_data_ai(df: pd.DataFrame, query: str, output: dict = None):

    q = (query or "").lower().strip()

    if q == "" or "auto summary" in q or "analysis start" in q:
        return generate_auto_answer(df)

    # ----------------------------
    # BASIC STRUCTURE
    # ----------------------------
    if any(x in q for x in ["rows", "row count"]):
        result = {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1])
        }
        return _pack_response("info", result, answer=f"The dataset has {result['rows']} rows and {result['columns']} columns.", details=result)

    if "columns" in q:
        result = list(df.columns)
        return _pack_response("info", result, answer=f"The dataset has {len(result)} columns.", details={"columns": result})

    if "summary" in q or "describe" in q:
        result = df.describe(include="all").to_dict()
        return _pack_response("analysis", result, answer="Generated a full statistical summary for all columns.", details=result)

    if "head" in q:
        result = df.head(5).to_dict()
        return _pack_response("preview", result, answer="Showing the first 5 rows.", details=result)

    # ----------------------------
    # DATA QUALITY
    # ----------------------------
    if "quality" in q or "good" in q:
        missing = df.isnull().mean().mean()
        duplicates = df.duplicated().sum()
        constant = len([c for c in df.columns if df[c].nunique() <= 1])

        result = {
            "missing_rate": round(float(missing), 4),
            "duplicates": int(duplicates),
            "constant_columns": int(constant)
        }
        return _pack_response("quality", result, answer="Computed data quality metrics.", details=result)

    # ----------------------------
    # MISSING VALUES
    # ----------------------------
    if "missing" in q or "null" in q:
        missing = df.isnull().sum()
        missing = missing[missing > 0]

        result = missing.to_dict() if not missing.empty else "No missing values"
        return _pack_response("missing", result, answer="Checked missing values.", details={"missing": result})

    # ----------------------------
    # CORRELATION
    # ----------------------------
    if "correlation" in q:
        num = df.select_dtypes(include=np.number)
        if num.shape[1] < 2:
            return _pack_response("error", "Not enough numeric columns", answer="Not enough numeric columns to compute correlations.", details={})
        result = num.corr().to_dict()
        return _pack_response("correlation", result, answer="Computed correlation matrix for numeric columns.", details=result)

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

        result = outliers if outliers else "No outliers detected"
        return _pack_response("outliers", result, answer="Outlier scan complete.", details={"outliers": result})

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

        payload = _pack_response("column_insight", info, answer=f"Generated profile for column '{col}'.", details=info)
        payload["column"] = col
        return payload

    # ----------------------------
    # FALLBACK
    # ----------------------------
    return _pack_response(
        "fallback",
        "Query not understood. Try: rows, columns, missing, summary, correlation, outliers, or column name",
        answer="I could not parse that question. Try rows, summary, missing, or correlation.",
        details={},
    )