# engines/talk_to_data.py
import pandas as pd
import numpy as np
import re

def find_column_from_query(df, query):
    """
    Attempt to detect column names mentioned in the query.
    """
    query_lower = query.lower()
    for col in df.columns:
        if col.lower() in query_lower:
            return col
    return None


def talk_to_data_ai(df: pd.DataFrame, query: str):
    """
    Advanced Talk-to-Your-Data AI
    Natural-language friendly and business-aware.
    """
    query_lower = query.lower()

    # -----------------------------
    # BASIC DATASET INFO
    # -----------------------------
    if "how many rows" in query_lower or "row count" in query_lower:
        return f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns."

    if "columns" in query_lower:
        return list(df.columns)

    if "summary" in query_lower or "describe" in query_lower:
        return df.describe(include='all').to_dict()

    if "head" in query_lower:
        return df.head(5).to_dict()

    # -----------------------------
    # DATA QUALITY CHECK
    # -----------------------------
    if "good" in query_lower or "quality" in query_lower:
        total_missing = df.isnull().sum().sum()
        constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
        duplicate_rows = df.duplicated().sum()

        issues = []
        if total_missing > 0:
            issues.append(f"{total_missing} missing values")
        if constant_cols:
            issues.append(f"constant columns: {constant_cols}")
        if duplicate_rows > 0:
            issues.append(f"{duplicate_rows} duplicate rows")

        if not issues:
            return "✅ Dataset looks clean and well-structured."
        else:
            return f"⚠️ Dataset issues detected: {', '.join(issues)}"

    # -----------------------------
    # MISSING VALUES ANALYSIS
    # -----------------------------
    if "missing" in query_lower or "null" in query_lower:
        missing_counts = df.isnull().sum().sort_values(ascending=False)
        missing_cols = missing_counts[missing_counts > 0]
        if missing_cols.empty:
            return "✅ No missing values detected in any column."
        else:
            return missing_cols.to_dict()

    # -----------------------------
    # CONSTANT COLUMNS
    # -----------------------------
    if "constant" in query_lower:
        constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
        if constant_cols:
            return {"constant_columns": constant_cols}
        else:
            return "✅ No constant columns detected."

    # -----------------------------
    # OUTLIERS / EXTREME VALUES
    # -----------------------------
    if "outlier" in query_lower or "extreme" in query_lower:
        col = find_column_from_query(df, query)
        numeric_cols = df.select_dtypes(include=np.number).columns

        if col and col in numeric_cols:
            top_n = 5
            match = re.search(r'\d+', query)
            if match:
                top_n = int(match.group())
            values = df[col].sort_values(
                key=lambda x: abs(x - x.mean()), ascending=False
            ).head(top_n)
            return {col: values.tolist()}

        outliers = {}
        for col in numeric_cols:
            values = df[col].sort_values(
                key=lambda x: abs(x - x.mean()), ascending=False
            ).head(5)
            outliers[col] = values.tolist()
        return outliers

    # -----------------------------
    # MEAN / AVERAGE / SUM
    # -----------------------------
    if "average" in query_lower or "mean" in query_lower:
        col = find_column_from_query(df, query)
        if col and np.issubdtype(df[col].dtype, np.number):
            return f"Average of {col} is {df[col].mean():,.2f}"
        return df.select_dtypes(include=np.number).mean().to_dict()

    if "sum" in query_lower or "total" in query_lower:
        col = find_column_from_query(df, query)
        if col and np.issubdtype(df[col].dtype, np.number):
            return f"Total of {col} is {df[col].sum():,.2f}"
        return df.select_dtypes(include=np.number).sum().to_dict()

    # -----------------------------
    # UNIQUE VALUES
    # -----------------------------
    if "unique" in query_lower:
        col = find_column_from_query(df, query)
        if col:
            return df[col].unique().tolist()
        return {c: df[c].nunique() for c in df.columns}

    # -----------------------------
    # CORRELATION
    # -----------------------------
    if "correlation" in query_lower:
        numeric_df = df.select_dtypes(include=np.number)
        if numeric_df.shape[1] < 2:
            return "Not enough numeric columns to calculate correlation."
        return numeric_df.corr().to_dict()

    # -----------------------------
    # COLUMN-SPECIFIC INSIGHTS
    # -----------------------------
    col = find_column_from_query(df, query)
    if col:
        info = {
            "dtype": str(df[col].dtype),
            "missing_values": int(df[col].isnull().sum()),
            "unique_values": df[col].nunique()
        }
        if np.issubdtype(df[col].dtype, np.number):
            info.update({
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max())
            })
        return {col: info}

    # -----------------------------
    # FALLBACK
    # -----------------------------
    return (
        "🤖 I understood your query but could not match a specific rule. "
        "Try asking about: rows, columns, summary, quality, missing values, "
        "constant columns, outliers, average, total, unique values, correlation, or any specific column."
    )
