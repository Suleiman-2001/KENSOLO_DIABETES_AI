# engines/autofix_engine.py
import pandas as pd
import numpy as np

def autofix_missing(df, null_threshold=0.2):
    """
    Intelligent missing value handler.

    Strategy:
    - <= threshold → impute
    - > threshold → drop column (NOT rows, safer for ML stability)
    """

    df = df.copy()

    filled_cols = []
    dropped_cols = []

    for col in df.columns:
        null_count = df[col].isnull().sum()

        if null_count == 0:
            continue

        null_pct = null_count / len(df)

        # ----------------------------
        # IMPUTATION STRATEGY
        # ----------------------------
        if null_pct <= null_threshold:

            if pd.api.types.is_numeric_dtype(df[col]):
                fill_value = df[col].mean()
                df[col] = df[col].fillna(fill_value)

                filled_cols.append({
                    "column": col,
                    "nulls": int(null_count),
                    "action": f"filled_with_mean ({fill_value:.4f})"
                })

            else:
                df[col] = df[col].fillna("N/A")

                filled_cols.append({
                    "column": col,
                    "nulls": int(null_count),
                    "action": "filled_with_NA"
                })

        # ----------------------------
        # DROP COLUMN (safer than row drop)
        # ----------------------------
        else:
            df = df.drop(columns=[col])
            dropped_cols.append({
                "column": col,
                "nulls": int(null_count),
                "action": "dropped_column_due_to_high_missing"
            })

    return df, {
        "filled": filled_cols,
        "dropped": dropped_cols
    }


# ----------------------------
# DUPLICATES FIX
# ----------------------------
def autofix_duplicates(df):
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)

    return df, {
        "duplicates_removed": int(removed)
    }


# ----------------------------
# CONSTANT COLUMN FIX
# ----------------------------
def autofix_constant(df):
    const_cols = [c for c in df.columns if df[c].nunique() <= 1]

    df = df.drop(columns=const_cols)

    return df, {
        "constant_columns_removed": const_cols
    }


# ----------------------------
# MASTER PIPELINE
# ----------------------------
def apply_autofix(df):
    """
    Full automated data cleaning pipeline
    """

    df = df.copy()
    summary = {}

    # 1. Missing values
    df, miss = autofix_missing(df)
    summary["missing"] = miss

    # 2. Duplicates
    df, dup = autofix_duplicates(df)
    summary["duplicates"] = dup

    # 3. Constant columns
    df, const = autofix_constant(df)
    summary["constant_columns"] = const

    summary["final_shape"] = {
        "rows": df.shape[0],
        "columns": df.shape[1]
    }

    return df, summary