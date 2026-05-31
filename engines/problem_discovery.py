# engines/problem_discovery.py

import pandas as pd
import numpy as np


def discover_problem(df: pd.DataFrame):
    """
    Data Quality & Problem Discovery Engine

    Detects:
    - Missing values
    - Constant columns
    - Duplicates
    - Outliers (IQR)
    - High cardinality
    - Potential mis-typed columns
    """

    issues = []
    n_rows = len(df)

    if n_rows == 0:
        return {"status": {"issue_type": "Empty Dataset", "severity": "Critical"}}

    # ----------------------------
    # 1. Missing Values
    # ----------------------------
    for col in df.columns:
        missing = df[col].isnull().sum()

        if missing > 0:
            issues.append({
                "column": col,
                "issue_type": "Missing Values",
                "details": f"{missing} missing values ({round(missing/n_rows*100, 2)}%)",
                "severity": "High" if missing / n_rows > 0.2 else "Medium"
            })

    # ----------------------------
    # 2. Constant Columns
    # ----------------------------
    for col in df.columns:
        if df[col].nunique(dropna=False) <= 1:
            issues.append({
                "column": col,
                "issue_type": "Constant Column",
                "details": "No variance detected",
                "severity": "Medium"
            })

    # ----------------------------
    # 3. Duplicate Rows
    # ----------------------------
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        issues.append({
            "column": "All",
            "issue_type": "Duplicate Rows",
            "details": f"{duplicates} duplicates ({round(duplicates/n_rows*100, 2)}%)",
            "severity": "High" if duplicates / n_rows > 0.1 else "Medium"
        })

    # ----------------------------
    # 4. Outliers (IQR method)
    # ----------------------------
    num_cols = df.select_dtypes(include=[np.number]).columns

    for col in num_cols:
        try:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            if IQR == 0:
                continue

            outliers = df[
                (df[col] < Q1 - 1.5 * IQR) |
                (df[col] > Q3 + 1.5 * IQR)
            ]

            count = len(outliers)

            if count > 0:
                issues.append({
                    "column": col,
                    "issue_type": "Outliers",
                    "details": f"{count} outliers ({round(count/n_rows*100, 2)}%)",
                    "severity": "High" if count / n_rows > 0.05 else "Medium"
                })

        except Exception:
            continue

    # ----------------------------
    # 5. High Cardinality (categorical)
    # ----------------------------
    cat_cols = df.select_dtypes(include=["object"]).columns

    for col in cat_cols:
        unique_ratio = df[col].nunique(dropna=True) / max(1, n_rows)

        if unique_ratio > 0.5 and df[col].nunique() > 50:
            issues.append({
                "column": col,
                "issue_type": "High Cardinality",
                "details": f"{df[col].nunique()} unique values",
                "severity": "Medium"
            })

    # ----------------------------
    # 6. Mis-typed numeric stored as text
    # ----------------------------
    for col in cat_cols:
        sample = df[col].dropna().head(50)

        if len(sample) > 0:
            numeric_like = pd.to_numeric(sample, errors="coerce").notna().mean()

            if numeric_like > 0.8:
                issues.append({
                    "column": col,
                    "issue_type": "Numeric Stored as Text",
                    "details": "Column appears numeric but stored as object",
                    "severity": "High"
                })

    # ----------------------------
    # Final Output Formatting
    # ----------------------------
    if not issues:
        return {
            "status": {
                "column": "All",
                "issue_type": "No issues detected",
                "details": "Dataset is clean",
                "severity": "None"
            }
        }

    return {
        f"issue_{i+1}": issue
        for i, issue in enumerate(issues)
    }