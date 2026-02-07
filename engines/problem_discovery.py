# engines/problem_discovery.py
import pandas as pd
import numpy as np

def discover_problem(df):
    """
    Detect potential data issues in a DataFrame.
    Returns a structured dictionary suitable for JSON/CSV export
    and compatible with Streamlit .items() calls.
    """

    issues = []

    # ----------------------------
    # 1️⃣ Missing values
    # ----------------------------
    for col in df.columns:
        missing = df[col].isnull().sum()
        if missing > 0:
            issues.append({
                "column": col,
                "issue_type": "Missing Values",
                "details": f"{missing} missing values",
                "severity": "High" if missing / len(df) > 0.2 else "Medium"
            })

    # ----------------------------
    # 2️⃣ Constant columns
    # ----------------------------
    for col in df.columns:
        if df[col].nunique() <= 1:
            issues.append({
                "column": col,
                "issue_type": "Constant Column",
                "details": "All values are constant",
                "severity": "Medium"
            })

    # ----------------------------
    # 3️⃣ Duplicate rows
    # ----------------------------
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        issues.append({
            "column": "All",
            "issue_type": "Duplicate Rows",
            "details": f"{duplicates} duplicate rows",
            "severity": "High"
        })

    # ----------------------------
    # 4️⃣ Outliers in numeric columns (IQR method)
    # ----------------------------
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers_count = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)].shape[0]
        if outliers_count > 0:
            issues.append({
                "column": col,
                "issue_type": "Outliers",
                "details": f"{outliers_count} potential outliers detected",
                "severity": "Medium" if outliers_count / len(df) < 0.05 else "High"
            })

    # ----------------------------
    # 5️⃣ High cardinality in categorical columns
    # ----------------------------
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        unique_count = df[col].nunique()
        if unique_count > 50:  # threshold can be adjusted
            issues.append({
                "column": col,
                "issue_type": "High Cardinality",
                "details": f"{unique_count} unique values",
                "severity": "Medium"
            })

    # ----------------------------
    # 6️⃣ Potential date columns not recognized
    # ----------------------------
    for col in df.columns:
        if "date" not in col.lower() and df[col].dtype == "object":
            try:
                pd.to_datetime(df[col], errors='raise')
                issues.append({
                    "column": col,
                    "issue_type": "Potential Date Column",
                    "details": "Column may contain dates",
                    "severity": "Low"
                })
            except:
                continue

    # ----------------------------
    # Convert list of issues into a dictionary with unique keys
    # ----------------------------
    if not issues:
        return {"status": {"column": "All", "issue_type": "No issues detected", "details": "", "severity": "None"}}

    issues_dict = {}
    for idx, issue in enumerate(issues, start=1):
        key = f"issue_{idx}"
        issues_dict[key] = issue

    return issues_dict
