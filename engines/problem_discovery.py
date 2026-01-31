# engines/problem_discovery.py
import pandas as pd

def discover_problem(df):
    issues = {}

    # ----------------------------
    # 1️⃣ Missing values
    # ----------------------------
    for col in df.columns:
        missing = df[col].isnull().sum()
        if missing > 0:
            issues[col] = f"{missing} missing values"

    # ----------------------------
    # 2️⃣ Constant columns
    # ----------------------------
    for col in df.columns:
        if df[col].nunique() <= 1:
            issues[col] = "Constant column"

    # ----------------------------
    # 3️⃣ Duplicate rows
    # ----------------------------
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        issues["duplicates"] = f"{duplicates} duplicate rows"

    # ----------------------------
    # 4️⃣ Outliers in numeric columns (IQR method)
    # ----------------------------
    num_cols = df.select_dtypes(include=['number']).columns
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)].shape[0]
        if outliers > 0:
            issues[col] = f"{outliers} potential outliers detected"

    # ----------------------------
    # 5️⃣ High cardinality in categorical columns
    # ----------------------------
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].nunique() > 50:  # threshold can be adjusted
            issues[col] = f"High cardinality: {df[col].nunique()} unique values"

    # ----------------------------
    # 6️⃣ Potential date columns not recognized
    # ----------------------------
    for col in df.columns:
        if col.lower().count("date") == 0 and df[col].dtype == "object":
            try:
                pd.to_datetime(df[col], errors='raise')
                issues[col] = "Potential date column"
            except:
                continue

    return issues if issues else {"status": "No issues detected"}
