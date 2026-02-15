# core/quality_engine.py
import pandas as pd

def data_quality_score(df: pd.DataFrame):
    total_missing = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()
    constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
    
    # Simple scoring system
    score = 100
    if df.size > 0:
        score -= (total_missing / df.size) * 40
    score -= min(duplicate_rows, 10) * 2
    score -= len(constant_cols) * 2
    score = max(0, int(score))
    
    if score >= 85:
        status = "High Quality ✅"
    elif score >= 60:
        status = "Moderate ⚠️"
    else:
        status = "Poor ❌"
    
    issues = {
        "Missing Values": total_missing,
        "Duplicate Rows": duplicate_rows,
        "Constant Columns": constant_cols,
        "Score": score,
        "Status": status
    }
    return issues
