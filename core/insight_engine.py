# core/insight_engine.py
import pandas as pd

def auto_insights(df: pd.DataFrame):
    insights = []
    
    # Missing values
    total_missing = df.isnull().sum().sum()
    if total_missing == 0:
        insights.append("No missing values detected ✅")
    else:
        insights.append(f"{total_missing} missing values detected ⚠️")
    
    # Constant columns
    constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
    if constant_cols:
        insights.append(f"Constant columns: {constant_cols} ⚠️")
    
    # Top 3 categories for categorical numeric mix
    numeric_cols = df.select_dtypes(include='number').columns
    if not numeric_cols.empty:
        col = numeric_cols[0]
        top3 = df[col].nlargest(3).tolist()
        insights.append(f"Top 3 values for {col}: {top3}")
    
    # High variability
    for col in numeric_cols:
        if df[col].std() > df[col].mean() * 0.8:
            insights.append(f"High variability detected in {col} ⚠️")
    
    return insights
