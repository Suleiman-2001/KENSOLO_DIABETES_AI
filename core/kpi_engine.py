# core/kpi_engine.py
import pandas as pd

def detect_kpis(df: pd.DataFrame):
    kpi_keywords = ["amount","revenue","sales","cost","price","profit","quantity"]
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    
    kpi_candidates = [col for col in numeric_cols if any(k in col.lower() for k in kpi_keywords)]
    if not kpi_candidates and numeric_cols:
        kpi_candidates = [numeric_cols[0]]  # fallback
    
    kpi_summary = {}
    for col in kpi_candidates:
        kpi_summary[col] = {
            "Total": df[col].sum(),
            "Average": df[col].mean(),
            "Max": df[col].max(),
            "Min": df[col].min()
        }
    return kpi_summary
