# core/overview_engine.py
import pandas as pd

def dataset_overview(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(exclude='number').columns.tolist()
    
    overview = {
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Numeric Columns": numeric_cols,
        "Categorical Columns": categorical_cols
    }
    return overview
