import pandas as pd


def run_business_intelligence(df):
    """Basic business intelligence fallback.

    Returns simple business insight metrics so the pipeline can continue
    when dedicated business engines are not present.
    """
    numeric_df = df.select_dtypes(include=["number"])
    insights = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "numeric_summary": numeric_df.describe().to_dict() if numeric_df.shape[1] > 0 else {},
        "categorical_summary": {
            col: int(df[col].nunique(dropna=True))
            for col in df.select_dtypes(include=["object"]).columns
        }
    }
    return insights
