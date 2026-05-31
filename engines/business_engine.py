import pandas as pd


def run_business_intelligence(df):
    """Basic business intelligence fallback.

    Returns simple business insight metrics so the pipeline can continue
    when dedicated business engines are not present.
    """
    insights = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "numeric_summary": df.select_dtypes(include=["number"]).describe().to_dict(),
        "categorical_summary": {
            col: int(df[col].nunique(dropna=True))
            for col in df.select_dtypes(include=["object"]).columns
        }
    }
    return insights
