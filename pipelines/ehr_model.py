import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def build_ehr_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )


def train_ehr_model(df: pd.DataFrame, target: str = "medication_flag"):
    numeric_df = df.select_dtypes(include=["number"]).copy()
    if target in numeric_df.columns:
        x = numeric_df.drop(columns=[target])
        y = numeric_df[target]
    else:
        x = numeric_df
        y = pd.Series([0] * len(df))

    model = build_ehr_pipeline()
    model.fit(x, y)
    return model, list(x.columns)
