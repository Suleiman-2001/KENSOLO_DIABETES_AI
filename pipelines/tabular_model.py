import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def build_tabular_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )


def train_tabular_model(df: pd.DataFrame, target: str = "outcome"):
    features = [c for c in df.columns if c != target]
    x = df[features].select_dtypes(include=["number"]).copy()
    y = df[target]

    model = build_tabular_pipeline()
    model.fit(x, y)
    return model, features
