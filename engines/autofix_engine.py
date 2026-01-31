# engines/autofix_engine.py
import pandas as pd

def autofix_missing(df):
    """Fill missing values automatically"""
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype != "object":
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna("N/A", inplace=True)
    return df

def autofix_duplicates(df):
    """Remove duplicate rows"""
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    if before - after > 0:
        print(f"🧹 Removed {before - after} duplicate rows")
    return df

def autofix_constant(df):
    """Remove constant columns"""
    const_cols = [c for c in df.columns if df[c].nunique() <= 1]
    df = df.drop(columns=const_cols)
    if const_cols:
        print(f"🧹 Dropped constant columns: {const_cols}")
    return df

def apply_autofix(df):
    """Apply all autofix steps"""
    df = autofix_missing(df)
    df = autofix_duplicates(df)
    df = autofix_constant(df)
    print("✅ Autofix applied to dataset")
    return df
