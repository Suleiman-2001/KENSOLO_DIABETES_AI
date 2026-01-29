# utils/target_selector.py
def select_targets(df, column_types):
    """
    Generic target selector for any dataset.
    """
    targets = []
    for col, t in column_types.items():
        if t != "numerical":
            continue
        if col.lower().endswith("id"):
            continue
        if df[col].nunique() < 3:
            continue
        if df[col].isnull().mean() > 0.3:
            continue
        targets.append(col)
    return targets
