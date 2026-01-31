def select_targets(df, column_types, max_targets=3):
    """
    Auto-selects best targets for prediction.
    Returns dictionary with 'numerical' and 'categorical' lists.
    """
    numerical_targets = []
    categorical_targets = []

    for col, t in column_types.items():
        if col not in df.columns:
            continue

        nunique = df[col].nunique(dropna=True)

        if t == "numerical" and nunique > 5:
            numerical_targets.append(col)

        if t == "text":
            if 2 <= nunique <= 20:  # small categories for classification
                categorical_targets.append(col)

    return {
        "numerical": numerical_targets[:max_targets],
        "categorical": categorical_targets[:max_targets]
    }
