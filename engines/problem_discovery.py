# engines/problem_discovery.py
def discover_problem(df):
    """
    Detect missing values and data issues in a dataframe
    """
    issues = {}
    for col in df.columns:
        missing = df[col].isnull().sum()
        if missing > 0:
            issues[col] = f"{missing} missing values"
    return issues if issues else {"status": "No issues detected"}
