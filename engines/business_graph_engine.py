import os


def generate_business_graphs(df, business_insights, folder="outputs/graphs"):
    """Fallback business graph generator.

    Returns an empty list if no graphs are generated, but ensures the
    business pipeline stays intact.
    """
    os.makedirs(folder, exist_ok=True)
    return []
