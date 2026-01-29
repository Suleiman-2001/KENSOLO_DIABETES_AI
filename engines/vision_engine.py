# engines/vision_engine.py
import matplotlib.pyplot as plt
import os

def generate_graphs(df, targets=None, folder="outputs/graphs"):
    """
    Generate histograms for numeric columns, bar charts for categorical columns.
    Always creates folder even if no targets.
    """
    os.makedirs(folder, exist_ok=True)  # create folder if missing

    if targets is None:
        targets = df.columns.tolist()  # include all columns

    for col in targets:
        if col not in df.columns:
            continue

        plt.figure()
        if df[col].dtype != "O":  # numeric
            df[col].hist()
            plt.ylabel("Count")
        else:  # categorical
            df[col].value_counts().plot.bar()
            plt.ylabel("Frequency")
        plt.title(f"Distribution of {col}")
        safe_name = col.replace("/", "_").replace("\\", "_")
        plt.savefig(f"{folder}/{safe_name}.png")
        plt.close()
