import os
import re
import matplotlib
matplotlib.use("Agg")  # save files without GUI
import matplotlib.pyplot as plt
import pandas as pd

def sanitize_filename(name: str) -> str:
    name = str(name)
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = name.replace(" ", "_")
    return name

def generate_graphs(df: pd.DataFrame, folder_path=None):
    # ----------------------------
    # Default folder inside project
    # ----------------------------
    if folder_path is None:
        PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
        folder_path = os.path.join(PROJECT_DIR, "..", "outputs", "graphs")

    folder_path = os.path.abspath(folder_path)
    os.makedirs(folder_path, exist_ok=True)
    saved_files = []

    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) == 0:
        print("⚠️ No numeric columns found. No graphs generated.")
        return saved_files

    for col in numeric_cols:
        try:
            data = pd.to_numeric(df[col], errors='coerce').dropna()
            if data.empty:
                print(f"⚠️ Column '{col}' has no valid numeric data. Skipping.")
                continue

            plt.figure(figsize=(6,4))
            data.hist(bins=30)
            plt.title(f"{col} Distribution")
            plt.xlabel(col)
            plt.ylabel("Frequency")

            safe_col = sanitize_filename(col)
            file_path = os.path.join(folder_path, f"{safe_col}_hist.png")
            plt.savefig(file_path, bbox_inches="tight")
            plt.close()

            saved_files.append(file_path)
            print(f"📊 Graph saved: {file_path}")

        except Exception as e:
            print(f"⚠️ Error creating graph for column '{col}': {e}")

    print(f"✅ All valid graphs saved in folder: {folder_path}")
    return saved_files
