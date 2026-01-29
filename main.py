# main.py
import os
import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from core.router import route_to_engines

def main():
    Tk().withdraw()

    file_path = askopenfilename(
        title="Select your dataset (CSV/XLSX)",
        filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")]
    )
    if not file_path:
        print("❌ No file selected. Exiting...")
        return

    ext = os.path.splitext(file_path)[1].lower()
    df = pd.read_csv(file_path) if ext == ".csv" else pd.read_excel(file_path)

    print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    column_types = {}
    for col in df.columns:
        if df[col].dtype == "object":
            if "image" in col.lower() or col.endswith("_path"):
                column_types[col] = "image"
            else:
                column_types[col] = "text"
        else:
            column_types[col] = "numerical"

    output = route_to_engines(df, column_types)

    print("\n🛠 Problem Discovery:")
    print(output["problem_discovery"])

    print("\n📊 Predictions:")
    print(output["predictions"])

    print("\n🎯 Recommendations:")
    print(output["recommendations"])

    print("\n🧪 Self Critic:")
    print(output["self_critic"])

if __name__ == "__main__":
    main()
