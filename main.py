import os
import warnings
import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from core.router import route_to_engines

# ----------------------------
# Suppress unnecessary warnings
# ----------------------------
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=Warning)

def main():
    Tk().withdraw()  # Hide Tkinter root window

    # ----------------------------
    # 1️⃣ Ask user for dataset
    # ----------------------------
    file_path = askopenfilename(
        title="Select your dataset (CSV/XLSX)",
        filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")]
    )
    if not file_path:
        print("❌ No file selected. Exiting...")
        return

    # ----------------------------
    # 2️⃣ Load dataset
    # ----------------------------
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext == ".xlsx":
        df = pd.read_excel(file_path)
    else:
        print("❌ Unsupported file type. Exiting...")
        return

    print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # ----------------------------
    # 3️⃣ Detect column types
    # ----------------------------
    column_types = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower.endswith("id") or col_lower == "id":
            column_types[col] = "identifier"
        elif "image" in col_lower or col_lower.endswith("_path"):
            column_types[col] = "image"
        elif df[col].dtype == "object":
            column_types[col] = "text"
        else:
            column_types[col] = "numerical"

    # ----------------------------
    # 4️⃣ Run KENSOLO AI (router)
    # ----------------------------
    print("🚀 Running KENSOLO AI...")
    output = route_to_engines(df, column_types)

    # ----------------------------
    # 5️⃣ Display outputs
    # ----------------------------
    print("\n🛠 Problem Discovery:")
    for k, v in output.get("problem_discovery", {}).items():
        print(f"{k}: {v}")

    print("\n📊 Predictions:")
    for k, v in output.get("predictions", {}).items():
        print(f"{k}: {v}")

    print("\n🎯 Recommendations:")
    for k, v in output.get("recommendations", {}).items():
        print(f"{k}: {v}")

    print("\n🧪 Self Critic:")
    for k, v in output.get("self_critic", {}).items():
        print(f"{k}: {v}")

    # ----------------------------
    # 6️⃣ Final confirmation
    # ----------------------------
    print("\n🚀 KENSOLO AI processing complete!")
    print(f"📄 PDF report saved at: {output.get('report_path')}")
    print(f"📈 Graphs saved in folder: {output.get('graph_folder')}")

if __name__ == "__main__":
    main()
