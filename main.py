import os
import warnings
import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# ----------------------------
# Suppress unnecessary warnings
# ----------------------------
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=Warning)

def main():
    # ----------------------------
    # Setup multiprocessing and joblib
    # ----------------------------
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    import joblib
    joblib.parallel.DEFAULT_MP_CONTEXT = mp.get_context("spawn")

    # ----------------------------
    # Import router and export engine after spawn method
    # ----------------------------
    from core.router import route_to_engines
    from engines import export_engine

    Tk().withdraw()  # Hide Tkinter root window

    # ----------------------------
    # 1️⃣ Ask user for dataset
    # ----------------------------
    file_path = askopenfilename(
        title="Select your dataset",
        filetypes=[
            ("All Supported", "*.csv;*.xlsx;*.xls;*.json;*.parquet;*.tsv;*.ods;*.h5;*.hdf5;*.db;*.sqlite"),
            ("CSV files", "*.csv"),
            ("Excel files", "*.xlsx;*.xls;*.ods"),
            ("JSON", "*.json"),
            ("Parquet", "*.parquet"),
            ("TSV", "*.tsv"),
            ("HDF5", "*.h5;*.hdf5"),
            ("SQLite", "*.db;*.sqlite"),
            ("All files", "*.*")
        ]
    )
    if not file_path:
        print("No file selected. Exiting...")
        return

    # ----------------------------
    # 2️⃣ Load dataset (Multi-format support)
    # ----------------------------
    ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if ext == ".csv":
            try:
                df = pd.read_csv(file_path)
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding="ISO-8859-1")
        
        elif ext == ".xlsx":
            df = pd.read_excel(file_path, engine="openpyxl")
        
        elif ext == ".xls":
            df = pd.read_excel(file_path, engine="xlrd")
        
        elif ext == ".ods":
            df = pd.read_excel(file_path, engine="odf")
        
        elif ext == ".tsv":
            try:
                df = pd.read_csv(file_path, sep="\t")
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, sep="\t", encoding="ISO-8859-1")
        
        elif ext == ".json":
            df = pd.read_json(file_path)
        
        elif ext == ".parquet":
            df = pd.read_parquet(file_path)
        
        elif ext in [".db", ".sqlite"]:
            import sqlite3
            conn = sqlite3.connect(file_path)
            tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
            if len(tables) > 0:
                table_name = tables.iloc[0, 0]
                df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                conn.close()
            else:
                conn.close()
                print("No tables found in SQLite database")
                return
        
        elif ext in [".h5", ".hdf5"]:
            df = pd.read_hdf(file_path)
        
        else:
            print("Unsupported file type. Exiting...")
            return
    
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # ----------------------------
    # Coerce common dtypes to avoid pandas StringDtype issues
    # ----------------------------
    def _coerce_df_types(df_in):
        df_out = df_in.copy()
        for col in df_out.columns:
            try:
                if pd.api.types.is_string_dtype(df_out[col].dtype):
                    df_out[col] = df_out[col].astype(object)

                if "date" in col.lower() or "time" in col.lower():
                    parsed = pd.to_datetime(df_out[col], errors="coerce")
                    if parsed.notna().sum() > 0:
                        df_out[col] = parsed

                if pd.api.types.is_object_dtype(df_out[col].dtype):
                    coerced = pd.to_numeric(df_out[col], errors="coerce")
                    if coerced.notna().sum() / max(1, len(coerced)) > 0.5:
                        df_out[col] = coerced
            except Exception:
                continue
        return df_out

    df = _coerce_df_types(df)

    print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # ----------------------------
    # 3️⃣ Detect column types
    # ----------------------------
    column_types = {}
    for col in df.columns:
     col_lower = col.lower()
    dtype = df[col].dtype
    
    if col_lower.endswith("id") or col_lower == "id":
        column_types[col] = "identifier"
    elif "image" in col_lower or col_lower.endswith("_path"):
        column_types[col] = "image"
    elif pd.api.types.is_numeric_dtype(df[col]):
        column_types[col] = "numerical"
    elif pd.api.types.is_datetime64_any_dtype(df[col]):
        column_types[col] = "datetime"
    else:
        column_types[col] = "text"

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
    # 6️⃣ Auto-export to Excel
    # ----------------------------
    try:
        excel_path = export_engine.save_to_excel(output)
        print(f"\n💹 Excel report saved at: {excel_path}")
    except Exception as e:
        print(f"❌ Failed to export Excel report: {e}")

    # ----------------------------
    # 7️⃣ Final confirmation
    # ----------------------------
    print("\n🚀 KENSOLO AI processing complete!")
    print(f"📄 PDF report saved at: {output.get('report_path')}")
    print(f"📈 Graphs saved in folder: {output.get('graph_folder')}")

if __name__ == "__main__":
    main()