import os
import sys
import warnings
import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ----------------------------
# Suppress warnings
# ----------------------------
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=Warning)


def main():

    # ----------------------------
    # Multiprocessing setup
    # ----------------------------
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    import joblib
    joblib.parallel.DEFAULT_MP_CONTEXT = mp.get_context("spawn")

    # ----------------------------
    # Core imports
    # ----------------------------
    from core.router import route_to_engines
    from engines import export_engine

    root = Tk()
    root.withdraw()
    root.update_idletasks()

    # ----------------------------
    # 1️⃣ Select dataset
    # ----------------------------
    try:
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
    finally:
        try:
            root.destroy()
        except Exception:
            pass

    if not file_path:
        print("No file selected. Exiting...")
        return

    # ----------------------------
    # 2️⃣ Load dataset
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
            df = pd.read_csv(file_path, sep="\t")

        elif ext == ".json":
            df = pd.read_json(file_path)

        elif ext == ".parquet":
            df = pd.read_parquet(file_path)

        elif ext in [".db", ".sqlite"]:
            import sqlite3
            conn = sqlite3.connect(file_path)
            tables = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table';", conn
            )
            if len(tables) > 0:
                table_name = tables.iloc[0, 0]
                df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            else:
                print("No tables found in SQLite database")
                return
            conn.close()

        elif ext in [".h5", ".hdf5"]:
            df = pd.read_hdf(file_path)

        else:
            print("Unsupported file type. Exiting...")
            return

    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # ----------------------------
    # 3️⃣ Data type cleaning
    # ----------------------------
    def _coerce_df_types(df_in):
        df_out = df_in.copy()

        for col in df_out.columns:
            try:
                if "date" in col.lower() or "time" in col.lower():
                    parsed = pd.to_datetime(df_out[col], errors="coerce")
                    if parsed.notna().sum() > 0:
                        df_out[col] = parsed

                elif pd.api.types.is_object_dtype(df_out[col]):
                    coerced = pd.to_numeric(df_out[col], errors="coerce")
                    if coerced.notna().sum() / max(1, len(coerced)) > 0.5:
                        df_out[col] = coerced

            except Exception:
                continue

        return df_out

    df = _coerce_df_types(df)

    print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # ----------------------------
    # 4️⃣ Automatic dataset detection layer
    # ----------------------------
    target_candidates = [
        col for col in df.columns
        if df[col].nunique() < 20 and df[col].dtype != "object"
    ]

    selected_target = None
    if target_candidates:
        numeric_df = df.select_dtypes(include=["number"])
        corr_matrix = numeric_df.corr().abs() if not numeric_df.empty else pd.DataFrame()

        scored_candidates = []
        for candidate in target_candidates:
            if candidate in corr_matrix.columns:
                corr_values = corr_matrix[candidate].drop(labels=[candidate], errors="ignore").dropna()
                score = float(corr_values.mean()) if not corr_values.empty else 0.0
            else:
                score = 0.0
            scored_candidates.append((candidate, score))

        selected_target = max(scored_candidates, key=lambda x: x[1])[0]

    column_types = {}

    for col in df.columns:
        col_lower = col.lower()
        dtype = df[col].dtype
        nunique = df[col].nunique(dropna=True)

        if col_lower.endswith("id") or col_lower == "id":
            column_types[col] = "identifier"

        elif "image" in col_lower or col_lower.endswith("_path"):
            column_types[col] = "image"

        elif pd.api.types.is_datetime64_any_dtype(dtype):
            column_types[col] = "datetime"

        elif pd.api.types.is_numeric_dtype(dtype):
            column_types[col] = "categorical" if nunique < 20 else "numerical"

        elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
            column_types[col] = "categorical"

        else:
            column_types[col] = "text"

    print(f"🔎 Target candidates: {target_candidates}")
    if selected_target:
        print(f"🎯 Selected target candidate: {selected_target}")

    # ----------------------------
    # 5️⃣ Run AI engine
    # ----------------------------
    print("🚀 Running KENSOLO AI...")
    output = route_to_engines(df, column_types)

    # ----------------------------
    # 6️⃣ Display results
    # ----------------------------
    print("\n🛠 Problem Discovery:")
    for k, v in output.get("problem_discovery", {}).items():
        print(f"{k}: {v}")

    print("\n🧭 Task Detection:")
    for k, v in output.get("task_detection", {}).items():
        print(f"{k}: {v}")

    print("\n🧪 Unsupervised Learning:")
    for k, v in output.get("unsupervised_learning", {}).items():
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

    print("\n🔍 Explainability (SHAP/LIME):")
    for target, explanation in output.get("explanations", {}).items():
        if explanation.get("feature_importance"):
            top_features = ", ".join(item.get("feature", "") for item in explanation.get("feature_importance", [])[:5])
            print(f"- {target} (SHAP top features): {top_features}")
        if explanation.get("lime_explanations"):
            lime_features = explanation.get("lime_explanations", [])[0].get("top_features", [])
            lime_text = ", ".join(item.get("feature", "") for item in lime_features[:5])
            print(f"- {target} (LIME top features): {lime_text}")

    print("\n📘 Experiment Summary:")
    for k, v in output.get("experiment_summary", {}).items():
        print(f"- {k}: {v}")

    print("\n🏆 Model leaderboard:")
    for item in output.get("model_leaderboard", []):
        if item.get("status") == "success":
            cv_metric = item.get("cv_primary") if item.get("cv_primary") is not None else item.get("cv_accuracy")
            print(f"- {item.get('model')}: cv={cv_metric} | holdout_f1={item.get('holdout_f1')}")

    print("\n🧬 Feature Engineering:")
    for k, v in output.get("feature_engineering", {}).items():
        print(f"{k}: {v}")

    print("\n📈 Risk Scoring:")
    for k, v in output.get("risk_scoring", {}).items():
        print(f"{k}: {v}")

    print("\n🛰 Model Monitoring:")
    for k, v in output.get("model_monitoring", {}).items():
        print(f"{k}: {v}")

    print("\n🔎 Diabetes Detection:")
    for k, v in output.get("diabetes_detection", {}).items():
        print(f"{k}: {v}")

    print("\n🧠 Clinical Decisions:")
    for decision in output.get("decisions", {}).get("decisions", []):
        print(f"- {decision.get('decision')}: {decision.get('recommended_action', 'N/A')} (confidence={decision.get('confidence')})")

    print("\n📅 Longitudinal Forecasts:")
    for target, forecast_list in output.get("longitudinal_forecasts", {}).items():
        years_text = ", ".join(str(item.get("years")) for item in forecast_list if item.get("years") is not None)
        print(f"- {target}: forecast horizons (years) -> {years_text if years_text else 'N/A'}")

    print("\n🧩 Multimodal Layer:")
    multimodal = output.get("multimodal_layer", {})
    modalities = multimodal.get("modalities", {})
    print(f"- Architecture: {multimodal.get('architecture', 'N/A')}")
    print(f"- Tabular available: {modalities.get('tabular', {}).get('available', False)}")
    print(f"- EHR available: {modalities.get('ehr', {}).get('available', False)}")
    print(f"- Imaging available: {modalities.get('imaging', {}).get('available', False)}")

    # ----------------------------
    # 7️⃣ Export Excel
    # ----------------------------
    try:
        excel_path = export_engine.save_to_excel(output)
        print(f"\n💹 Excel report saved at: {excel_path}")
    except Exception as e:
        print(f"❌ Failed to export Excel report: {e}")

    # ----------------------------
    # 8️⃣ Final output
    # ----------------------------
    print("\n🚀 KENSOLO AI processing complete!")
    print(f"📄 PDF report saved at: {output.get('report_path')}")
    print(f"📈 Graphs saved in folder: {output.get('graph_folder')}")
    if output.get("risk_scoring"):
        print(f"🩺 Average diabetes risk score: {output.get('risk_scoring', {}).get('mean_risk_score', output.get('risk_scoring', {}).get('average_risk', 'N/A'))}")
    if output.get("model_monitoring"):
        print(f"🛰 Selected model: {output.get('model_monitoring', {}).get('training', {}).get('best_model', output.get('model_monitoring', {}).get('selected_model', 'N/A'))}")


if __name__ == "__main__":
    main()

    