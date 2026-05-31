import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def sanitize_filename(name: str) -> str:
    name = str(name)
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    return name.replace(" ", "_")


def generate_graphs(df: pd.DataFrame, folder_path=None):
    """
    Diabetes Graph Engine

    Generates:
    - Distributions (histograms)
    - Clinical risk visual signals (glucose, BMI, age, BP)
    - Correlation heatmap (important for diabetes patterns)
    """

    if folder_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        folder_path = os.path.join(base_dir, "..", "outputs", "graphs")

    os.makedirs(folder_path, exist_ok=True)
    saved_files = []

    numeric_cols = df.select_dtypes(include="number").columns

    if len(numeric_cols) == 0:
        print("⚠️ No numeric columns found.")
        return saved_files

    # ============================
    # 1️⃣ HISTOGRAMS (ALL NUMERIC)
    # ============================
    for col in numeric_cols:
        try:
            data = pd.to_numeric(df[col], errors="coerce").dropna()
            if data.empty:
                continue

            plt.figure(figsize=(6, 4))
            plt.hist(data, bins=30)

            plt.title(f"{col} Distribution")
            plt.xlabel(col)
            plt.ylabel("Frequency")

            file_path = os.path.join(folder_path, f"{sanitize_filename(col)}_hist.png")
            plt.savefig(file_path, bbox_inches="tight")
            plt.close()

            saved_files.append(file_path)

        except Exception:
            continue

    # ============================
    # 2️⃣ DIABETES CLINICAL RISK PLOTS
    # ============================
    clinical_map = {
        "glucose": 140,
        "bmi": 30,
        "bloodpressure": 140,
        "bp": 140,
        "age": 50
    }

    for col in numeric_cols:
        col_lower = col.lower()

        for key, threshold in clinical_map.items():
            if key in col_lower:

                try:
                    data = pd.to_numeric(df[col], errors="coerce").dropna()

                    plt.figure(figsize=(6, 4))
                    plt.hist(data, bins=30, alpha=0.7)

                    plt.axvline(threshold, color="red", linestyle="--")

                    plt.title(f"{col} Clinical Risk Distribution")
                    plt.xlabel(col)
                    plt.ylabel("Patients")

                    file_path = os.path.join(
                        folder_path,
                        f"{sanitize_filename(col)}_risk.png"
                    )

                    plt.savefig(file_path, bbox_inches="tight")
                    plt.close()

                    saved_files.append(file_path)

                except Exception:
                    continue

    # ============================
    # 3️⃣ CORRELATION HEATMAP (DIABETES INSIGHT CORE)
    # ============================
    try:
        if len(numeric_cols) > 1:
            plt.figure(figsize=(8, 6))

            corr = df[numeric_cols].corr()

            plt.imshow(corr, cmap="coolwarm", aspect="auto")
            plt.colorbar()

            plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90)
            plt.yticks(range(len(numeric_cols)), numeric_cols)

            plt.title("Feature Correlation Heatmap")

            file_path = os.path.join(folder_path, "correlation_heatmap.png")
            plt.savefig(file_path, bbox_inches="tight")
            plt.close()

            saved_files.append(file_path)

    except Exception:
        pass

    print(f"✅ Graphs generated: {len(saved_files)} files")

    return saved_files