# setup_ai_project.py
import os
import json

# ---- Base project path ----
base_path = r"F:\ARTIFICIAL INTELLIGENCE"
project_name = "AI_Data_Analytics"
project_path = os.path.join(base_path, project_name)

# ---- Folders ----
folders = [
    "core",
    "engines",
    "data"
]

# ---- Create folders ----
os.makedirs(project_path, exist_ok=True)
for folder in folders:
    os.makedirs(os.path.join(project_path, folder), exist_ok=True)
    # Create empty __init__.py in core and engines
    if folder in ["core", "engines"]:
        open(os.path.join(project_path, folder, "__init__.py"), "w").close()

print(f"✅ Project folders created at {project_path}")

# ---- core/router.py ----
router_code = '''
from engines.nlp_engine import run_nlp
from engines.recommendation import run_recommendations
from engines.predictive_model import run_predictive_model

def route_to_engines(df, column_types):
    print("🚀 Routing data to engines...")
    tfidf_matrix = run_nlp(df, column_types)
    rec_output = run_recommendations(df)
    pred_output = run_predictive_model(df, tfidf_matrix, column_types)
    print("✅ AI run completed for all columns!")
    return tfidf_matrix, rec_output, pred_output
'''

with open(os.path.join(project_path, "core", "router.py"), "w") as f:
    f.write(router_code.strip())

# ---- engines/nlp_engine.py ----
nlp_code = '''
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\\s]", "", text)
    return text

def run_nlp(df, column_types):
    print("💬 Running NLP engine...")
    text_cols = [col for col, t in column_types.items() if t == "text"]
    print(f"📝 Text columns detected: {text_cols}")
    if not text_cols:
        print("⚠️ No text columns found. Skipping NLP.")
        return None
    combined_text = df[text_cols].astype(str).agg(" ".join, axis=1)
    cleaned_text = combined_text.apply(clean_text)
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    X = vectorizer.fit_transform(cleaned_text)
    print("✅ NLP Feature extraction complete")
    print(f"📐 Feature matrix shape: {X.shape}")
    print(f"🔑 Sample features: {vectorizer.get_feature_names_out()[:10]}")
    return X
'''

with open(os.path.join(project_path, "engines", "nlp_engine.py"), "w") as f:
    f.write(nlp_code.strip())

# ---- engines/recommendation.py ----
rec_code = '''
def run_recommendations(df):
    recommendations = {}
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        print("⚠️ No numeric columns found for recommendations.")
        return {}
    for col in numeric_cols:
        mean_val = df[col].mean()
        recommendations[col] = []
        for i, val in enumerate(df[col]):
            if val < mean_val * 0.8:
                recommendations[col].append({"row": i+1, "recommendation": "Needs support", "value": val})
            elif val > mean_val * 1.2:
                recommendations[col].append({"row": i+1, "recommendation": "Excelling", "value": val})
            else:
                recommendations[col].append({"row": i+1, "recommendation": "Average", "value": val})
    print("✅ Recommendations generated")
    return recommendations
'''

with open(os.path.join(project_path, "engines", "recommendation.py"), "w") as f:
    f.write(rec_code.strip())

# ---- engines/predictive_model.py ----
pred_code = '''
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack

def run_predictive_model(df, tfidf_matrix, column_types):
    results = {}
    numeric_cols = [col for col, t in column_types.items() if t == "numerical"]
    if not numeric_cols:
        print("⚠️ No numeric columns found for prediction.")
        return results
    for target_col in numeric_cols:
        mean_val = df[target_col].mean()
        def label(val):
            if val < mean_val * 0.8:
                return "Needs Support"
            elif val > mean_val * 1.2:
                return "Excelling"
            else:
                return "Average"
        df['performance'] = df[target_col].apply(label)
        feature_cols = [c for c in numeric_cols if c != target_col]
        X_numeric = df[feature_cols].values if feature_cols else None
        X = hstack([tfidf_matrix, X_numeric]) if X_numeric is not None else tfidf_matrix
        y = df['performance']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        results[target_col] = {"test_indices": list(range(len(preds))), "predictions": preds.tolist()}
    print("✅ Predictions generated")
    return results
'''

with open(os.path.join(project_path, "engines", "predictive_model.py"), "w") as f:
    f.write(pred_code.strip())

# ---- main.py ----
main_code = '''
import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from core.router import route_to_engines
import os
import json
import sys

sys.path.append(os.path.dirname(__file__))

def main():
    Tk().withdraw()
    file_path = askopenfilename(
        title="Select your CSV/XLSX dataset",
        filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")]
    )
    if not file_path:
        print("No file selected. Exiting...")
        return

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in [".xls", ".xlsx"]:
        df = pd.read_excel(file_path)
    else:
        print(f"❌ Unsupported file type: {ext}")
        return

    print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    column_types = {col: "text" if df[col].dtype == "object" else "numerical" for col in df.columns}
    tfidf_matrix, rec_output, pred_output = route_to_engines(df, column_types)

    with open("recommendations.json", "w") as f:
        json.dump(rec_output, f, indent=2)
    with open("predictions.json", "w") as f:
        json.dump(pred_output, f, indent=2)

    print("✅ Recommendations saved to 'recommendations.json'")
    print("✅ Predictions saved to 'predictions.json'")

if __name__ == "__main__":
    main()
'''

with open(os.path.join(project_path, "main.py"), "w") as f:
    f.write(main_code.strip())

print("✅ All project files created successfully!")
