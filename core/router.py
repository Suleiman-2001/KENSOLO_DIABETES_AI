# core/router.py
import os
import copy

from engines.vision_engine import generate_graphs, save_predictions_and_recommendations
from engines.nlp_engine import run_nlp_analysis
from engines.predictive_engine import run_predictive_model
from engines.problem_discovery import discover_problem
from engines.self_critic import self_critic

from engines.business_engine import run_business_intelligence
from engines.business_graph_engine import generate_business_graphs
from engines.autofix_engine import apply_autofix
from engines.why_engine import explain_predictions  # ✅ Why Engine

# Output folders
BASE_OUTPUT = r"F:\ARTIFICIAL INTELLIGENCE\AI_Data_Analytics\outputs"
GRAPH_FOLDER = os.path.join(BASE_OUTPUT, "graphs")


def route_to_engines(df, column_types, autofix=True):
    """
    Routes dataset through KENSOLO AI engines:
    - Optional Autofix
    - Detects problems
    - Runs NLP analysis
    - Runs predictive models
    - Generates graphs (basic + business)
    - Generates explanations for predictions (Why Engine)
    - Saves predictions + recommendations
    - Generates business insights
    - Self-critic trust scoring
    """

    # ----------------------------
    # 0️⃣ Apply Autofix (optional)
    # ----------------------------
    fixes_summary = {}
    if autofix:
        try:
            df, fixes_summary = apply_autofix(df)
            print(f"🛠 Autofix applied: {fixes_summary}")
        except Exception as e:
            print(f"⚠️ Autofix failed: {e}")

        # Update column types after autofix
        column_types = {col: ("text" if df[col].dtype == "object" else "numerical")
                        for col in df.columns}

    # ----------------------------
    # 1️⃣ Problem discovery
    # ----------------------------
    problem_discovery = discover_problem(df)
    print("🛠 Problem Discovery Complete")

    # ----------------------------
    # 2️⃣ NLP analysis (optional)
    # ----------------------------
    text_cols = [col for col, typ in column_types.items() if typ == "text"]
    nlp_features = None
    if text_cols:
        try:
            nlp_features = run_nlp_analysis(df, text_columns=text_cols)
            print("🧠 NLP Analysis Complete")
        except Exception as e:
            print(f"⚠️ NLP Engine failed: {e}")
            nlp_features = None

    # ----------------------------
    # 3️⃣ Predictive modeling
    # ----------------------------
    numerical_cols = [col for col, typ in column_types.items() if typ == "numerical"]
    targets_dict = {"numerical": numerical_cols, "categorical": []}

    predictions = run_predictive_model(df, targets_dict)
    print("📊 Predictions Complete")

    # ----------------------------
    # 3️⃣.1️⃣ Why Engine - Explain predictions
    # ----------------------------
    why_explanations = {}
    try:
        for target, info in predictions.items():
            if "task" in info and info.get("best_model_pipeline", None):
                model_pipeline = info["best_model_pipeline"]
                # Generate SHAP explanations (or similar)
                why_explanations[target] = explain_predictions(model_pipeline, df, target)
        print("💡 Why Engine explanations generated")
    except Exception as e:
        why_explanations = {"error": str(e)}

    # ----------------------------
    # 4️⃣ Recommendations
    # ----------------------------
    recommendations = {}
    for target, info in predictions.items():
        recs = []
        for val in info.get("sample_predictions", []):
            try:
                val_num = float(val)
                if val_num < 50:
                    category = "Low"
                    rec = "Immediate support required"
                elif val_num < 75:
                    category = "Medium"
                    rec = "Monitor progress"
                else:
                    category = "High"
                    rec = "Offer advanced opportunities"
                recs.append({"prediction": val_num, "category": category, "recommendation": rec})
            except:
                recs.append({"prediction": val, "category": "Unknown", "recommendation": "Check data type"})
        recommendations[target] = recs

    print("🎯 Recommendations Complete")

    # ----------------------------
    # 5️⃣ Basic graphs (histograms)
    # ----------------------------
    graph_files = []
    try:
        graph_files = generate_graphs(df, targets_dict, folder=GRAPH_FOLDER)
        print("📈 Basic Graphs Generation Complete")
    except Exception as e:
        print(f"⚠️ Basic Graph engine failed: {e}")

    # ----------------------------
    # 6️⃣ Business Intelligence Insights
    # ----------------------------
    business_insights = {}
    try:
        business_insights = run_business_intelligence(df)
        print("📌 Business Intelligence Complete")
    except Exception as e:
        business_insights = {"error": str(e)}
        print(f"⚠️ Business engine failed: {e}")

    # ----------------------------
    # 7️⃣ Business graphs (trends + top rankings)
    # ----------------------------
    business_graph_files = []
    try:
        business_graph_files = generate_business_graphs(
            df=df,
            business_insights=business_insights,
            folder=GRAPH_FOLDER
        )
        print("📊 Business Graphs Generated")
    except Exception as e:
        print(f"⚠️ Business graph engine failed: {e}")

    # Merge graphs
    all_graphs = list(set(graph_files + business_graph_files))

    # ----------------------------
    # 8️⃣ Save predictions & recommendations
    # ----------------------------
    # Remove 'best_model_pipeline' before saving JSON
    predictions_to_save = {
        k: {key: val for key, val in v.items() if key != "best_model_pipeline"}
        for k, v in predictions.items()
    }

    saved_files = save_predictions_and_recommendations(predictions_to_save, recommendations, folder=BASE_OUTPUT)
    print("💾 Predictions & Recommendations Saved")

    # ----------------------------
    # 9️⃣ Self-critic
    # ----------------------------
    critic = self_critic(df, predictions)

    # ----------------------------
    # 🔟 Prepare output dict
    # ----------------------------
    output = {
        "autofix_summary": fixes_summary,
        "problem_discovery": problem_discovery,
        "nlp_features_generated": bool(nlp_features is not None),
        "predictions": predictions,  # keep pipelines in memory for Why Engine
        "why_explanations": why_explanations,
        "recommendations": recommendations,
        "business_insights": business_insights,
        "self_critic": critic,
        "graphs": all_graphs,
        "graph_folder": GRAPH_FOLDER,
        "report_path": os.path.join(BASE_OUTPUT, "report.pdf"),
        "saved_files": saved_files,
    }

    return output
