# core/router.py
import os
import copy

from engines.vision_engine import generate_graphs, save_predictions_and_recommendations
from engines.nlp_engine import run_nlp_analysis
from engines.predictive_engine import run_predictive_model
from engines.problem_discovery import discover_problem
from engines.self_critic import self_critic
from engines.decision_engine import run_decision_intelligence
from engines.business_engine import run_business_intelligence
from engines.business_graph_engine import generate_business_graphs
from engines.autofix_engine import apply_autofix
from engines.why_engine import explain_predictions  # ✅ Why Engine
from engines.adaptive_engine import run_adaptive_analytics  # ✅ Self-Learning / Adaptive Analytics

# ----------------------------
# Output folders
# ----------------------------
BASE_OUTPUT = r"F:\ARTIFICIAL INTELLIGENCE\AI_Data_Analytics\outputs"
GRAPH_FOLDER = os.path.join(BASE_OUTPUT, "graphs")

os.makedirs(BASE_OUTPUT, exist_ok=True)
os.makedirs(GRAPH_FOLDER, exist_ok=True)


def route_to_engines(df, column_types, autofix=True, industry=None):
    """
    Routes dataset through KENSOLO AI engines
    (NON-BREAKING UPDATE + Large Dataset Optimization + Industry Smart Mode)
    """

    # ----------------------------
    # 0️⃣ Prepare working copy and sample large datasets
    # ----------------------------
    fixes_summary = {}
    working_df = copy.deepcopy(df)

    MAX_ROWS = 10000  # ⚡ Performance threshold
    if len(working_df) > MAX_ROWS:
        print(f"⚡ Dataset has {len(working_df)} rows, sampling {MAX_ROWS} for faster processing...")
        working_df = working_df.sample(n=MAX_ROWS, random_state=42).reset_index(drop=True)

    # ----------------------------
    # 0️⃣ Apply Autofix (safe handling)
    # ----------------------------
    if autofix:
        try:
            autofix_result = apply_autofix(working_df)

            # ✅ Handle both (df, fixes) or more outputs
            if isinstance(autofix_result, tuple):
                working_df = autofix_result[0]
                fixes_summary = autofix_result[1] if len(autofix_result) > 1 else {}
            else:
                working_df = autofix_result

            print(f"🛠 Autofix applied: {fixes_summary}")

        except Exception as e:
            print(f"⚠️ Autofix failed: {e}")
            working_df = df

        # 🔄 Refresh column types after autofix
        column_types = {
            col: ("text" if working_df[col].dtype == "object" else "numerical")
            for col in working_df.columns
        }

    # ----------------------------
    # 0️⃣ Industry Smart Mode (Enhanced)
    # ----------------------------
    industry_insights = {}
    if industry:
        industry_lower = industry.lower()
        print(f"🚀 Industry Smart Mode enabled for: {industry}")

        # ---- Finance ----
        if industry_lower == "finance":
            for col in working_df.columns:
                if "currency" in col.lower() or "account" in col.lower():
                    unique_vals = working_df[col].nunique()
                    missing_vals = working_df[col].isna().sum()
                    if unique_vals > 1000:
                        industry_insights[col] = f"High cardinality ({unique_vals}) expected in financial datasets"
                    if missing_vals > 0:
                        industry_insights[col + "_missing"] = f"{missing_vals} missing values detected"

            # Detect large outliers in Amount-like columns
            for col in working_df.columns:
                if "amount" in col.lower() or "balance" in col.lower():
                    mean_val = working_df[col].mean()
                    std_val = working_df[col].std()
                    outliers = working_df[(working_df[col] < mean_val - 3*std_val) | (working_df[col] > mean_val + 3*std_val)]
                    if not outliers.empty:
                        industry_insights[col + "_outliers"] = f"{len(outliers)} extreme values detected"

        # ---- Healthcare ----
        elif industry_lower == "healthcare":
            for col in working_df.columns:
                if "patient" in col.lower():
                    missing_count = working_df[col].isna().sum()
                    if missing_count > 0:
                        industry_insights[col] = f"Missing patient IDs detected: {missing_count} rows"
                if "age" in col.lower() or "dob" in col.lower():
                    if working_df[col].dtype in [int, float]:
                        invalid_vals = working_df[(working_df[col] < 0) | (working_df[col] > 120)]
                        if not invalid_vals.empty:
                            industry_insights[col + "_invalid"] = f"{len(invalid_vals)} invalid age values detected"

        # ---- Retail ----
        elif industry_lower == "retail":
            if "sales" in working_df.columns:
                sales_mean = working_df["sales"].mean()
                sales_std = working_df["sales"].std()
                industry_insights["sales_anomaly_threshold"] = f"Mean ± 3*STD = {sales_mean - 3*sales_std:.2f} - {sales_mean + 3*sales_std:.2f}"
                anomalies = working_df[(working_df["sales"] < sales_mean - 3*sales_std) | (working_df["sales"] > sales_mean + 3*sales_std)]
                if not anomalies.empty:
                    industry_insights["sales_anomalies_detected"] = f"{len(anomalies)} sales anomalies detected"

        print(f"💡 Industry insights generated: {len(industry_insights)} items")

    # ----------------------------
    # 1️⃣ Problem discovery
    # ----------------------------
    problem_discovery = discover_problem(working_df)
    print("🛠 Problem Discovery Complete")

    # ----------------------------
    # 2️⃣ NLP analysis
    # ----------------------------
    text_cols = [c for c, t in column_types.items() if t == "text"]
    nlp_features = None

    if text_cols:
        try:
            nlp_features = run_nlp_analysis(working_df, text_columns=text_cols)
            print("🧠 NLP Analysis Complete")
        except Exception as e:
            print(f"⚠️ NLP Engine failed: {e}")

    # ----------------------------
    # 3️⃣ Predictive modeling
    # ----------------------------
    numerical_cols = [c for c, t in column_types.items() if t == "numerical"]
    targets_dict = {"numerical": numerical_cols, "categorical": []}

    predictions = run_predictive_model(working_df, targets_dict)
    print("📊 Predictions Complete")

    # ----------------------------
    # 3️⃣.1️⃣ Why Engine
    # ----------------------------
    why_explanations = {}
    try:
        for target, info in predictions.items():
            model_pipeline = info.get("best_model_pipeline")
            if model_pipeline:
                why_explanations[target] = explain_predictions(model_pipeline, working_df, target)
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
                v = float(val)
                low, med = 50, 75
                if industry and industry.lower() == "finance":
                    low, med = 40, 80  # Adjust thresholds for finance
                if v < low:
                    recs.append({"prediction": v, "category": "Low", "recommendation": "Immediate support required"})
                elif v < med:
                    recs.append({"prediction": v, "category": "Medium", "recommendation": "Monitor progress"})
                else:
                    recs.append({"prediction": v, "category": "High", "recommendation": "Offer advanced opportunities"})
            except Exception:
                recs.append({"prediction": val, "category": "Unknown", "recommendation": "Check data type"})
        recommendations[target] = recs

    print("🎯 Recommendations Complete")

    # ----------------------------
    # 5️⃣ Basic graphs
    # ----------------------------
    graph_files = []
    try:
        graph_files = generate_graphs(working_df, targets_dict, folder=GRAPH_FOLDER)
        print("📈 Basic Graphs Generation Complete")
    except Exception as e:
        print(f"⚠️ Basic Graph engine failed: {e}")

    # ----------------------------
    # 6️⃣ Business Intelligence
    # ----------------------------
    try:
        business_insights = run_business_intelligence(working_df)
        print("📌 Business Intelligence Complete")
    except Exception as e:
        business_insights = {"error": str(e)}
        print(f"⚠️ Business engine failed: {e}")

    # ----------------------------
    # 7️⃣ Business graphs
    # ----------------------------
    business_graph_files = []
    try:
        business_graph_files = generate_business_graphs(
            df=working_df,
            business_insights=business_insights,
            folder=GRAPH_FOLDER
        )
        print("📊 Business Graphs Generated")
    except Exception as e:
        print(f"⚠️ Business graph engine failed: {e}")

    all_graphs = list(set(graph_files + business_graph_files))

    # ----------------------------
    # 8️⃣ Save predictions
    # ----------------------------
    predictions_to_save = {
        k: {kk: vv for kk, vv in v.items() if kk != "best_model_pipeline"}
        for k, v in predictions.items()
    }

    saved_files = save_predictions_and_recommendations(predictions_to_save, recommendations, folder=BASE_OUTPUT)
    print("💾 Predictions & Recommendations Saved")

    # ----------------------------
    # 9️⃣ Self-critic
    # ----------------------------
    critic = self_critic(working_df, predictions)

    # ----------------------------
    # 🔟 Decision Intelligence
    # ----------------------------
    decision_intelligence = run_decision_intelligence(
        predictions=predictions,
        business_insights=business_insights,
        self_critic=critic
    )
    print("🧠 Decision Intelligence Complete")

    # ----------------------------
    # 1️⃣1️⃣ Self-Learning / Adaptive Analytics
    # ----------------------------
    adaptive_insights = run_adaptive_analytics(
        df=working_df,
        predictions=predictions,
        recommendations=recommendations,
        industry=industry
    )
    print("🧠 Self-Learning / Adaptive Analytics Complete")

    # ----------------------------
    # Final output
    # ----------------------------
    return {
        "autofix_summary": fixes_summary,
        "problem_discovery": problem_discovery,
        "nlp_features_generated": nlp_features is not None and not (hasattr(nlp_features, 'empty') and nlp_features.empty),
        "predictions": predictions,
        "why_explanations": why_explanations,
        "recommendations": recommendations,
        "business_insights": business_insights,
        "self_critic": critic,
        "decision_intelligence": decision_intelligence,
        "adaptive_insights": adaptive_insights,  # ✅ Self-Learning / Adaptive Analytics output
        "graphs": all_graphs,
        "graph_folder": GRAPH_FOLDER,
        "report_path": os.path.join(BASE_OUTPUT, "report.pdf"),
        "saved_files": saved_files,
        "industry_insights": industry_insights,  # ✅ Enhanced Industry Smart Mode insights
    }
