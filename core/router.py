# core/router.py
import os
import copy
import warnings
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

from engines.vision_engine import generate_graphs, save_predictions_and_recommendations
from engines.nlp_engine import run_nlp_analysis
from engines.predictive_engine import run_predictive_model
from engines.problem_discovery import discover_problem
from engines.self_critic import self_critic
from engines.decision_engine import run_decision_intelligence
from engines.business_engine import run_business_intelligence
from engines.business_graph_engine import generate_business_graphs
from engines.autofix_engine import apply_autofix
from engines.why_engine import explain_predictions
from engines.adaptive_engine import run_adaptive_analytics
from engines.talk_to_data import talk_to_data_ai
from core.kpi_engine import detect_kpis
from core.quality_engine import data_quality_score
from core.insight_engine import auto_insights
from core.overview_engine import dataset_overview

warnings.filterwarnings("ignore")

# ----------------------------
# Output folders
# ----------------------------
BASE_OUTPUT = r"F:\ARTIFICIAL INTELLIGENCE\AI_Data_Analytics\outputs"
GRAPH_FOLDER = os.path.join(BASE_OUTPUT, "graphs")
POWERBI_FOLDER = os.path.join(BASE_OUTPUT, "for_powerbi")

os.makedirs(BASE_OUTPUT, exist_ok=True)
os.makedirs(GRAPH_FOLDER, exist_ok=True)
os.makedirs(POWERBI_FOLDER, exist_ok=True)

# ----------------------------
# Chunking & threading settings
# ----------------------------
CHUNK_SIZE = 100_000  # Adjustable
MAX_PROCESSES = min(4, os.cpu_count() or 1)  # Dynamically adjust based on CPU

# ----------------------------
# Helper to process heavy engines per chunk
# ----------------------------
def process_chunk(chunk_df, column_types, industry=None):
    results = {}
    try:
        results["problem_discovery"] = discover_problem(chunk_df)
        text_cols = [c for c, t in column_types.items() if t == "text"]
        results["nlp_features"] = run_nlp_analysis(chunk_df, text_columns=text_cols) if text_cols else None
        numerical_cols = [c for c, t in column_types.items() if t == "numerical"]
        targets_dict = {"numerical": numerical_cols, "categorical": []}
        results["predictions"] = run_predictive_model(chunk_df, targets_dict)
        results["graph_files"] = generate_graphs(chunk_df, targets_dict, folder=GRAPH_FOLDER)
    except Exception as e:
        results["error"] = str(e)
    return results

# ----------------------------
# Safe CSV export helper
# ----------------------------
def safe_to_csv(data, filepath):
    try:
        if isinstance(data, dict):
            if all(isinstance(v, dict) for v in data.values()):
                df = pd.json_normalize(data, sep="_").T.reset_index()
                df.rename(columns={"index": "target"}, inplace=True)
            elif all(isinstance(v, list) for v in data.values()):
                rows = []
                for k, lst in data.items():
                    for item in lst:
                        row = item.copy() if isinstance(item, dict) else {"value": item}
                        row["target"] = k
                        rows.append(row)
                df = pd.DataFrame(rows)
            else:
                df = pd.DataFrame(list(data.items()), columns=["key", "value"])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame([data])
        df.to_csv(filepath, index=False)
        print(f"✅ Saved CSV: {filepath}")
    except Exception as e:
        print(f"⚠️ Failed to save {filepath}: {e}")

# ----------------------------
# Enhanced Save Predictions & Recommendations
# ----------------------------
def save_predictions_and_recommendations(predictions, recommendations, folder):
    os.makedirs(folder, exist_ok=True)
    paths = {}

    # Save predictions.json
    pred_json_path = os.path.join(folder, "predictions.json")
    pd.Series(predictions).to_json(pred_json_path, indent=4)
    paths["predictions.json"] = pred_json_path

    # Save recommendations.json
    rec_json_path = os.path.join(folder, "recommendations.json")
    pd.Series(recommendations).to_json(rec_json_path, indent=4)
    paths["recommendations.json"] = rec_json_path

    # Save predictions.csv
    pred_csv_path = os.path.join(folder, "predictions.csv")
    pd.DataFrame.from_dict(predictions, orient="index").to_csv(pred_csv_path)
    paths["predictions.csv"] = pred_csv_path

    # Save recommendations.csv
    rec_csv_path = os.path.join(folder, "recommendations.csv")
    pd.DataFrame([
        {**item, "target": target}
        for target, rec_list in recommendations.items()
        for item in rec_list
    ]).to_csv(rec_csv_path, index=False)
    paths["recommendations.csv"] = rec_csv_path

    return paths

# ----------------------------
# ROUTER FUNCTION
# ----------------------------
def route_to_engines(df, column_types, autofix=True, industry=None, query=None):
    """
    Routes dataset through KENSOLO AI engines
    Optimized for chunked + threaded execution while keeping all engines intact.
    """
    fixes_summary = {}
    working_df = copy.deepcopy(df)

    # ----------------------------
    # Handle massive datasets with sampling
    # ----------------------------
    if len(working_df) > 20_000:
        sample_size = min(150_000, len(working_df))
        print(f"⚡ Dataset has {len(working_df)} rows, sampling {sample_size} for faster processing...")
        working_df = working_df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    # ----------------------------
    # Autofix
    # ----------------------------
    if autofix:
        try:
            autofix_result = apply_autofix(working_df)
            if isinstance(autofix_result, tuple):
                working_df = autofix_result[0]
                fixes_summary = autofix_result[1] if len(autofix_result) > 1 else {}
            else:
                working_df = autofix_result
            print(f"🛠 Autofix applied: {fixes_summary}")
        except Exception as e:
            print(f"⚠️ Autofix failed: {e}")
            working_df = df

        # Update column types
        column_types = {
            col: ("text" if working_df[col].dtype == "object" else "numerical")
            for col in working_df.columns
        }

    # ----------------------------
    # Industry Insights
    # ----------------------------
    industry_insights = {}
    if industry:
        try:
            from engines.industry_engine import generate_industry_insights
            industry_insights = generate_industry_insights(working_df, industry)
            print(f"💡 Industry insights generated: {len(industry_insights)} items")
        except Exception as e:
            industry_insights = {"error": str(e)}
            print(f"⚠️ Industry Smart Mode failed: {e}")

    # ----------------------------
    # Chunking for heavy engines
    # ----------------------------
    if len(working_df) > CHUNK_SIZE:
        chunks = [working_df[i:i + CHUNK_SIZE] for i in range(0, len(working_df), CHUNK_SIZE)]
        use_chunking = True
    else:
        chunks = [working_df]
        use_chunking = False

    aggregated_predictions = {}
    aggregated_nlp = None
    aggregated_graphs = []

    if use_chunking:
        print(f"⚡ Large dataset detected, processing in {len(chunks)} chunks...")
        with ProcessPoolExecutor(max_workers=MAX_PROCESSES) as executor:
            futures = [executor.submit(process_chunk, chunk, column_types, industry) for chunk in chunks]
            for future in as_completed(futures):
                result = future.result()
                # Aggregate predictions
                for k, v in result.get("predictions", {}).items():
                    if k not in aggregated_predictions:
                        aggregated_predictions[k] = v
                # Aggregate NLP features
                nlp_chunk = result.get("nlp_features")
                if nlp_chunk is not None:
                    aggregated_nlp = pd.concat([aggregated_nlp, nlp_chunk]) if aggregated_nlp is not None else nlp_chunk
                # Aggregate graph files
                aggregated_graphs.extend(result.get("graph_files", []))
    else:
        # Small dataset
        result = process_chunk(working_df, column_types, industry)
        aggregated_predictions = result.get("predictions", {})
        aggregated_nlp = result.get("nlp_features")
        aggregated_graphs = result.get("graph_files", [])

    # ----------------------------
    # Keep all engines intact
    # ----------------------------
    problem_discovery = result.get("problem_discovery", {})

    # Why Engine
    why_explanations = {}
    try:
        for target, info in aggregated_predictions.items():
            model_pipeline = info.get("best_model_pipeline")
            if model_pipeline:
                why_explanations[target] = explain_predictions(model_pipeline, working_df, target)
        print("💡 Why Engine explanations generated")
    except Exception as e:
        why_explanations = {"error": str(e)}

    # Recommendations
    recommendations = {}
    for target, info in aggregated_predictions.items():
        recs = []
        for val in info.get("sample_predictions", []):
            try:
                v = float(val)
                low, med = (40, 80) if industry and industry.lower() == "finance" else (50, 75)
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
    # Business Intelligence
    # ----------------------------
    try:
        business_insights = run_business_intelligence(working_df)
        print("📌 Business Intelligence Complete")
    except Exception as e:
        business_insights = {"error": str(e)}
        print(f"⚠️ Business engine failed: {e}")

    # ----------------------------
    # Business Graphs
    # ----------------------------
    try:
        business_graph_files = generate_business_graphs(
            df=working_df,
            business_insights=business_insights,
            folder=GRAPH_FOLDER
        )
        aggregated_graphs.extend(business_graph_files)
        print("📊 Business Graphs Generated")
    except Exception as e:
        print(f"⚠️ Business graph engine failed: {e}")

    all_graphs = list(set(aggregated_graphs))

    # ----------------------------
    # Save predictions & recommendations
    # ----------------------------
    predictions_to_save = {
        k: {kk: vv for kk, vv in v.items() if kk != "best_model_pipeline"}
        for k, v in aggregated_predictions.items()
    }
    saved_files = save_predictions_and_recommendations(predictions_to_save, recommendations, folder=BASE_OUTPUT)
    print("💾 Predictions & Recommendations Saved:", saved_files)

    # ----------------------------
    # Self-Critic
    # ----------------------------
    critic = self_critic(working_df, aggregated_predictions)

    # ----------------------------
    # Decision Intelligence
    # ----------------------------
    decision_intelligence = run_decision_intelligence(
        predictions=aggregated_predictions,
        business_insights=business_insights,
        self_critic=critic
    )
    print("🧠 Decision Intelligence Complete")

    # ----------------------------
    # KPI / Quality / Insight / Overview engines
    # ----------------------------
    try:
        kpi_summary = detect_kpis(working_df)
        quality_summary = data_quality_score(working_df)
        insight_summary = auto_insights(working_df)
        overview_summary = dataset_overview(working_df)
        print("📊 KPI / Quality / Insight / Overview engines completed")
    except Exception as e:
        kpi_summary = {}
        quality_summary = {}
        insight_summary = {}
        overview_summary = {}
        print(f"⚠️ KPI / Quality / Insight / Overview engines failed: {e}")

    # ----------------------------
    # Adaptive Analytics
    # ----------------------------
    try:
        adaptive_insights = run_adaptive_analytics(
            df=working_df,
            predictions=aggregated_predictions,
            recommendations=recommendations,
            industry=industry
        )
        print("🧠 Self-Learning / Adaptive Analytics Complete")
    except Exception as e:
        adaptive_insights = {"error": str(e)}
        print(f"⚠️ Adaptive Analytics failed: {e}")

    # ----------------------------
    # Memory Engine
    # ----------------------------
    try:
        from engines.memory_engine import track_dataset_history
        memory_info = track_dataset_history(working_df, aggregated_predictions)
        print("🗄 Memory Engine: Dataset history tracked")
    except Exception as e:
        memory_info = {"error": str(e)}
        print(f"⚠️ Memory Engine failed: {e}")

    # ----------------------------
    # Impact Engine
    # ----------------------------
    try:
        from engines.impact_engine import calculate_revenue_cost_impact
        impact_results = calculate_revenue_cost_impact(aggregated_predictions, business_insights)
        print("💰 Impact Engine: Revenue/Cost impact calculated")
    except Exception as e:
        impact_results = {"error": str(e)}
        print(f"⚠️ Impact Engine failed: {e}")

    # ----------------------------
    # Industry Engine
    # ----------------------------
    try:
        from engines.industry_engine import adapt_recommendations_by_industry
        industry_adjusted_recommendations = adapt_recommendations_by_industry(recommendations, industry) if industry else {}
        print("🏭 Industry Engine: Recommendations adapted to industry")
    except Exception as e:
        industry_adjusted_recommendations = {"error": str(e)}
        print(f"⚠️ Industry Engine failed: {e}")

    # ----------------------------
    # Assumption Engine
    # ----------------------------
    try:
        from engines.assumption_engine import run_scenario_analysis
        scenario_results = run_scenario_analysis(aggregated_predictions, business_insights, recommendations)
        print("🔮 Assumption Engine: Scenario analysis complete")
    except Exception as e:
        scenario_results = {"error": str(e)}
        print(f"⚠️ Assumption Engine failed: {e}")

    # ----------------------------
    # Talk-to-Your-Data AI
    # ----------------------------
    talk_to_data_result = None
    if query:
        try:
            talk_to_data_result = talk_to_data_ai(df=working_df, query=query)
            print(f"💬 Talk-to-Your-Data AI result: {talk_to_data_result}")
        except Exception as e:
            talk_to_data_result = {"error": str(e)}
            print(f"⚠️ Talk-to-Your-Data AI failed: {e}")

    # ----------------------------
    # Export CSVs for Power BI
    # ----------------------------
    safe_to_csv(kpi_summary, os.path.join(POWERBI_FOLDER, "kpi_summary.csv"))
    safe_to_csv(quality_summary, os.path.join(POWERBI_FOLDER, "quality_summary.csv"))
    safe_to_csv(insight_summary, os.path.join(POWERBI_FOLDER, "insight_summary.csv"))
    safe_to_csv(overview_summary, os.path.join(POWERBI_FOLDER, "overview_summary.csv"))
    safe_to_csv(recommendations, os.path.join(POWERBI_FOLDER, "recommendations.csv"))
    safe_to_csv(list(industry_insights.items()), os.path.join(POWERBI_FOLDER, "industry_insights.csv"))

    print(f"✅ All Power BI CSVs saved to: {POWERBI_FOLDER}")

    # ----------------------------
    # Final output
    # ----------------------------
    return {
        "kpi_summary": kpi_summary,
        "quality_summary": quality_summary,
        "insight_summary": insight_summary,
        "overview_summary": overview_summary,
        "autofix_summary": fixes_summary,
        "problem_discovery": problem_discovery,
        "nlp_features_generated": aggregated_nlp is not None and not (hasattr(aggregated_nlp, 'empty') and aggregated_nlp.empty),
        "predictions": aggregated_predictions,
        "why_explanations": why_explanations,
        "recommendations": recommendations,
        "business_insights": business_insights,
        "self_critic": critic,
        "decision_intelligence": decision_intelligence,
        "adaptive_insights": adaptive_insights,
        "memory_info": memory_info,
        "impact_results": impact_results,
        "industry_adjusted_recommendations": industry_adjusted_recommendations,
        "scenario_analysis": scenario_results,
        "graphs": all_graphs,
        "graph_folder": GRAPH_FOLDER,
        "report_path": os.path.join(BASE_OUTPUT, "report.pdf"),
        "saved_files": saved_files,
        "industry_insights": industry_insights,
        "talk_to_data_result": talk_to_data_result,
    }
