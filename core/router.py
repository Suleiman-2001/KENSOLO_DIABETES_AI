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
from core.insight_engine import auto_insights as generate_industry_insights
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


def _coerce_dataframe_types(df):
    """Coerce common column types to usable pandas/numpy dtypes.

    - Convert pandas StringDtype to object for downstream libs that expect numpy dtypes.
    - Parse columns with 'date'/'time' in their name to datetime where possible.
    - Convert object columns that look numeric to numeric (if >50% parseable).
    Returns a new DataFrame (shallow copy) with coerced columns.
    """
    out = df.copy()
    for col in out.columns:
        try:
            # Convert pandas nullable StringDtype to object
            if pd.api.types.is_string_dtype(out[col].dtype):
                out[col] = out[col].astype(object)

            # Try date/time parsing for obvious column names
            if "date" in col.lower() or "time" in col.lower():
                parsed = pd.to_datetime(out[col], errors="coerce")
                if parsed.notna().sum() > 0:
                    out[col] = parsed

            # If object column, check if majority can be numeric and coerce
            if pd.api.types.is_object_dtype(out[col].dtype):
                coerced = pd.to_numeric(out[col], errors="coerce")
                non_null = coerced.notna().sum()
                if non_null / max(1, len(coerced)) > 0.5:
                    out[col] = coerced
        except Exception:
            # Best-effort coercion — ignore failures per-column
            continue
    return out

# ----------------------------
# Helper to process heavy engines per chunk
# ----------------------------
def process_chunk(chunk_df, column_types, industry=None):
    results = {}
    try:
        results["problem_discovery"] = discover_problem(chunk_df)
        text_cols = [c for c, t in column_types.items() if t == "text"]
        results["nlp_features"] = run_nlp_analysis(chunk_df, text_columns=text_cols) if text_cols else None
        # build numerical targets but skip any datetime columns (could have slipped through earlier)
        numerical_cols = []
        for c, t in column_types.items():
            if t == "numerical":
                # double-check dtype on the chunk
                if not (pd.api.types.is_datetime64_any_dtype(chunk_df[c].dtype) or
                        pd.api.types.is_datetime64tz_dtype(chunk_df[c].dtype)):
                    numerical_cols.append(c)
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

def route_to_engines(df, column_types, autofix=True, industry=None, query=None):
    """
    Routes dataset through KENSOLO AI engines
    Optimized for large-scale speed while keeping ALL engines intact.
    NOTHING removed. NOTHING skipped.
    """

    fixes_summary = {}

    # =========================================================
    # 1️⃣ REMOVE DEEP COPY (MAJOR SPEED BOOST)
    # =========================================================
    working_df = df  # removed copy.deepcopy(df)

    # =========================================================
    # 1.1️⃣ Coerce common column dtypes so downstream ML libs don't choke
    # =========================================================
    try:
        working_df = _coerce_dataframe_types(working_df)
        print("ℹ️ Coerced dataframe dtypes for ML compatibility")
    except Exception as e:
        print(f"⚠️ Type coercion failed: {e}")

    print(f"⚡ Dataset size: {len(working_df)} rows")

    # =========================================================
    # 2️⃣ Autofix (UNCHANGED)
    # =========================================================
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

    # Recompute column types after coercion/autofix so downstream engines get correct hints
    column_types = {}
    for col in working_df.columns:
        dtype = working_df[col].dtype
        if pd.api.types.is_datetime64_any_dtype(dtype) or pd.api.types.is_datetime64tz_dtype(dtype):
            column_types[col] = "datetime"
        elif dtype == "object":
            column_types[col] = "text"
        else:
            column_types[col] = "numerical"


    # =========================================================
    # 3️⃣ Industry Insights (UNCHANGED)
    # =========================================================
    industry_insights = {}
    if industry:
        try:
            industry_insights = generate_industry_insights(working_df)  # no second import needed
            print(f"💡 Industry insights generated: {len(industry_insights)} items")
        except Exception as e:
            industry_insights = {"error": str(e)}
            print(f"⚠️ Industry Smart Mode failed: {e}")


    # =========================================================
    # 4️⃣ Chunking (UNCHANGED STRUCTURE)
    # =========================================================
    if len(working_df) > CHUNK_SIZE:
        chunks = [working_df[i:i + CHUNK_SIZE] for i in range(0, len(working_df), CHUNK_SIZE)]
        use_chunking = True
    else:
        chunks = [working_df]
        use_chunking = False

    aggregated_predictions = {}
    aggregated_nlp = None
    aggregated_graphs = []
    problem_discovery = {}

    # =========================================================
    # 5️⃣ HEAVY ENGINE EXECUTION (OPTIMIZED BUT NOT REMOVED)
    # =========================================================
    if use_chunking:
        print(f"⚡ Large dataset detected, processing in {len(chunks)} chunks...")

        with ProcessPoolExecutor(max_workers=MAX_PROCESSES) as executor:
            futures = []

            for chunk in chunks:
                # 🔥 SAMPLE ONLY FOR ML/NLP INSIDE PROCESS
                if len(chunk) > 150_000:
                    chunk_for_model = chunk.sample(150_000, random_state=42)
                else:
                    chunk_for_model = chunk

                futures.append(
                    executor.submit(process_chunk, chunk_for_model, column_types, industry)
                )

            for future in as_completed(futures):
                result = future.result()

                # Fix aggregation safely
                problem_discovery.update(result.get("problem_discovery", {}))

                for k, v in result.get("predictions", {}).items():
                    aggregated_predictions[k] = v

                nlp_chunk = result.get("nlp_features")
                if nlp_chunk is not None:
                    aggregated_nlp = (
                        pd.concat([aggregated_nlp, nlp_chunk])
                        if aggregated_nlp is not None else nlp_chunk
                    )

                aggregated_graphs.extend(result.get("graph_files", []))

    else:
        result = process_chunk(working_df, column_types, industry)
        aggregated_predictions = result.get("predictions", {})
        aggregated_nlp = result.get("nlp_features")
        aggregated_graphs = result.get("graph_files", [])
        problem_discovery = result.get("problem_discovery", {})

    # =========================================================
    # WHY ENGINE (UNCHANGED)
    # =========================================================
    why_explanations = {}
    try:
        for target, info in aggregated_predictions.items():
            model_pipeline = info.get("best_model_pipeline")
            if model_pipeline:
                why_explanations[target] = explain_predictions(
                    model_pipeline, working_df, target
                )
        print("💡 Why Engine explanations generated")
    except Exception as e:
        why_explanations = {"error": str(e)}

    # =========================================================
    # RECOMMENDATIONS (UNCHANGED)
    # =========================================================
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

    # =========================================================
    # BUSINESS INTELLIGENCE (FULL DATA)
    # =========================================================
    try:
        business_insights = run_business_intelligence(working_df)
        print("📌 Business Intelligence Complete")
    except Exception as e:
        business_insights = {"error": str(e)}

    # =========================================================
    # BUSINESS GRAPHS (UNTOUCHED)
    # =========================================================
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

    # =========================================================
    # EVERYTHING BELOW REMAINS 100% UNCHANGED
    # =========================================================

    predictions_to_save = {
        k: {kk: vv for kk, vv in v.items() if kk != "best_model_pipeline"}
        for k, v in aggregated_predictions.items()
    }

    saved_files = save_predictions_and_recommendations(
        predictions_to_save, recommendations, folder=BASE_OUTPUT
    )

    critic = self_critic(working_df, aggregated_predictions)

    decision_intelligence = run_decision_intelligence(
        predictions=aggregated_predictions,
        business_insights=business_insights,
        self_critic=critic
    )

    kpi_summary = detect_kpis(working_df)
    quality_summary = data_quality_score(working_df)
    insight_summary = generate_industry_insights(working_df)
    overview_summary = dataset_overview(working_df)

    adaptive_insights = run_adaptive_analytics(
        df=working_df,
        predictions=aggregated_predictions,
        recommendations=recommendations,
        industry=industry
    )

    try:
        from engines.memory_engine import track_dataset_history
        memory_info = track_dataset_history(working_df, aggregated_predictions)
    except Exception as e:
        memory_info = {"error": str(e)}

    try:
        from engines.impact_engine import calculate_revenue_cost_impact
        impact_results = calculate_revenue_cost_impact(aggregated_predictions, business_insights)
    except Exception as e:
        impact_results = {"error": str(e)}

    try:
        from engines.industry_engine import adapt_recommendations_by_industry
        industry_adjusted_recommendations = (
            adapt_recommendations_by_industry(recommendations, industry) if industry else {}
        )
    except Exception as e:
        industry_adjusted_recommendations = {"error": str(e)}

    try:
        from engines.assumption_engine import run_scenario_analysis
        scenario_results = run_scenario_analysis(
            aggregated_predictions, business_insights, recommendations
        )
    except Exception as e:
        scenario_results = {"error": str(e)}

    talk_to_data_result = None
    if query:
        try:
            talk_to_data_result = talk_to_data_ai(df=working_df, query=query)
        except Exception as e:
            talk_to_data_result = {"error": str(e)}

    # Power BI exports unchanged
    safe_to_csv(kpi_summary, os.path.join(POWERBI_FOLDER, "kpi_summary.csv"))
    safe_to_csv(quality_summary, os.path.join(POWERBI_FOLDER, "quality_summary.csv"))
    safe_to_csv(insight_summary, os.path.join(POWERBI_FOLDER, "insight_summary.csv"))
    safe_to_csv(overview_summary, os.path.join(POWERBI_FOLDER, "overview_summary.csv"))
    safe_to_csv(recommendations, os.path.join(POWERBI_FOLDER, "recommendations.csv"))
    if isinstance(industry_insights, dict):
            safe_to_csv(list(industry_insights.items()), os.path.join(POWERBI_FOLDER, "industry_insights.csv"))
    else:
            safe_to_csv(industry_insights, os.path.join(POWERBI_FOLDER, "industry_insights.csv"))


    print(f"✅ All Power BI CSVs saved to: {POWERBI_FOLDER}")

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
