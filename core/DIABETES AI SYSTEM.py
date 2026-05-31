import os
import warnings
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from ..engines.medical_vision_engine import generate_graphs, save_predictions_and_recommendations
    from ..engines.clinical_nlp_engine import run_nlp_analysis
    from ..engines.predictive_engine import run_predictive_model
    from ..engines.problem_discovery import discover_problem
    from ..engines.self_critic import self_critic
    from ..engines.decision_engine import run_decision_intelligence
    from ..engines.business_engine import run_business_intelligence
    from ..engines.business_graph_engine import generate_business_graphs
    from ..engines.autofix_engine import apply_autofix
    from ..engines.explanation_engine import explain_predictions
    from ..engines.recommendation_engine import run_recommendations
    from ..engines.adaptive_engine import run_adaptive_analytics
    from ..engines.talk_to_data import talk_to_data_ai

    # =========================
    # DIABETES CORE ENGINES
    # =========================
    from .diabetes_kpi_engine import detect_kpis  # now: clinical biomarkers engine
    from .clinical_data_quality_engine import data_quality_score
    from .diabetes_insight_engine import auto_insights as generate_clinical_insights
    from .diabetes_overview_engine import dataset_overview
except ImportError:
    from engines.medical_vision_engine import generate_graphs, save_predictions_and_recommendations
    from engines.clinical_nlp_engine import run_nlp_analysis
    from engines.predictive_engine import run_predictive_model
    from engines.problem_discovery import discover_problem
    from engines.self_critic import self_critic
    from engines.decision_engine import run_decision_intelligence
    from engines.business_engine import run_business_intelligence
    from engines.business_graph_engine import generate_business_graphs
    from engines.autofix_engine import apply_autofix
    from engines.explanation_engine import explain_predictions
    from engines.recommendation_engine import run_recommendations
    from engines.adaptive_engine import run_adaptive_analytics
    from engines.talk_to_data import talk_to_data_ai

    # =========================
    # DIABETES CORE ENGINES
    # =========================
    from core.diabetes_kpi_engine import detect_kpis  # now: clinical biomarkers engine
    from core.clinical_data_quality_engine import data_quality_score
    from core.diabetes_insight_engine import auto_insights as generate_clinical_insights
    from core.diabetes_overview_engine import dataset_overview

warnings.filterwarnings("ignore")

# ----------------------------
BASE_OUTPUT = r"F:\ARTIFICIAL INTELLIGENCE\AI_Data_Analytics\outputs"
GRAPH_FOLDER = os.path.join(BASE_OUTPUT, "graphs")
POWERBI_FOLDER = os.path.join(BASE_OUTPUT, "for_powerbi")

os.makedirs(BASE_OUTPUT, exist_ok=True)
os.makedirs(GRAPH_FOLDER, exist_ok=True)
os.makedirs(POWERBI_FOLDER, exist_ok=True)

CHUNK_SIZE = 100_000
MAX_PROCESSES = min(4, os.cpu_count() or 1)


def _coerce_dataframe_types(df):
    out = df.copy()
    for col in out.columns:
        try:
            if pd.api.types.is_string_dtype(out[col].dtype):
                out[col] = out[col].astype(object)

            if "date" in col.lower() or "time" in col.lower():
                parsed = pd.to_datetime(out[col], errors="coerce")
                if parsed.notna().sum() > 0:
                    out[col] = parsed

            if pd.api.types.is_object_dtype(out[col].dtype):
                coerced = pd.to_numeric(out[col], errors="coerce")
                if coerced.notna().mean() > 0.5:
                    out[col] = coerced
        except:
            continue
    return out


def _find_diabetes_classification_targets(df):
    diabetes_candidates = []

    for col in df.columns:
        col_lower = col.lower()
        if any(token in col_lower for token in ["diabetes", "diabetic", "has_diabetes", "diabetes_status", "diabetes_flag", "diagnosis"]):
            diabetes_candidates.append(col)
            continue

        if any(token in col_lower for token in ["outcome", "class", "label", "status"]):
            values = df[col].dropna().astype(str).str.lower().unique()
            if not values.size:
                continue
            if set(values.tolist()) & {"0", "1", "yes", "no", "positive", "negative", "diabetes", "diabetic", "non-diabetic", "nondiabetic", "healthy", "sick"}:
                diabetes_candidates.append(col)

    return diabetes_candidates


def process_chunk(chunk_df, column_types):
    results = {}
    try:
        results["problem_discovery"] = discover_problem(chunk_df)

        text_cols = [c for c, t in column_types.items() if t == "text"]
        results["nlp_features"] = run_nlp_analysis(
            chunk_df,
            text_columns=text_cols
        ) if text_cols else None

        numerical_cols = [
            c for c, t in column_types.items()
            if t == "numerical"
        ]

        categorical_cols = [
            c for c, t in column_types.items()
            if t == "categorical"
        ]

        targets_dict = {"numerical": numerical_cols, "categorical": categorical_cols}

        results["predictions"] = run_predictive_model(chunk_df, targets_dict)

        results["graph_files"] = generate_graphs(
            chunk_df,
            targets_dict,
            folder=GRAPH_FOLDER
        )

    except Exception as e:
        results["error"] = str(e)

    return results


def route_to_engines(df, column_types, autofix=True, context=None, query=None):
    """
    DIABETES AI ROUTER:
    Converts dataset into clinical risk intelligence pipeline
    """

    working_df = df

    # ----------------------------
    # preprocessing
    # ----------------------------
    working_df = _coerce_dataframe_types(working_df)
    print(f"⚡ Dataset size: {len(working_df)} rows")

    if autofix:
        try:
            result = apply_autofix(working_df)
            working_df = result[0] if isinstance(result, tuple) else result
        except:
            pass

    # fill missing values
    num_cols = working_df.select_dtypes(include=["number"]).columns
    working_df[num_cols] = working_df[num_cols].fillna(working_df[num_cols].mean())

    cat_cols = working_df.select_dtypes(include=["object"]).columns
    working_df[cat_cols] = working_df[cat_cols].fillna("Unknown")

    # recompute schema
    column_types = {}
    diabetes_targets = _find_diabetes_classification_targets(working_df)

    for col in working_df.columns:
        if pd.api.types.is_numeric_dtype(working_df[col]):
            column_types[col] = "numerical"
        elif pd.api.types.is_datetime64_any_dtype(working_df[col]):
            column_types[col] = "datetime"
        elif col in diabetes_targets:
            column_types[col] = "categorical"
        else:
            column_types[col] = "text"

    if diabetes_targets:
        print(f"🔎 Detected diabetes classification target(s): {diabetes_targets}")
    else:
        print("🔎 No explicit diabetes classification target detected.")

    # ----------------------------
    # chunking
    # ----------------------------
    chunks = (
        [working_df[i:i + CHUNK_SIZE] for i in range(0, len(working_df), CHUNK_SIZE)]
        if len(working_df) > CHUNK_SIZE
        else [working_df]
    )

    aggregated_predictions = {}
    aggregated_nlp = None
    aggregated_graphs = []
    problem_discovery = {}

    # ----------------------------
    # processing
    # ----------------------------
    if len(chunks) > 1:
        with ThreadPoolExecutor(max_workers=MAX_PROCESSES) as executor:
            futures = [
                executor.submit(process_chunk, c, column_types)
                for c in chunks
            ]

            for f in as_completed(futures):
                r = f.result()

                problem_discovery.update(r.get("problem_discovery", {}))
                aggregated_predictions.update(r.get("predictions", {}))
                aggregated_graphs.extend(r.get("graph_files", []))

                if r.get("nlp_features") is not None:
                    aggregated_nlp = r["nlp_features"]

    else:
        r = process_chunk(working_df, column_types)
        aggregated_predictions = r.get("predictions", {})
        aggregated_graphs = r.get("graph_files", [])
        problem_discovery = r.get("problem_discovery", {})

    self_critic_result = self_critic(working_df, aggregated_predictions)

    # ----------------------------
    # EXPLANATION LAYER (CLINICAL)
    # ----------------------------
    explanations = {}
    for target, info in aggregated_predictions.items():
        model = info.get("best_model_pipeline")
        if model:
            explanations[target] = explain_predictions(model, working_df, target)

    # ----------------------------
    # DIABETES RECOMMENDATIONS
    # ----------------------------
    recommendations = run_recommendations(aggregated_predictions)
    print("🎯 Diabetes recommendations generated")

    # ----------------------------
    # CLINICAL ANALYTICS LAYER
    # ----------------------------
    clinical_insights = run_business_intelligence(working_df)

    graph_files = generate_business_graphs(
        df=working_df,
        business_insights=clinical_insights,
        folder=GRAPH_FOLDER
    )

    aggregated_graphs.extend(graph_files)

    # ----------------------------
    # CORE DIABETES METRICS
    # ----------------------------
    kpi_summary = detect_kpis(working_df)
    quality_summary = data_quality_score(working_df)
    insight_summary = generate_clinical_insights(working_df)
    overview_summary = dataset_overview(working_df)

    decision_intelligence = run_decision_intelligence(
        predictions=aggregated_predictions,
        business_insights=clinical_insights,
        self_critic=self_critic_result,
        clinical_kpis=kpi_summary
    )

    adaptive_insights = run_adaptive_analytics(
        df=working_df,
        predictions=aggregated_predictions,
        recommendations=recommendations,
        industry="diabetes"
    )

    # ----------------------------
    # OPTIONAL TALK TO DATA
    # ----------------------------
    talk_to_data_result = None
    if query:
        talk_to_data_result = talk_to_data_ai(working_df, query)

    # ----------------------------
    # OUTPUT EXPORTS
    # ----------------------------
    saved_files = save_predictions_and_recommendations(
        aggregated_predictions,
        recommendations,
        folder=BASE_OUTPUT
    )

    return {
        "kpi_summary": kpi_summary,
        "quality_summary": quality_summary,
        "insight_summary": insight_summary,
        "overview_summary": overview_summary,
        "predictions": aggregated_predictions,
        "recommendations": recommendations,
        "clinical_insights": clinical_insights,
        "explanations": explanations,
        "graphs": aggregated_graphs,
        "problem_discovery": problem_discovery,
        "self_critic": self_critic_result,
        "decisions": decision_intelligence,
        "adaptive_insights": adaptive_insights,
        "talk_to_data_result": talk_to_data_result,
        "saved_files": saved_files
    }