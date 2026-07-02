import os
import sys
import warnings
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from engines.medical_vision_engine import generate_graphs, save_predictions_and_recommendations
from engines.clinical_nlp_engine import run_nlp_analysis
from engines.problem_discovery import discover_problem
from engines.self_critic import self_critic
from engines.decision_engine import run_decision_intelligence
from engines.business_engine import run_business_intelligence
from engines.business_graph_engine import generate_business_graphs
from engines.autofix_engine import apply_autofix
from engines.explanation_engine import explain_predictions
from engines.adaptive_engine import run_adaptive_analytics
from engines.talk_to_data import talk_to_data_ai
from engines.recommendation_engine import run_recommendations
from engines.diabetes_automl_engine import run_predictive_model as run_advanced_predictive_ai
from engines.predictive_engine import run_predictive_model as run_general_predictive_ai
from engines.unsupervised_engine import run_unsupervised_learning
from engines.memory_engine import track_dataset_history
from engines.future_risk_engine import predict_future_risk
from ingestion.ehr_loader import build_ehr_patient_features, load_ehr_datasets
from ingestion.imaging_loader import build_imaging_patient_features, load_imaging_dataset

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
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _compute_patient_linkage(df, ehr_features, imaging_features):
    tabular_ids = set()
    if "patient_id" in df.columns:
        tabular_ids = set(pd.to_numeric(df["patient_id"], errors="coerce").dropna().astype(int).tolist())
    else:
        tabular_ids = set(range(1, len(df) + 1))

    ehr_ids = set()
    if isinstance(ehr_features, pd.DataFrame) and not ehr_features.empty and "patient_id" in ehr_features.columns:
        ehr_ids = set(pd.to_numeric(ehr_features["patient_id"], errors="coerce").dropna().astype(int).tolist())

    imaging_ids = set()
    if isinstance(imaging_features, pd.DataFrame) and not imaging_features.empty and "patient_id" in imaging_features.columns:
        imaging_ids = set(pd.to_numeric(imaging_features["patient_id"], errors="coerce").dropna().astype(int).tolist())

    tabular_n = max(1, len(tabular_ids))

    overlap_tab_ehr = tabular_ids.intersection(ehr_ids)
    overlap_tab_imaging = tabular_ids.intersection(imaging_ids)
    overlap_all = tabular_ids.intersection(ehr_ids).intersection(imaging_ids)

    return {
        "tabular_patients": int(len(tabular_ids)),
        "ehr_patients": int(len(ehr_ids)),
        "imaging_patients": int(len(imaging_ids)),
        "tabular_ehr_overlap": int(len(overlap_tab_ehr)),
        "tabular_imaging_overlap": int(len(overlap_tab_imaging)),
        "all_modalities_overlap": int(len(overlap_all)),
        "tabular_ehr_overlap_rate": float(len(overlap_tab_ehr) / tabular_n),
        "tabular_imaging_overlap_rate": float(len(overlap_tab_imaging) / tabular_n),
        "all_modalities_overlap_rate": float(len(overlap_all) / tabular_n),
    }


def _save_multimodal_report(multimodal_layer, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    report_json = os.path.join(output_folder, "multimodal_report.json")
    report_csv = os.path.join(output_folder, "multimodal_ablation.csv")

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(_json_safe(multimodal_layer), f, indent=4)

    ablation = ((multimodal_layer or {}).get("fusion", {}).get("fusion_metrics", {}).get("ablation", {}))
    rows = []
    if isinstance(ablation, dict):
        for key, val in ablation.items():
            if isinstance(val, dict):
                rows.append(
                    {
                        "experiment": key,
                        "status": val.get("status"),
                        "features": val.get("features"),
                        "accuracy": val.get("accuracy"),
                        "roc_auc": val.get("roc_auc"),
                    }
                )
            else:
                rows.append({"experiment": key, "status": "scalar", "features": None, "accuracy": None, "roc_auc": val})

    pd.DataFrame(rows).to_csv(report_csv, index=False)

    return {
        "multimodal_report_json": report_json,
        "multimodal_ablation_csv": report_csv,
    }


def _load_optional_dataset_summary(file_name):
    dataset_path = os.path.join(DATA_DIR, file_name)
    if not os.path.exists(dataset_path):
        return {
            "available": False,
            "file": file_name,
            "path": dataset_path,
            "rows": 0,
            "columns": 0,
        }

    try:
        data = pd.read_csv(dataset_path)
        return {
            "available": True,
            "file": file_name,
            "path": dataset_path,
            "rows": int(len(data)),
            "columns": int(data.shape[1]),
        }
    except Exception as e:
        return {
            "available": False,
            "file": file_name,
            "path": dataset_path,
            "rows": 0,
            "columns": 0,
            "error": str(e),
        }


def _generate_longitudinal_forecasts(predictions, df):
    forecasts = {}

    for target, info in (predictions or {}).items():
        model = info.get("best_model_pipeline")
        if model is None:
            continue

        current_features = df.drop(columns=[target], errors="ignore").head(1)
        if current_features.empty:
            continue

        target_forecasts = []
        for years in [1, 3, 5]:
            try:
                target_forecasts.append(predict_future_risk(model, current_features, years=years))
            except Exception as e:
                target_forecasts.append({
                    "years": years,
                    "error": str(e),
                })

        forecasts[target] = target_forecasts

    return forecasts


def _build_multimodal_layer(df, predictions, longitudinal_forecasts):
    tabular_signal = {
        "available": True,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "targets": list((predictions or {}).keys()),
        "longitudinal_forecasts_ready": bool(longitudinal_forecasts),
    }

    ehr_df, ehr_summary = load_ehr_datasets(DATA_DIR)
    imaging_df, imaging_summary = load_imaging_dataset(DATA_DIR)

    ehr_features = build_ehr_patient_features(ehr_df)
    imaging_features = build_imaging_patient_features(imaging_df)

    fusion_status = {
        "status": "skipped",
        "strategy": "feature_level_fusion",
        "note": "Fusion is enabled but could not run.",
    }

    try:
        from pipelines.fusion_layer import late_fusion

        fusion_status = late_fusion(
            tabular_output=tabular_signal,
            ehr_output=ehr_summary,
            imaging_output=imaging_summary,
            tabular_df=df,
            ehr_features=ehr_features,
            imaging_features=imaging_features,
        )
    except Exception as e:
        fusion_status["error"] = str(e)

    if not ehr_features.empty:
        ehr_summary["feature_rows"] = int(len(ehr_features))
        ehr_summary["feature_columns"] = int(ehr_features.shape[1])

    if not imaging_features.empty:
        imaging_summary["feature_rows"] = int(len(imaging_features))
        imaging_summary["feature_columns"] = int(imaging_features.shape[1])

    linkage = _compute_patient_linkage(df, ehr_features, imaging_features)

    return {
        "architecture": "real_multimodal_fusion",
        "modalities": {
            "tabular": tabular_signal,
            "ehr": ehr_summary,
            "imaging": imaging_summary,
        },
        "patient_linkage": linkage,
        "fusion": fusion_status,
    }


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
    target_candidates = [
        col for col in df.columns
        if df[col].nunique() < 20 and df[col].dtype != "object"
    ]
    return target_candidates


def _select_preferred_diabetes_target(df, candidates):
    if not candidates:
        return []

    numeric_df = df.select_dtypes(include=["number"]).copy()
    if numeric_df.empty:
        return [candidates[0]]

    corr_matrix = numeric_df.corr().abs()
    candidate_scores = []

    for candidate in candidates:
        if candidate not in corr_matrix.columns:
            candidate_scores.append((candidate, 0.0))
            continue

        scores = corr_matrix[candidate].drop(labels=[candidate], errors="ignore").dropna()
        score = float(scores.mean()) if not scores.empty else 0.0
        candidate_scores.append((candidate, score))

    best_candidate = max(candidate_scores, key=lambda x: x[1])[0]
    return [best_candidate]


def _detect_column_types(df, target_candidates=None):
    target_candidates = set(target_candidates or [])
    detected = {}

    for col in df.columns:
        series = df[col]
        col_lower = col.lower()
        nunique = series.nunique(dropna=True)

        if pd.api.types.is_datetime64_any_dtype(series):
            detected[col] = "datetime"
        elif col_lower.endswith("id") or col_lower == "id":
            detected[col] = "identifier"
        elif pd.api.types.is_numeric_dtype(series):
            # Low-cardinality numeric columns usually behave like categories.
            if col in target_candidates or nunique < 20:
                detected[col] = "categorical"
            else:
                detected[col] = "numerical"
        elif pd.api.types.is_bool_dtype(series):
            detected[col] = "categorical"
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
            detected[col] = "categorical"
        else:
            detected[col] = "text"

    return detected


def _score_target_by_correlation(df, target):
    numeric_df = df.select_dtypes(include=["number"]).copy()
    if numeric_df.empty or target not in numeric_df.columns:
        return 0.0

    corr_matrix = numeric_df.corr().abs()
    if target not in corr_matrix.columns:
        return 0.0

    scores = corr_matrix[target].drop(labels=[target], errors="ignore").dropna()
    return float(scores.mean()) if not scores.empty else 0.0


def _is_target_like_column(col_name):
    low = str(col_name).lower()
    return not (low == "id" or low.endswith("id") or "_id" in low)


def _detect_task_and_target(df):
    text_columns = [
        c for c in df.columns
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c])
    ]
    text_heavy = (len(text_columns) / max(1, len(df.columns))) >= 0.5

    binary_candidates = [
        c for c in df.columns
        if _is_target_like_column(c)
        and df[c].nunique(dropna=True) == 2
        and not pd.api.types.is_datetime64_any_dtype(df[c])
    ]

    continuous_candidates = [
        c for c in df.columns
        if _is_target_like_column(c)
        and pd.api.types.is_numeric_dtype(df[c])
        and df[c].nunique(dropna=True) >= 20
    ]

    if binary_candidates:
        ranked = sorted(
            binary_candidates,
            key=lambda c: _score_target_by_correlation(df, c),
            reverse=True,
        )
        return {
            "task": "classification",
            "target": ranked[0],
            "reason": "binary column detected",
            "candidates": ranked,
            "text_heavy": text_heavy,
        }

    if continuous_candidates:
        ranked = sorted(
            continuous_candidates,
            key=lambda c: _score_target_by_correlation(df, c),
            reverse=True,
        )
        return {
            "task": "regression",
            "target": ranked[0],
            "reason": "continuous column detected",
            "candidates": ranked,
            "text_heavy": text_heavy,
        }

    if text_heavy:
        return {
            "task": "nlp",
            "target": None,
            "reason": "text-heavy dataset",
            "candidates": text_columns,
            "text_heavy": text_heavy,
        }

    return {
        "task": "clustering",
        "target": None,
        "reason": "no clear target",
        "candidates": [],
        "text_heavy": text_heavy,
    }


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

        results["graph_files"] = generate_graphs(
            chunk_df,
            {"numerical": numerical_cols, "categorical": categorical_cols},
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
    diabetes_target_candidates = _find_diabetes_classification_targets(working_df)
    diabetes_targets = _select_preferred_diabetes_target(working_df, diabetes_target_candidates)
    column_types = _detect_column_types(working_df, diabetes_target_candidates)

    task_detection = _detect_task_and_target(working_df)
    detected_task = task_detection.get("task")
    detected_target = task_detection.get("target")

    print(f"🧭 Task detection: {detected_task} ({task_detection.get('reason')})")
    if detected_target:
        print(f"🎯 Detected target for task: {detected_target}")

    if diabetes_targets:
        print(f"🔎 Detected diabetes target candidate(s): {diabetes_target_candidates}")
        print(f"🎯 Selected target based on correlation: {diabetes_targets[0]}")
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
    monitoring_summary = {}
    unsupervised_learning = {}

    if detected_target is None:
        print("⚙ No explicit target detected. Attempting diabetes surrogate modeling...")
        advanced_ai = run_advanced_predictive_ai(
            working_df,
            diabetes_targets
        )

        aggregated_predictions = advanced_ai.get("predictions", {})
        feature_engineering = advanced_ai.get("feature_engineering", {})
        model_monitoring = advanced_ai.get("model_monitoring", {})
        risk_scoring = advanced_ai.get("risk_scoring", {})
        diabetes_detection = advanced_ai.get("diabetes_detection", {})
        model_leaderboard = advanced_ai.get("model_leaderboard", [])

        if not aggregated_predictions:
            print("⚙ Surrogate modeling unavailable. Running unsupervised fallback mode...")
            unsupervised_learning = run_unsupervised_learning(working_df)
            feature_engineering = {}
            model_monitoring = {
                "status": "completed",
                "task": "unsupervised",
                "engine": "unsupervised_engine",
                "reason": "No target detected; fallback mode activated",
            }
            risk_scoring = {}
            diabetes_detection = {
                "detected_targets": [],
                "prediction_target": None,
                "strategy": "unsupervised_fallback",
                "future_likelihood_supported": False,
                "detected_task": detected_task,
            }
            model_leaderboard = []

    elif detected_task == "classification" and diabetes_targets:
        advanced_ai = run_advanced_predictive_ai(
            working_df,
            diabetes_targets
        )

        aggregated_predictions = advanced_ai.get("predictions", {})
        feature_engineering = advanced_ai.get("feature_engineering", {})
        model_monitoring = advanced_ai.get("model_monitoring", {})
        risk_scoring = advanced_ai.get("risk_scoring", {})
        diabetes_detection = advanced_ai.get("diabetes_detection", {})
        model_leaderboard = advanced_ai.get("model_leaderboard", [])

    elif detected_task in {"classification", "regression"} and detected_target:
        target_dict = {
            "categorical": [detected_target] if detected_task == "classification" else [],
            "numerical": [detected_target] if detected_task == "regression" else [],
        }
        generic_predictions = run_general_predictive_ai(working_df, target_dict)
        aggregated_predictions = generic_predictions
        feature_engineering = {}
        model_monitoring = {
            "status": "completed",
            "task": detected_task,
            "selected_target": detected_target,
            "engine": "general_predictive_engine",
        }
        risk_scoring = {}
        diabetes_detection = {
            "detected_targets": [detected_target],
            "prediction_target": detected_target,
            "strategy": "task_detected",
            "future_likelihood_supported": False,
            "detected_task": detected_task,
        }
        model_leaderboard = []

    else:
        print(f"⚠ Detected task is {detected_task}. Supervised predictive modeling skipped.")
        feature_engineering = {}
        model_monitoring = {
            "status": "skipped",
            "reason": f"Detected task '{detected_task}' does not expose a supervised target",
            "task": detected_task,
        }
        risk_scoring = {}
        diabetes_detection = {
            "detected_targets": [],
            "prediction_target": None,
            "strategy": "task_detected_no_supervised_target",
            "future_likelihood_supported": False,
            "detected_task": detected_task,
        }
        model_leaderboard = []

    if aggregated_predictions:
        aggregated_predictions, monitoring_summary = track_dataset_history(working_df, aggregated_predictions)

    if monitoring_summary:
        model_monitoring = {
            **model_monitoring,
            "dataset_monitoring": monitoring_summary,
        }

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
    # LONGITUDINAL + MULTIMODAL LAYER
    # ----------------------------
    longitudinal_forecasts = _generate_longitudinal_forecasts(aggregated_predictions, working_df)
    multimodal_layer = _build_multimodal_layer(working_df, aggregated_predictions, longitudinal_forecasts)

    if longitudinal_forecasts and isinstance(diabetes_detection, dict):
        diabetes_detection["future_likelihood_supported"] = True
        diabetes_detection["longitudinal_years_supported"] = [1, 3, 5]

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
    selected_model = None
    selected_model_metrics = {}
    final_target = None

    for target, info in aggregated_predictions.items():
        if info.get("best_model"):
            selected_model = info.get("best_model")
            final_target = target
            selected_model_metrics = {
                "target": target,
                "task": info.get("task", "unknown"),
                "accuracy": info.get("accuracy"),
                "precision": info.get("precision"),
                "recall": info.get("recall"),
                "f1_score": info.get("f1_score"),
                "roc_auc": info.get("roc_auc"),
                "best_model": info.get("best_model"),
            }
            break

    best_cv_model = None
    if model_monitoring.get("training", {}).get("best_model"):
        best_cv_model = model_monitoring["training"].get("best_model")

    baseline_model = None
    if model_leaderboard:
        for item in model_leaderboard:
            if item.get("status") == "success":
                baseline_model = item.get("model")
                break

    experiment_summary = {
        "target_count": len(aggregated_predictions),
        "targets": list(aggregated_predictions.keys()),
        "validation_strategy": (
            "Stratified train/test split (20% holdout) plus cross-validation "
            "for model selection and leaderboard ranking"
            if model_leaderboard else
            "Validation strategy not fully available"
        ),
        "baseline_model": baseline_model,
        "final_model": selected_model or best_cv_model,
        "final_target": final_target,
        "selected_model_metrics": selected_model_metrics,
        "leaderboard_count": len(model_leaderboard),
    }

    saved_files = save_predictions_and_recommendations(
        aggregated_predictions,
        recommendations,
        folder=BASE_OUTPUT
    )

    multimodal_files = _save_multimodal_report(multimodal_layer, BASE_OUTPUT)
    saved_files = {**saved_files, **multimodal_files}

    report_path = os.path.join(BASE_OUTPUT, "report.pdf")
    try:
        from fpdf import FPDF

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "KENSOLO AI Diabetes Report", ln=True, align="C")
        pdf.ln(10)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(
            0,
            8,
            "This report summarizes AI predictions, clinical recommendations, and decision intelligence generated for the diabetes analytics pipeline."
        )
        pdf.output(report_path)
    except Exception:
        report_path = None

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
        "saved_files": saved_files,
        "graph_folder": GRAPH_FOLDER,
        "report_path": report_path,
        "diabetes_targets": diabetes_targets,
        "feature_engineering": feature_engineering,
        "model_monitoring": model_monitoring,
        "risk_scoring": risk_scoring,
        "diabetes_detection": diabetes_detection,
        "dataset_monitoring": monitoring_summary,
        "experiment_summary": experiment_summary,
        "model_leaderboard": model_leaderboard,
        "task_detection": task_detection,
        "unsupervised_learning": unsupervised_learning,
        "longitudinal_forecasts": longitudinal_forecasts,
        "multimodal_layer": multimodal_layer,
    }


if __name__ == "__main__":
    print("core/router.py is a library module and does not run by itself.")
    print("Use the application entrypoint instead:")
    print("  python \"f:/ARTIFICIAL INTELLIGENCE DIABETES/AI_Data_Analytics/main.py\"")
    print("  python \"f:/ARTIFICIAL INTELLIGENCE DIABETES/AI_Data_Analytics/core/DIABETES AI SYSTEM.py\"")