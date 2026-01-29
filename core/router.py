# core/router.py
from engines.problem_discovery import discover_problem
from engines.nlp_engine import run_nlp
from engines.predictive_engine import run_predictive_model
from engines.recommendation_engine import run_recommendations
from engines.self_critic import self_critic
from engines.vision_engine import generate_graphs
from utils.target_selector import select_targets
from utils.result_saver import save_results

def route_to_engines(df, column_types):
    problems = discover_problem(df)
    tfidf = run_nlp(df, column_types)
    targets = select_targets(df, column_types)

    predictions = run_predictive_model(df, tfidf, targets) or {}
    critic = self_critic(df, predictions)

    recommendations = {} if critic["blocked"] else run_recommendations(predictions)

    output = {
        "problem_discovery": problems,
        "predictions": predictions,
        "recommendations": recommendations,
        "self_critic": critic
    }

    # Save results and generate graphs
    save_results(output)
    generate_graphs(df, targets)

    return output
