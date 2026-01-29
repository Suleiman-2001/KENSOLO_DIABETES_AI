from engines.recommendation_engine import run_recommendations
from engines.predictive_engine import run_predictive_model

def run_generic_ai(df, tfidf_matrix, col_types):
    print("\n🧠 Running Generic Intelligence Engine...")

    run_recommendations(df)
    run_predictive_model(df, tfidf_matrix, col_types)

    print("✅ Generic AI analysis complete")
