# engines/nlp_engine.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def run_nlp_analysis(df, text_columns, max_features=50):
    """
    Clinical-safe NLP feature engine.

    Enhancements:
    - Handles missing/empty text safely
    - Keeps feature metadata for explainability
    - Prevents pipeline breakage on low-quality text
    """

    nlp_features = pd.DataFrame()
    feature_map = {}

    for col in text_columns:

        if col not in df.columns:
            continue

        # Safe text cleaning
        text_data = df[col].fillna("").astype(str)

        # Skip column if completely empty
        if text_data.str.strip().eq("").all():
            continue

        vectorizer = TfidfVectorizer(max_features=max_features)

        tfidf_matrix = vectorizer.fit_transform(text_data)

        feature_names = [f"{col}_{f}" for f in vectorizer.get_feature_names_out()]

        # Convert to DataFrame
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=feature_names
        )

        # Store mapping for explainability layer
        feature_map[col] = list(vectorizer.get_feature_names_out())

        # Merge safely
        nlp_features = pd.concat([nlp_features, tfidf_df], axis=1)

    return nlp_features, feature_map