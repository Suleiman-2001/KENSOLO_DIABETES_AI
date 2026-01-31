# engines/nlp_engine.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def run_nlp_analysis(df, text_columns):
    """
    Convert text columns into TF-IDF features for predictive modeling.
    """
    nlp_features = pd.DataFrame()

    for col in text_columns:
        vectorizer = TfidfVectorizer(max_features=50)
        tfidf_matrix = vectorizer.fit_transform(df[col].astype(str))
        feature_names = [f"{col}_{feat}" for feat in vectorizer.get_feature_names_out()]
        nlp_features = pd.concat(
            [nlp_features, pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)],
            axis=1
        )

    return nlp_features
