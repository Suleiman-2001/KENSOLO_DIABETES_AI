import re
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def run_nlp(df, column_types):
    print("Running NLP engine...")
    text_cols = [col for col, t in column_types.items() if t == "text"]
    print(f"Text columns detected: {text_cols}")
    if not text_cols:
        return None
    combined_text = df[text_cols].astype(str).agg(" ".join, axis=1)
    cleaned_text = combined_text.apply(clean_text)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(cleaned_text)
    print("NLP Feature extraction complete")
    return X
