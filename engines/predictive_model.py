from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack

def run_predictive_model(df, tfidf_matrix, column_types):
    results = {}
    numeric_cols = [col for col, t in column_types.items() if t == "numerical"]
    if not numeric_cols:
        return results
    for target_col in numeric_cols:
        mean_val = df[target_col].mean()
        def label(val):
            if val < mean_val * 0.8:
                return 'Needs Support'
            elif val > mean_val * 1.2:
                return 'Excelling'
            else:
                return 'Average'
        df['performance'] = df[target_col].apply(label)
        feature_cols = [c for c in numeric_cols if c != target_col]
        X_numeric = df[feature_cols].values if feature_cols else None
        X = hstack([tfidf_matrix, X_numeric]) if X_numeric is not None else tfidf_matrix
        y = df['performance']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        results[target_col] = {'test_indices': list(range(len(preds))), 'predictions': preds.tolist()}
    print("Predictions generated")
    return results
