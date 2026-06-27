import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def _prepare_numeric_frame(df):
    numeric_df = df.select_dtypes(include=["number", "bool"]).copy()
    if numeric_df.empty:
        return pd.DataFrame()

    # Convert booleans to numeric and impute missing values.
    for col in numeric_df.columns:
        if pd.api.types.is_bool_dtype(numeric_df[col]):
            numeric_df[col] = numeric_df[col].astype(int)

    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
    numeric_df = numeric_df.fillna(numeric_df.median(numeric_only=True))
    numeric_df = numeric_df.fillna(0)
    return numeric_df


def _select_k_for_kmeans(X_scaled):
    n_samples = X_scaled.shape[0]
    if n_samples < 3:
        return 1, None

    upper = min(8, max(2, n_samples // 10), n_samples - 1)
    if upper < 2:
        return 1, None

    best_k = 2
    best_score = -1.0

    for k in range(2, upper + 1):
        try:
            model = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = model.fit_predict(X_scaled)
            if len(np.unique(labels)) < 2:
                continue
            score = silhouette_score(X_scaled, labels)
            if score > best_score:
                best_score = float(score)
                best_k = k
        except Exception:
            continue

    if best_score < 0:
        return 2, None

    return best_k, best_score


def run_unsupervised_learning(df):
    numeric_df = _prepare_numeric_frame(df)

    if numeric_df.empty:
        return {
            "status": "skipped",
            "reason": "No numeric features available for unsupervised learning",
            "kmeans": {},
            "pca": {},
            "anomaly_detection": {},
        }

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_df)

    # PCA summary for structure discovery.
    n_components = min(3, X_scaled.shape[1], X_scaled.shape[0])
    pca_result = {}
    if n_components >= 1:
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        pca_result = {
            "n_components": int(n_components),
            "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
            "sample_projection": X_pca[:10].tolist(),
        }

    # KMeans clustering.
    best_k, silhouette = _select_k_for_kmeans(X_scaled)
    if best_k <= 1:
        cluster_labels = np.zeros(X_scaled.shape[0], dtype=int)
        kmeans_result = {
            "n_clusters": 1,
            "silhouette_score": None,
            "cluster_counts": {"0": int(len(cluster_labels))},
            "sample_labels": cluster_labels[:20].tolist(),
        }
    else:
        kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        counts = pd.Series(cluster_labels).value_counts().sort_index().to_dict()
        kmeans_result = {
            "n_clusters": int(best_k),
            "silhouette_score": float(silhouette) if silhouette is not None else None,
            "cluster_counts": {str(k): int(v) for k, v in counts.items()},
            "sample_labels": cluster_labels[:20].tolist(),
            "inertia": float(kmeans.inertia_),
        }

    # Isolation Forest anomaly detection.
    contamination = 0.05 if X_scaled.shape[0] >= 40 else 0.1
    iso = IsolationForest(random_state=42, contamination=contamination)
    anomaly_flags = iso.fit_predict(X_scaled)
    anomaly_mask = anomaly_flags == -1
    anomaly_scores = iso.decision_function(X_scaled)

    anomaly_result = {
        "anomaly_count": int(anomaly_mask.sum()),
        "anomaly_rate": float(anomaly_mask.mean()),
        "sample_anomaly_indices": np.where(anomaly_mask)[0][:20].tolist(),
        "sample_anomaly_scores": [float(x) for x in anomaly_scores[:20]],
        "contamination": float(contamination),
    }

    return {
        "status": "completed",
        "features_used": numeric_df.columns.tolist(),
        "row_count": int(len(numeric_df)),
        "kmeans": kmeans_result,
        "pca": pca_result,
        "anomaly_detection": anomaly_result,
    }
