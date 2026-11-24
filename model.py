"""Reusable machine-learning pipeline for Mall Customer Segmentation PR 1."""
from __future__ import annotations

import io
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage

REQUIRED_COLUMNS = ["CustomerID", "Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]

def load_dataset(source) -> pd.DataFrame:
    """Load a CSV path, uploaded file, or file-like object."""
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        df = pd.read_csv(source)
    # Accept common alternative spellings while preserving the PR's column names.
    aliases = {
        "Annual_Income": "Annual Income (k$)",
        "Annual Income": "Annual Income (k$)",
        "Spending_Score": "Spending Score (1-100)",
        "Spending Score": "Spending Score (1-100)",
    }
    df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns and v not in df.columns})
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "CSV is missing required columns: " + ", ".join(missing) +
            ". Expected: CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100)."
        )
    return df

def preprocess_data(raw_df: pd.DataFrame):
    """Clean/encode data and scale the three numeric features as specified in PR 1."""
    df = load_dataset(raw_df)
    df = df.drop_duplicates().copy()
    df = df.dropna(subset=REQUIRED_COLUMNS).copy()
    df["Annual_Income"] = pd.to_numeric(df["Annual Income (k$)"], errors="coerce")
    df["Spending_Score"] = pd.to_numeric(df["Spending Score (1-100)"], errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df = df.dropna(subset=["Age", "Annual_Income", "Spending_Score"]).copy()

    # Match the assignment's requested encoding: Female=0, Male=1.
    gender_map = {"Female": 0, "Male": 1, "F": 0, "M": 1}
    gender_values = df["Gender"].astype(str).str.strip().str.title()
    encoded = gender_values.map(gender_map)
    if encoded.isna().any():
        encoder = LabelEncoder()
        encoded = pd.Series(encoder.fit_transform(gender_values), index=df.index)
    df["Gender"] = encoded.astype(int)

    # Keep the four modelling columns; CustomerID is used only as an identifier.
    clean_df = df[["Gender", "Age", "Annual_Income", "Spending_Score"]].reset_index(drop=True)
    scaler = StandardScaler()
    numeric_cols = ["Age", "Annual_Income", "Spending_Score"]
    scaled = pd.DataFrame(
        scaler.fit_transform(clean_df[numeric_cols]),
        columns=numeric_cols,
        index=clean_df.index
    )
    df_scaled = pd.concat([clean_df[["Gender"]], scaled], axis=1)
    df_2f = df_scaled[["Annual_Income", "Spending_Score"]].copy()
    return clean_df, df_scaled, df_2f, scaler

def _safe_silhouette(X, labels):
    labels = np.asarray(labels)
    unique = set(labels)
    if -1 in unique:
        mask = labels != -1
        X_eval = np.asarray(X)[mask]
        labels_eval = labels[mask]
    else:
        X_eval = np.asarray(X)
        labels_eval = labels
    if len(set(labels_eval)) < 2 or len(set(labels_eval)) >= len(labels_eval):
        return np.nan
    return float(silhouette_score(X_eval, labels_eval))

def find_kmeans_k(X):
    inertias, silhouettes = [], []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(float(km.inertia_))
        if k >= 2:
            silhouettes.append((k, _safe_silhouette(X, km.labels_)))
    valid = [(k, score) for k, score in silhouettes if np.isfinite(score)]
    best_k = max(valid, key=lambda x: x[1])[0] if valid else 5
    return inertias, silhouettes, int(best_k)

def fit_all_models(clean_df, df_2f, kmeans_k=None, hierarchical_k=5):
    X = df_2f
    inertias, sil_scores, best_sil_k = find_kmeans_k(X)
    k = int(kmeans_k or best_sil_k)
    # The PR describes k=5 as typical; the app uses the data-driven silhouette optimum by default.
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    hier_k = max(2, min(int(hierarchical_k), len(X)-1))
    hier = AgglomerativeClustering(n_clusters=hier_k, linkage="ward").fit(X)

    nn = NearestNeighbors(n_neighbors=min(4, len(X))).fit(X)
    distances, _ = nn.kneighbors(X)
    k_distances = np.sort(distances[:, -1])
    grid_rows = []
    best_db = None
    best_db_score = -np.inf
    for eps in [0.2, 0.3, 0.4, 0.5, 0.6]:
        for min_samples in [3, 4, 5, 6]:
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
            n_clusters = len(set(labels) - {-1})
            n_noise = int(np.sum(labels == -1))
            score = _safe_silhouette(X, labels)
            grid_rows.append({
                "eps": eps, "min_samples": min_samples,
                "n_clusters": n_clusters, "noise_points": n_noise,
                "silhouette_score": score
            })
            # Prefer meaningful multi-cluster solutions, with a valid silhouette and lower noise.
            if n_clusters >= 2 and np.isfinite(score):
                adjusted = score - (n_noise / len(labels)) * 0.05
                if adjusted > best_db_score:
                    best_db_score, best_db = adjusted, (eps, min_samples)
    grid_df = pd.DataFrame(grid_rows)
    if best_db is None:
        # Safe fallback if the data do not form two density-separated clusters at the tested settings.
        best_db = (0.5, 4)
    db_eps, db_min_samples = best_db
    db = DBSCAN(eps=db_eps, min_samples=db_min_samples).fit(X)

    results = clean_df.copy()
    results["KMeans_Cluster"] = km.labels_
    results["Hier_Cluster"] = hier.labels_
    results["DBSCAN_Cluster"] = db.labels_
    return {
        "results": results, "kmeans": km, "hierarchical": hier, "dbscan": db,
        "inertias": inertias, "silhouette_scores": sil_scores,
        "best_k": best_sil_k, "selected_k": k, "k_distances": k_distances,
        "dbscan_grid": grid_df, "dbscan_eps": db_eps, "dbscan_min_samples": db_min_samples,
        "linkage_matrix": linkage(X, method="ward")
    }

def cluster_profile(results: pd.DataFrame, label_col: str) -> pd.DataFrame:
    return results.groupby(label_col)[["Age", "Annual_Income", "Spending_Score"]].mean().round(2)

def metrics_summary(X, results, model_output):
    rows = []
    for algorithm, col in [
        ("K-Means", "KMeans_Cluster"),
        ("Agglomerative", "Hier_Cluster"),
        ("DBSCAN", "DBSCAN_Cluster"),
    ]:
        labels = results[col].to_numpy()
        if algorithm == "DBSCAN":
            mask = labels != -1
            X_eval, y_eval = np.asarray(X)[mask], labels[mask]
        else:
            X_eval, y_eval = np.asarray(X), labels
        n_clusters = len(set(y_eval))
        valid = len(y_eval) > n_clusters and n_clusters >= 2
        rows.append({
            "Algorithm": algorithm,
            "n_clusters": n_clusters,
            "Silhouette Score": silhouette_score(X_eval, y_eval) if valid else np.nan,
            "Davies-Bouldin Index": davies_bouldin_score(X_eval, y_eval) if valid else np.nan,
            "Calinski-Harabasz Index": calinski_harabasz_score(X_eval, y_eval) if valid else np.nan,
        })
    return pd.DataFrame(rows)
