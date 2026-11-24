import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from model import load_dataset, preprocess_data, fit_all_models, cluster_profile, metrics_summary

st.set_page_config(page_title="Mall Customer Segmentation | PR 1", page_icon="🛍️", layout="wide")
sns.set_style("whitegrid")
RED = "#CC0000"

st.title("Mall Customer Segmentation")
st.caption("Unsupervised Learning · PR 1 · K-Means | Agglomerative Hierarchical | DBSCAN")
st.markdown("Explore customer groups using the Mall Customer Segmentation dataset.")

with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader("Upload Mall_Customers.csv", type=["csv"])
    st.caption("Use the Kaggle Mall Customer Segmentation CSV with 200 rows and 5 columns.")
    st.markdown("[Download dataset from Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)")
    k_override = st.selectbox("K-Means clusters", ["Auto (Silhouette)", 3, 4, 5, 6, 7], index=0)
    hier_k = st.slider("Hierarchical clusters", min_value=2, max_value=10, value=5)

if uploaded is None:
    st.info("Upload **Mall_Customers.csv** in the sidebar to run the analysis.")
    st.subheader("Expected CSV columns")
    st.code("CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100)")
    st.stop()

try:
    raw = load_dataset(uploaded)
    clean_df, df_scaled, df_2f, scaler = preprocess_data(raw)
    output = fit_all_models(clean_df, df_2f, None if k_override == "Auto (Silhouette)" else int(k_override), hier_k)
    results = output["results"]
except Exception as exc:
    st.error(f"Could not process the uploaded dataset: {exc}")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Customers analysed", len(clean_df))
c2.metric("K-Means clusters", output["selected_k"])
c3.metric("DBSCAN noise points", int((results["DBSCAN_Cluster"] == -1).sum()))

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview & EDA", "K-Means", "Hierarchical", "DBSCAN", "Comparison & Insights"
])

with tab1:
    st.subheader("Dataset preview")
    st.dataframe(raw.head(), use_container_width=True)
    st.write("Cleaned data shape:", clean_df.shape)
    st.write("Missing values after cleaning:", int(clean_df.isna().sum().sum()))
    st.subheader("Feature distributions")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, ["Age", "Annual_Income", "Spending_Score"]):
        sns.histplot(clean_df[col], kde=True, ax=ax, color=RED)
        ax.set_title(col)
    st.pyplot(fig)
    plt.close(fig)
    st.subheader("Pairplot and correlation")
    pair = sns.pairplot(clean_df, vars=["Age", "Annual_Income", "Spending_Score"])
    st.pyplot(pair.figure)
    plt.close(pair.figure)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(clean_df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("**Why scaling?** Distance-based methods are affected by feature magnitude. StandardScaler puts Age, Annual_Income and Spending_Score on comparable scales. Gender remains binary and is not scaled.")

with tab2:
    st.subheader("Elbow Method")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, 11), output["inertias"], marker="o")
    ax.set(title="Elbow Method — Optimal k for K-Means", xlabel="Number of clusters (k)", ylabel="Inertia")
    st.pyplot(fig)
    plt.close(fig)
    st.subheader("Silhouette Score")
    ss = pd.DataFrame(output["silhouette_scores"], columns=["k", "Silhouette Score"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ss["k"], ss["Silhouette Score"], marker="o")
    ax.set(title="Silhouette Score by k", xlabel="Number of clusters (k)", ylabel="Silhouette Score")
    st.pyplot(fig)
    plt.close(fig)
    st.write(f"Data-driven silhouette choice: **k = {output['best_k']}**. Current fitted model: **k = {output['selected_k']}**.")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=results, x="Annual_Income", y="Spending_Score", hue="KMeans_Cluster", palette="Set1", ax=ax)
    ax.scatter(output["kmeans"].cluster_centers_[:, 0] * scaler.scale_[1] + scaler.mean_[1],
               output["kmeans"].cluster_centers_[:, 1] * scaler.scale_[2] + scaler.mean_[2],
               c="black", marker="X", s=180, label="Centroids")
    ax.set_title("K-Means Clustering — Annual Income vs Spending Score")
    st.pyplot(fig)
    plt.close(fig)
    st.subheader("K-Means cluster profiles")
    st.dataframe(cluster_profile(results, "KMeans_Cluster"), use_container_width=True)

with tab3:
    st.subheader("Ward linkage dendrogram")
    from scipy.cluster.hierarchy import dendrogram
    fig, ax = plt.subplots(figsize=(13, 5))
    dendrogram(output["linkage_matrix"], truncate_mode="lastp", p=30, leaf_rotation=90, leaf_font_size=8, ax=ax)
    ax.set_title("Dendrogram — Agglomerative Hierarchical Clustering (Ward Linkage)")
    ax.set_xlabel("Customer groups"); ax.set_ylabel("Ward distance")
    st.pyplot(fig)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=results, x="Annual_Income", y="Spending_Score", hue="Hier_Cluster", palette="Set2", ax=ax)
    ax.set_title("Agglomerative Hierarchical Clustering")
    st.pyplot(fig)
    plt.close(fig)
    st.subheader("Hierarchical cluster profiles")
    st.dataframe(cluster_profile(results, "Hier_Cluster"), use_container_width=True)
    st.subheader("K-Means vs Hierarchical")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.scatterplot(data=results, x="Annual_Income", y="Spending_Score", hue="KMeans_Cluster", palette="Set1", ax=axes[0], legend=False)
    axes[0].set_title("K-Means")
    sns.scatterplot(data=results, x="Annual_Income", y="Spending_Score", hue="Hier_Cluster", palette="Set2", ax=axes[1], legend=False)
    axes[1].set_title("Agglomerative")
    st.pyplot(fig)
    plt.close(fig)

with tab4:
    st.subheader("4-NN k-distance plot")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(output["k_distances"])
    ax.set(title="4-NN Distance Plot — Estimating DBSCAN eps", xlabel="Points sorted by distance", ylabel="4th-neighbour distance")
    st.pyplot(fig)
    plt.close(fig)
    st.subheader("DBSCAN parameter grid")
    st.dataframe(output["dbscan_grid"].round(3), use_container_width=True)
    st.write(f"Selected eps = **{output['dbscan_eps']}**, min_samples = **{output['dbscan_min_samples']}**.")
    fig, ax = plt.subplots(figsize=(8, 5))
    noise = results["DBSCAN_Cluster"] == -1
    sns.scatterplot(data=results[~noise], x="Annual_Income", y="Spending_Score", hue="DBSCAN_Cluster", palette="Set2", ax=ax)
    ax.scatter(results.loc[noise, "Annual_Income"], results.loc[noise, "Spending_Score"], c="black", marker="x", label="Noise")
    ax.set_title("DBSCAN Clustering — Annual Income vs Spending Score")
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("**DBSCAN concepts:** `eps` sets the neighbourhood radius; `min_samples` sets the minimum number of nearby points needed to form a dense region. Points outside dense regions are labelled noise (`-1`).")

with tab5:
    st.subheader("Three-algorithm comparison")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, col, title, palette in [
        (axes[0], "KMeans_Cluster", "K-Means", "Set1"),
        (axes[1], "Hier_Cluster", "Agglomerative", "Set2"),
        (axes[2], "DBSCAN_Cluster", "DBSCAN", "Set2"),
    ]:
        sns.scatterplot(data=results, x="Annual_Income", y="Spending_Score", hue=col, palette=palette, ax=ax, legend=False)
        ax.set_title(title)
    fig.suptitle("Clustering Algorithm Comparison — Mall Customer Segmentation")
    st.pyplot(fig)
    plt.close(fig)
    metric_df = metrics_summary(df_2f, results, output)
    st.subheader("Metrics summary")
    st.dataframe(metric_df.round(3), use_container_width=True)
    st.markdown("""
    ### Business Insights for Mall Management
    - **High income, high spending:** reward with loyalty benefits, premium previews and personalised offers.
    - **High income, low spending:** use premium product discovery, concierge-style recommendations and targeted promotions.
    - **Low income, high spending:** offer budget bundles, value deals and reward points.
    - **Low income, low spending:** use low-cost seasonal promotions and entry-level offers.
    - **Middle segments:** test category-specific campaigns and monitor response before increasing spend.

    **Recommendation:** K-Means is a practical default for deployment because its segments and centroids are easy to explain to a marketing team. Use Hierarchical clustering when a dendrogram-based hierarchy is useful, and DBSCAN when density and noise detection are important. Validate segment labels against actual campaign outcomes before operational use.
    """)
    st.download_button("Download clustered customer results (CSV)", results.to_csv(index=False).encode("utf-8"), "mall_customer_clusters.csv", "text/csv")

st.divider()
st.caption("Red & White Skill Education · Unsupervised Learning · PR 1")
