import os
import numpy as np
import pandas as pd
import geopandas as gp
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from adjustText import adjust_text

# -----------------------------
# 0. Load data
# -----------------------------
country='PL'
version = 'v3'
datadir = fr'C:\PROCESSING\2025_built_dynamics\results_landscapes_{version}'
result_df = gp.read_file(
    datadir + os.sep + f'landscapes_{country}_GHSL_54009_{version}.gpkg',
    ignore_geometry=True
)

# Fix ambiguous names
fixed_names = {
    '2-1-1': '2-1-1 wysoczyzny słabo rozcięte',
    '2-1-2': '2-1-2 wysoczyzny silnie rozcięte'
}
for code, newname in fixed_names.items():
    mask = result_df['nazwa'].str.contains(code, na=False)
    result_df.loc[mask, 'nazwa'] = newname

nazwa = result_df["nazwa"].values

# Group polygons by landscape class
pop_cols = [c for c in result_df.columns if c.startswith("pop_")]
bu_cols  = [c for c in result_df.columns if c.startswith("bu_")]

agg_dict = {col: "sum" for col in pop_cols + bu_cols}
agg_dict["pow_km2"] = "sum"

result_df = result_df.groupby("nazwa", as_index=False).agg(agg_dict)

# -----------------------------
# 1. Build time-series matrices
# -----------------------------
years = [int(c.split("_")[1]) for c in pop_cols]

### NEW: POPULATION DENSITY ###
pop = result_df.set_index("nazwa")[pop_cols]
pop_density = pop.div(result_df.set_index("nazwa")["pow_km2"], axis=0)
pop_density.columns = years

### NEW: BUILT-UP DENSITY ###
bu = result_df.set_index("nazwa")[bu_cols]
bu_density = bu.div(result_df.set_index("nazwa")["pow_km2"], axis=0)
bu_density.columns = ["bu_"+str(y) for y in years]

### NEW: COMBINE BOTH ###
cluster_df = pd.concat([pop_density, bu_density], axis=1)

print("cluster_df.shape:", cluster_df.shape)

# -----------------------------
# 2. Row-wise standardization
# -----------------------------
X = (cluster_df.sub(cluster_df.mean(axis=1), axis=0)
                 .div(cluster_df.std(axis=1), axis=0)).values

# -----------------------------
# 3. PCA
# -----------------------------
pca = PCA(n_components=0.95, svd_solver="full")
pcs = pca.fit_transform(X)

print("pcs.shape:", pcs.shape)
print("Explained variance ratios:", pca.explained_variance_ratio_)

plt.figure(figsize=(6,4))
plt.bar(range(1, len(pca.explained_variance_ratio_)+1),
        pca.explained_variance_ratio_*100)
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance (%)")
plt.title("Variance explained by first PCs")
plt.show()

# -----------------------------
# 4. Choose k (elbow + silhouette)
# -----------------------------
Sum_of_squared_distances = []
sil_scores = []
K = range(2, 7)

for k in K:
    km = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels = km.fit_predict(pcs)
    Sum_of_squared_distances.append(km.inertia_)
    sil_scores.append(silhouette_score(pcs, labels))

plt.figure(figsize=(6,4))
plt.plot(K, Sum_of_squared_distances, "bx-")
plt.xlabel("k"); plt.ylabel("Within-cluster SSE")
plt.title("Elbow Method")
plt.show()

plt.figure(figsize=(6,4))
plt.plot(K, sil_scores, "ro-")
plt.xlabel("k"); plt.ylabel("Silhouette Score")
plt.title("Silhouette Analysis")
plt.show()

# -----------------------------
# 5. Select number of clusters
# -----------------------------
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
cluster_labels = kmeans.fit_predict(pcs)

# -----------------------------
# 6. Result table
# -----------------------------
res = pd.DataFrame(
    pcs[:, :min(5, pcs.shape[1])],
    index=cluster_df.index,
    columns=[f"PC{i+1}" for i in range(pcs.shape[1])]
)
res["cluster"] = cluster_labels

# -----------------------------
# 7. PC1–PC2 plot
# -----------------------------
plt.figure(figsize=(10,6))
scatter = plt.scatter(
    res["PC1"], res["PC2"],
    c=res["cluster"], cmap="cool", s=80, edgecolor="k"
)

texts = []
for i, name in enumerate(res.index):
    texts.append(plt.text(res["PC1"].iloc[i], res["PC2"].iloc[i], name, fontsize=8))

adjust_text(texts, arrowprops=dict(arrowstyle="-", lw=0.5))

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.title(f"Areas in PC1–PC2 space (k={n_clusters} clusters)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# -----------------------------
# 8. Plot dynamics per cluster
# -----------------------------
for c in range(n_clusters):
    members_pop = pop_density.loc[res["cluster"] == c]
    members_bu  = bu_density.loc[res["cluster"] == c]

    fig, ax = plt.subplots(1, 2, figsize=(12,4), sharex=True)

    # Population trajectories
    for idx, row in members_pop.iterrows():
        ax[0].plot(years, row.values, "k-", alpha=0.3)
    ax[0].plot(years, members_pop.mean(axis=0), "r-", lw=2)
    ax[0].set_title(f"Cluster {c+1} — Population density")
    ax[0].set_ylabel("Pop per km²")
    ax[0].grid(True, alpha=0.3)

    # Built-up trajectories
    for idx, row in members_bu.iterrows():
        ax[1].plot(years, row.values, "k-", alpha=0.3)
    ax[1].plot(years, members_bu.mean(axis=0), "r-", lw=2)
    ax[1].set_title(f"Cluster {c+1} — Built-up density")
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
