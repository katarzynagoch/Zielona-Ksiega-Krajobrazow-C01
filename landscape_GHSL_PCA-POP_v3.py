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
datadir = r'C:\PROCESSING\2025_built_dynamics\results_landscapes_%s'%version
result_df = gp.read_file(datadir + os.sep + f'landscapes_{country}_GHSL_54009_{version}.gpkg',
                          ignore_geometry=True)

# SAME FIXES AS BEFORE
fixed_names = {
    '2-1-1': '2-1-1 wysoczyzny słabo rozcięte',
    '2-1-2': '2-1-2 wysoczyzny silnie rozcięte'
}
for code, newname in fixed_names.items():
    mask = result_df['nazwa'].str.contains(code, na=False)
    result_df.loc[mask, 'nazwa'] = newname
    
    
# Filter uncertain instances
# result_df = result_df[~result_df['nazwa'].str.contains('2-2-2 wysoczyzny silnie', na=False)]
nazwa = result_df["nazwa"].values

# Group polygons by the landscape polygon ID
pop_cols = [c for c in result_df.columns if c.startswith("pop_")]
bu_cols = [c for c in result_df.columns if c.startswith("bu_")]
# Build aggregation dict for ALL pop_ and bu_ columns
agg_dict = {col: "sum" for col in pop_cols + bu_cols}
agg_dict['pow_km2']='sum'

result_df  = result_df.groupby("nazwa", as_index=False).agg(agg_dict)

# -----------------------------
# 1. Build area × years matrix (rows = areas, cols = years)
# -----------------------------
# pop_cols = bu_cols
years = [int(c.split("_")[1]) for c in pop_cols]

# Population density
pop = result_df.set_index("nazwa")[pop_cols]
area_pop = pop.div(result_df.set_index("nazwa")["pow_km2"], axis=0)
area_pop.columns = years

bu_pop = result_df.set_index("nazwa")[pop_cols] 
for i, pc in enumerate(pop_cols): 
    bu_pop[pc] = bu_pop[pc].div(result_df.set_index("nazwa")[bu_cols[i]], axis=0)

cluster_df = area_pop
print("cluster_df.shape:", cluster_df.shape)   # (n_areas, n_years)

# -----------------------------
# 2. Standardize (row-wise z-score)
# -----------------------------
X = (cluster_df.sub(cluster_df.mean(axis=1), axis=0)
                 .div(cluster_df.std(axis=1), axis=0)).values

# -----------------------------
# 3. PCA (run once)
# -----------------------------
pca = PCA(n_components=0.95, svd_solver="full")
pcs = pca.fit_transform(X)

print("pcs.shape:", pcs.shape)
print("Explained variance ratios:", pca.explained_variance_ratio_)

# Plot explained variance
plt.figure(figsize=(6,4), dpi=120)
plt.bar(range(1, len(pca.explained_variance_ratio_)+1),
        pca.explained_variance_ratio_*100)
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance (%)")
plt.title("Variance explained by first PCs")
plt.show()

# -----------------------------
# 4. Cluster number selection (elbow + silhouette)
# -----------------------------
Sum_of_squared_distances = []
sil_scores = []
K = range(2, 7)

for k in K:
    km = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels = km.fit_predict(pcs)
    Sum_of_squared_distances.append(km.inertia_)
    sil_scores.append(silhouette_score(pcs, labels))

# Elbow plot
plt.figure(figsize=(6,4))
plt.plot(K, Sum_of_squared_distances, "bx-")
plt.xlabel("k")
plt.ylabel("Within-cluster SSE")
plt.title("Elbow Method For Optimal k")
plt.show()

# Silhouette plot
plt.figure(figsize=(6,4))
plt.plot(K, sil_scores, "ro-")
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Analysis")
plt.show()

# -----------------------------
# 5. Choose number of clusters explicitly
# -----------------------------
n_clusters = 2  # <--- set manually after inspecting plots

kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
cluster_labels = kmeans.fit_predict(pcs)

# -----------------------------
# 6. Results table
# -----------------------------
res = pd.DataFrame(
    pcs[:, :min(5, pcs.shape[1])],
    index=cluster_df.index,
    columns=[f"PC{i+1}" for i in range(pcs.shape[1])]
)
res["cluster"] = cluster_labels
print(res.head())

# -----------------------------
# 7. Plot PC1 vs PC2 with clusters
# -----------------------------
plt.figure(figsize=(10,6))
scatter = plt.scatter(
    res["PC1"], res["PC2"],
    c=res["cluster"], cmap="cool", s=80, edgecolor="k"
)

texts = []
for i, name in enumerate(res.index):
    texts.append(
        plt.text(res["PC1"].iloc[i], res["PC2"].iloc[i], name, fontsize=8)
    )

adjust_text(texts, arrowprops=dict(arrowstyle="-", lw=0.5))

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.title(f"Typy krajobrazów w przestrzeni PC1-PC2 (liczba klastrów k={n_clusters})")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(datadir + os.sep + 'PCA-POPdensity_%s.png'%version,dpi=150,bbox_inches='tight')
plt.show()

# -----------------------------
# 8. Plot time series per cluster
# -----------------------------
fig, axes = plt.subplots(1, n_clusters, figsize=(5*n_clusters, 5), sharey=True)
if n_clusters == 1:
    axes = [axes]  # ensure iterable

for c in range(n_clusters):
    ax = axes[c]
    members = cluster_df[res["cluster"] == c]
    
    # Plot all series in cluster (raw density, not standardized)
    for idx, row in members.iterrows():
        ax.plot(years, row.values, "k-", alpha=0.3)
        ax.text(years[-1]+0.2, row.values[-1], idx, fontsize=7, alpha=0.6)
    
    # Plot cluster centroid (average curve)
    centroid = members.mean(axis=0)
    ax.plot(years, centroid.values, "r-", lw=2)
    
    ax.set_title(f"Cluster {c+1} (n={len(members)})")
    ax.set_xlabel("Year")
    if c == 0:
        ax.set_ylabel("Population density")
    ax.grid(True, linestyle="--", alpha=0.3)

plt.suptitle("Population density trajectories per cluster", fontsize=14)
plt.tight_layout()
plt.show()


# Interpret PC2
pc2_loadings = pca.components_[1]    # second principal component
print(pd.Series(pc2_loadings, index=years))

pc1_loadings = pca.components_[0]    # second principal component
print(pd.Series(pc1_loadings, index=years))
