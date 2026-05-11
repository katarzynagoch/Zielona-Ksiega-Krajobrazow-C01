# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import geopandas as gp
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Optional (DTW time-series kmeans)
try:
    from tslearn.clustering import TimeSeriesKMeans
    tslearn_available = True
except Exception:
    tslearn_available = False

# Optional label adjustment
try:
    from adjustText import adjust_text
    have_adjusttext = True
except Exception:
    have_adjusttext = False

# -----------------------------
# 0. Prepare polygon-level data
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

# Identify pop columns and years
pop_cols = [c for c in result_df.columns if c.startswith("pop_")]
years = [int(c.split("_")[1]) for c in pop_cols]
years_sorted_idx = np.argsort(years)           # in case columns are not ordered
pop_cols = [pop_cols[i] for i in years_sorted_idx]
years = [years[i] for i in years_sorted_idx]

# Basic check
print("n polygons:", len(result_df))
print("years:", years[:5], "...", years[-5:])

# Build population density time series (pop per km2)
df = result_df.copy()  # keep original safe
if "pow_km2" not in df.columns:
    raise KeyError("pow_km2 (area km2) missing from your df")

pop_ts = df[pop_cols].div(df["pow_km2"], axis=0)
pop_ts.columns = [int(c.split("_")[1]) for c in pop_cols]  # numeric year columns

# Optionally drop polygons with missing or zero area/pop for stability
valid_mask = (~pop_ts.isnull().any(axis=1)) & (df["pow_km2"] > 0)
print("valid polygons:", valid_mask.sum())
pop_ts = pop_ts.loc[valid_mask]
meta = df.loc[valid_mask, ["FUN1_2025"]]            # keep functional type
if "geometry" in df.columns:
    geom = df.loc[valid_mask, "geometry"]           # optional for mapping
else:
    geom = None

# -----------------------------
# 1. (Optional) incorporate FUN1_2025 as a feature
# -----------------------------
# Two choices: (A) use FUN1_2025 only for interpretation (recommended),
#             (B) encode and append to feature matrix (careful: categorical weighting).

include_fun_as_feature = True   # set True to append one-hot encoding to features

if include_fun_as_feature:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ohe.fit(meta[["FUN1_2025"]])
    
    fun_encoded = ohe.transform(meta.loc[pop_ts.index, ["FUN1_2025"]])
    fun_cols = [f"FUN_{c}" for c in ohe.categories_[0]]
    
    fun_df = pd.DataFrame(fun_encoded, index=pop_ts.index, columns=fun_cols)

# -----------------------------
# 2. Standardize / transform time series
# -----------------------------
# Approach A: row-wise z-score (centering each polygon's trajectory)
#
# We'll do row-wise z-score so clustering cares about shape, not absolute level.
def row_zscore(df_ts):
    means = df_ts.mean(axis=1)
    stds = df_ts.std(axis=1).replace(0, 1.0)
    return (df_ts.sub(means, axis=0)).div(stds, axis=0)

X_ts = row_zscore(pop_ts)   # DataFrame, index=polygon IDs, columns=years

# If we append FUN features, scale them (so they don't dominate)
if include_fun_as_feature:
    fun_scaled = (fun_df - fun_df.mean()) / (fun_df.std().replace(0,1))
    X_full = pd.concat([X_ts, fun_scaled], axis=1).values
else:
    X_full = X_ts.values

# -----------------------------
# 3. PCA to reduce dimensionality (optional but recommended for KMeans)
# -----------------------------
# Use PCA to retain 95% variance; for KMeans on Euclidean space this helps.
pca = PCA(n_components=0.95, svd_solver="full", random_state=0)
pcs = pca.fit_transform(X_full)
print("PCA -> components:", pcs.shape[1], "explained:", pca.explained_variance_ratio_.cumsum()[-1])

# -----------------------------
# Utilities for k selection
# -----------------------------
def compute_silhouette_curve(X_emb, k_range, random_state=0):
    sils = []
    km_models = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_emb)
        sil = silhouette_score(X_emb, labels)
        sils.append(sil)
        km_models[k] = (km, labels)
    return np.array(sils), km_models

# Simple gap statistic implementation (Monte Carlo)
def gap_statistic(X_emb, refs=5, K=range(1,7), random_state=0):
    from sklearn.metrics import pairwise_distances
    rng = np.random.RandomState(random_state)
    shape = X_emb.shape
    # bounding box for uniform reference
    mins = X_emb.min(axis=0)
    maxs = X_emb.max(axis=0)
    gaps = []
    dispersions = []
    Wks = []
    for k in K:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(X_emb)
        labels = km.labels_
        # compute within-cluster dispersion
        wk = 0.0
        for j in range(k):
            cluster_j = X_emb[labels == j]
            if cluster_j.shape[0] > 1:
                wk += np.sum(pairwise_distances(cluster_j, cluster_j)**2) / (2.0 * cluster_j.shape[0])
        Wks.append(np.log(wk+1e-9))
        # reference
        ref_w = []
        for i in range(refs):
            X_ref = rng.uniform(mins, maxs, size=shape)
            km_ref = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            km_ref.fit(X_ref)
            labels_ref = km_ref.labels_
            wk_ref = 0.0
            for j in range(k):
                cj = X_ref[labels_ref == j]
                if cj.shape[0] > 1:
                    wk_ref += np.sum(pairwise_distances(cj, cj)**2) / (2.0 * cj.shape[0])
            ref_w.append(np.log(wk_ref+1e-9))
        gaps.append(np.mean(ref_w) - np.log(wk+1e-9))
        dispersions.append(np.std(ref_w) * np.sqrt(1 + 1.0/refs))
    return np.array(gaps), np.array(Wks), np.array(dispersions)

# Stability via bootstrap + adjusted rand index
def cluster_stability(X_emb, k, n_boot=25, sample_frac=0.8, random_state=0):
    rng = np.random.RandomState(random_state)
    base_km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    base_labels = base_km.fit_predict(X_emb)
    ARIs = []
    for i in range(n_boot):
        idx = rng.choice(np.arange(X_emb.shape[0]), size=int(sample_frac*X_emb.shape[0]), replace=False)
        km = KMeans(n_clusters=k, random_state=rng.randint(0,10000), n_init=10)
        labels_sub = km.fit_predict(X_emb[idx])
        # map base labels to subset
        base_sub = base_labels[idx]
        ARIs.append(adjusted_rand_score(base_sub, labels_sub))
    return np.mean(ARIs), np.std(ARIs)

# -----------------------------
# 4. Explore k
# -----------------------------
K_range = range(2, 8)
sil_vals, km_models = compute_silhouette_curve(pcs, K_range)

plt.figure(figsize=(6,4))
plt.plot(list(K_range), sil_vals, marker='o')
plt.xlabel("k"); plt.ylabel("Silhouette score"); plt.title("Silhouette curve")
plt.grid(True); plt.show()

# Gap statistic
gaps, Wks, sds = gap_statistic(pcs, refs=10, K=range(1,8), random_state=0)
plt.figure(figsize=(6,4))
Ks = np.arange(1,8)
plt.plot(Ks, gaps, marker='o')
plt.xlabel("k"); plt.ylabel("Gap statistic"); plt.title("Gap statistic")
plt.grid(True); plt.show()

# Stability check for candidate ks
stabilities = {}
for k in K_range:
    m, s = cluster_stability(pcs, k, n_boot=30, sample_frac=0.75, random_state=0)
    stabilities[k] = (m, s)
    print(f"k={k}, stability (mean ARI)={m:.3f} (+/-{s:.3f})")

# -----------------------------
# 5. Choose final k and clustering method
# -----------------------------
# Heuristics to choose k:
# - Silhouette peak(s)
# - Gap statistic local maximum(s)
# - Stability (higher ARI)
# - Interpretability with FUN1_2025

# Example: pick the k with highest silhouette
best_k = K_range[np.argmax(sil_vals)]
print("best_k by silhouette:", best_k)

# Or you can manually set:
best_k = 3

# Cluster using KMeans on PCA (fast + interpretable)
final_k = int(best_k)
kmeans = KMeans(n_clusters=final_k, random_state=0, n_init=20)
labels = kmeans.fit_predict(pcs)

# Optionally: DTW-based clustering (uncomment if tslearn installed)
# if tslearn_available:
#     ts_km = TimeSeriesKMeans(n_clusters=final_k, metric="dtw", random_state=0, n_init=2)
#     # ts_km expects shape (n_samples, n_timestamps, 1)
#     ts_input = X_ts.values.reshape((X_ts.shape[0], X_ts.shape[1], 1))
#     labels_dtw = ts_km.fit_predict(ts_input)
#     labels = labels_dtw  # override if you prefer DTW

# -----------------------------
# 6. Summarize & visualize clusters
# -----------------------------
res = pd.DataFrame({
    "label": X_ts.index,
    "cluster": labels,
    "FUN1_2025": meta["FUN1_2025"].values
}, index=X_ts.index)

# 6a. PCA scatter with clusters
plt.figure(figsize=(10,6))
sc = plt.scatter(pcs[:,0], pcs[:,1], c=labels, cmap="tab10", s=30, alpha=0.9, edgecolor="k")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.title(f"PCA scatter (k={final_k})")
plt.grid(True)
# if have_adjusttext:
#     txts = []
#     for i, idx in enumerate(X_ts.index):
#         txts.append(plt.text(pcs[i,0], pcs[i,1], str(idx), fontsize=6))
#     adjust_text(txts, arrowprops=dict(arrowstyle="-", lw=0.3))
plt.show()

# 6b. Cluster sizes and FUN1 composition
summary = res.groupby("cluster").agg(
    n_polygons = ("cluster", "size"),
    FUN_mode = ("FUN1_2025", lambda s: s.mode().iloc[0] if len(s.mode())>0 else np.nan)
)
print(summary)

# 6c. Plot cluster centroids (mean trajectory) with members
for c in range(final_k):
    members = X_ts.loc[labels == c]
    fig, ax = plt.subplots(1,1, figsize=(8,3))
    for i,row in members.iterrows():
        ax.plot(years, row.values, color="k", alpha=0.08)
    centroid = members.mean(axis=0)
    ax.plot(years, centroid.values, color="red", lw=2, label="centroid")
    ax.set_title(f"Cluster {c} (n={len(members)})")
    ax.set_xlabel("Year")
    ax.set_ylabel("Row-wise z-scored pop density")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.show()

# 6d. (Optional) Map clusters if geometry present
if geom is not None:
    g = gp.GeoDataFrame({"cluster": labels}, geometry=geom.values, index=geom.index)
    # merge with other attributes if needed
    g["cluster"] = g["cluster"].astype(int)
    # plot
    fig, ax = plt.subplots(1,1, figsize=(8,8))
    g.plot(column="cluster", ax=ax, categorical=True, legend=True, cmap="tab10", linewidth=0.1)
    ax.set_title("Clusters mapped to polygons")
    plt.show()

# Save cluster assignments back to the original large DF (if desired)
# df.loc[pop_ts.index, "cluster"] = labels
# df.to_file(...)

# -----------------------------
# 7. Next steps & interpretation
# -----------------------------
# - Inspect cluster FUN1_2025 distributions: which municipality types are in each cluster?
# - Compute metrics per cluster (growth rates, recent decades, total pop change)
# - If interpretability matters, examine a few exemplar polygons from each cluster
# - If temporal misalignment matters, consider DTW / TimeSeriesKMeans
