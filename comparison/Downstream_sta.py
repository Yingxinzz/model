import matplotlib.lines as mlines
import seaborn as sns
from sklearn import preprocessing
from matplotlib.colors import ListedColormap
import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
from sklearn.metrics import adjusted_rand_score
from scipy.optimize import linear_sum_assignment

adata_raw = sc.read_h5ad("../DATA_RAW/raw_adata_12.h5ad")
label_mapping = {"0": "1", "1": "2", "2": "3", "3": "4", "4": "5", "5": "6", "6": "7"}
adata_raw.obs["mclust"] = adata_raw.obs["mclust"].astype(str).map(label_mapping)
adata = sc.read_h5ad("../results/workflow/12_staligner_DLPFC.h5ad")
key = "new_batch"
celltype_key = "celltype"
cluster_key = "mclust"
sc.pp.neighbors(adata_raw, use_rep="X_pca", random_state=22)
sc.tl.umap(adata_raw, random_state=22)
sc.pp.neighbors(adata, use_rep="STAligner", random_state=22)
sc.tl.umap(adata, random_state=22)
fig, axs = plt.subplots(2, 6, figsize=(36, 12))
axs = axs.flatten()
spot_size = 100
umap_titles_raw = ["uncorrected", "RAW - Ground Truth", "RAW - clusters"]
umap_colors_raw = [key, celltype_key, cluster_key]
for i, (t, c) in enumerate(zip(umap_titles_raw, umap_colors_raw)):
    sc.pl.umap(adata_raw, color=c, title=t, ax=axs[i], show=False)

sc.tl.paga(adata_raw, groups=celltype_key)
sc.pl.paga(adata_raw, color=celltype_key, title="PAGA (raw)", ax=axs[3], show=False)

sc.pl.spatial(
    adata_raw,
    color=celltype_key,
    ax=axs[4],
    spot_size=spot_size,
    cmap="viridis",
    show=False,
)
sc.pl.spatial(
    adata_raw,
    color=cluster_key,
    ax=axs[5],
    spot_size=spot_size,
    cmap="viridis",
    show=False,
)

umap_titles_corr = ["corrected", "Corrected - Ground Truth", "Corrected - clusters"]
umap_colors_corr = [key, celltype_key, cluster_key]
for j, (t, c) in enumerate(zip(umap_titles_corr, umap_colors_corr)):
    sc.pl.umap(adata, color=c, title=t, ax=axs[6 + j], show=False)

sc.tl.paga(adata, groups=celltype_key)
sc.pl.paga(adata, color=celltype_key, title="PAGA (corrected)", ax=axs[9], show=False)
sc.pl.spatial(
    adata,
    color=celltype_key,
    title="Corrected spatial truth",
    ax=axs[10],
    spot_size=spot_size,
    cmap="viridis",
    show=False,
)
sc.pl.spatial(
    adata,
    color=cluster_key,
    title="Corrected spatial cluster",
    ax=axs[11],
    spot_size=spot_size,
    cmap="viridis",
    show=False,
)
plt.tight_layout()
plt.savefig("../results/workflow/part1_umap_paga_spatial.png", dpi=300, bbox_inches="tight")


#########part2 metrics
df = pd.read_csv("../results/workflow/model_performance_results.csv", index_col=0)
methods = ["RAW", "STAligner"]
fig, axes = plt.subplots(1, 3, figsize=(22, 6), sharey=False)
metrics = ["SCS"]
data = df.loc[methods, metrics].abs().T
x = np.arange(len(metrics))
width = 0.8 / len(methods)

for i, method in enumerate(methods):
    axes[0].bar(x + i * width, data.loc[metrics, method], width, label=method)

axes[0].set_xticks(x + width * (len(methods) - 1) / 2)
axes[0].set_xticklabels(metrics, rotation=30, ha="right")
axes[0].set_ylabel("Score")
axes[0].legend()

metrics = ["MoranI", "GearyC"]
data = df.loc[methods, metrics].T
x = np.arange(len(metrics))
width = 0.8 / len(methods)

for i, method in enumerate(methods):
    axes[1].bar(x + i * width, data.loc[metrics, method], width, label=method)

axes[1].set_xticks(x + width * (len(methods) - 1) / 2)
axes[1].set_xticklabels(metrics, rotation=30, ha="right")
axes[1].legend()

metrics = ["cLISI", "ASW_domain", "GC", "iLISI", "kBET", "ASW_batch", "ARI"]
data = df.loc[methods, metrics].T
x = np.arange(len(metrics))
width = 0.8 / len(methods)

for i, method in enumerate(methods):
    axes[2].bar(x + i * width, data.loc[metrics, method], width, label=method)

axes[2].set_xticks(x + width * (len(methods) - 1) / 2)
axes[2].set_xticklabels(metrics, rotation=30, ha="right")
axes[2].legend()

plt.tight_layout()
plt.savefig(
    "../results/workflow/part2_metrics_comparison_vertical.png", dpi=300, bbox_inches="tight"
)


#########part3 heatmap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

sc.tl.rank_genes_groups(
    adata, groupby="celltype", method="wilcoxon", key_added="rank_celltype"
)
sc.tl.rank_genes_groups(
    adata, groupby="mclust", method="wilcoxon", key_added="rank_mclust"
)
sc.tl.rank_genes_groups(
    adata_raw, groupby="mclust", method="wilcoxon", key_added="rank_mclust"
)


def get_top_markers(adata, key, n=5):
    groups = adata.uns[key]["names"].dtype.names
    markers = {}
    for g in groups:
        top_genes = adata.uns[key]["names"][g][:n].tolist()
        markers[g] = top_genes
    return markers


top5_celltype = get_top_markers(adata, "rank_celltype", n=5)
top5_mclust = get_top_markers(adata, "rank_mclust", n=5)
top5_mclust_raw = get_top_markers(adata_raw, "rank_mclust", n=5)


def get_color_map(categories):
    palette = sns.color_palette("tab20", len(categories))
    return dict(zip(categories, palette))


fig, axes = plt.subplots(1, 3, figsize=(24, 8))

# Heatmap 1: celltype top5 markers (corrected)
top_genes = sum(top5_celltype.values(), [])
groups = adata.obs["celltype"].unique()
df = pd.DataFrame(index=top_genes)
for g in groups:
    X = adata[adata.obs["celltype"] == g, top_genes].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    df[g] = X.mean(axis=0)

df = (df - df.mean(axis=1).values[:, None]) / df.std(axis=1).values[:, None]
sns.heatmap(df, cmap="viridis", ax=axes[0], cbar_kws={"label": "Z-score"})

colors = get_color_map(groups)
for i, g in enumerate(df.columns):
    axes[0].add_patch(
        mpatches.Rectangle((i, -0.5), 1, 0.3, color=colors[g], clip_on=False)
    )

axes[0].set_xticklabels(df.columns, rotation=45)

# Heatmap 2: mclust top5 markers (corrected)
top_genes = sum(top5_mclust.values(), [])
groups = adata.obs["mclust"].unique()
df = pd.DataFrame(index=top_genes)
for g in groups:
    X = adata[adata.obs["mclust"] == g, top_genes].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    df[g] = X.mean(axis=0)

df = (df - df.mean(axis=1).values[:, None]) / df.std(axis=1).values[:, None]
sns.heatmap(df, cmap="viridis", ax=axes[1], cbar_kws={"label": "Z-score"})

colors = get_color_map(groups)
for i, g in enumerate(df.columns):
    axes[1].add_patch(
        mpatches.Rectangle((i, -0.5), 1, 0.3, color=colors[g], clip_on=False)
    )

axes[1].set_xticklabels(df.columns, rotation=45)

# Heatmap 3: mclust top5 markers (raw)
top_genes = list(set().union(*top5_mclust_raw.values()))
groups = adata_raw.obs["mclust"].unique()
df = pd.DataFrame(index=top_genes)
for g in groups:
    X = adata_raw[adata_raw.obs["mclust"] == g, top_genes].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    df[g] = X.mean(axis=0)

df = (df - df.mean(axis=1).values[:, None]) / df.std(axis=1).values[:, None]
sns.heatmap(df, cmap="viridis", ax=axes[2], cbar_kws={"label": "Z-score"})

colors = get_color_map(groups)
for i, g in enumerate(df.columns):
    axes[2].add_patch(
        mpatches.Rectangle((i, -0.5), 1, 0.3, color=colors[g], clip_on=False)
    )

axes[2].set_xticklabels(df.columns, rotation=45)

plt.tight_layout()
plt.savefig("../results/workflow/part3_heatmap_with_colorbar.png", dpi=300, bbox_inches="tight")


##########part4vilion
sc.tl.rank_genes_groups(adata, groupby="celltype", method="wilcoxon", key_added="truth")
rank_df = sc.get.rank_genes_groups_df(adata, group=None, key="truth")
rank_df.to_csv("../results/workflow/rank_genes_mclust_corrected.csv", index=False)
deg_sig = rank_df[
    (rank_df["pvals_adj"] < 0.05) & (rank_df["logfoldchanges"] > 0.25)
].copy()
deg_sig = deg_sig.sort_values(["group", "logfoldchanges"], ascending=[True, False])
selected_genes = []
cluster_top_genes = []
for cluster, sub_df in deg_sig.groupby("group"):
    for gene in sub_df["names"]:
        if gene not in selected_genes:
            selected_genes.append(gene)
            cluster_top_genes.append({"cluster": cluster, "gene": gene})
            break

cluster_top_genes = pd.DataFrame(cluster_top_genes)
cluster_top_genes.to_csv("../results/workflow/cluster_top_genes_corrected.csv", index=False)

genes_to_plot = ["CXCL14", "HPCAL1", "CARTPT", "PVALB", "PCP4", "KRT17", "PLP1"]
fig, ax_list = plt.subplots(3, len(genes_to_plot), figsize=(6 * len(genes_to_plot), 18))
ax_list = ax_list.flatten()

# ---- Ground Truth (celltype) ----
for i, gene in enumerate(genes_to_plot):
    sc.pl.violin(
        adata,
        keys=[gene],
        groupby="celltype",
        jitter=True,
        rotation=45,
        size=4,
        scale="width",
        ax=ax_list[i],
        show=False,
    )
    ax_list[i].set_title(f"{gene} (celltype)")

# ---- Corrected mclust ----
unique_mclust = adata.obs["mclust"].unique()
palette = sns.color_palette("Set1", n_colors=len(unique_mclust))
palette_dict = {str(mclust): color for mclust, color in zip(unique_mclust, palette)}

for i, gene in enumerate(genes_to_plot):
    sc.pl.violin(
        adata,
        keys=[gene],
        groupby="mclust",
        jitter=True,
        rotation=45,
        size=4,
        scale="width",
        ax=ax_list[len(genes_to_plot) + i],
        palette=palette_dict,
        show=False,
    )
    ax_list[len(genes_to_plot) + i].set_title(f"{gene} (Corrected)")

# ---- Raw mclust ----
for i, gene in enumerate(genes_to_plot):
    sc.pl.violin(
        adata_raw,
        keys=[gene],
        groupby="mclust",
        jitter=True,
        rotation=45,
        size=4,
        scale="width",
        ax=ax_list[2 * len(genes_to_plot) + i],
        show=False,
    )
    ax_list[2 * len(genes_to_plot) + i].set_title(f"{gene} (Raw)")

plt.tight_layout()
plt.savefig("../results/workflow/part4_violin.png", dpi=300)
