#######conda activate STAligner
import scanpy as sc
import pandas as pd
import os
from sklearn.mixture import GaussianMixture

datasets = [
    "151673",
    "151674",
    "151675",
    "151676",
    "151669",
    "151670",
    "151671",
    "151672",
    "151507",
    "151508",
    "151509",
    "151510",
]
file_fold = "../RAW_SLICE/DLPFC/"
adatas = []
for dataset in datasets:
    adata = sc.read_visium(
        os.path.join(file_fold, dataset),
        count_file=dataset + "_filtered_feature_bc_matrix.h5",
        load_images=True,
    )
    adata.var_names_make_unique()
    Ann_df = pd.read_csv(
        os.path.join(file_fold, dataset, dataset + "_truth.txt"),
        sep="\t",
        header=None,
        index_col=0,
    )
    Ann_df.columns = ["Ground Truth"]
    Ann_df[Ann_df.isna()] = "unknown"
    Ann_df.index = [x + "_" + dataset for x in Ann_df.index]
    adata.obs_names = [x + "_" + dataset for x in adata.obs_names]
    adata.obs["celltype"] = Ann_df.loc[adata.obs_names, "Ground Truth"].astype(
        "category"
    )
    adata = adata[adata.obs["celltype"] != "unknown"]
    coords = pd.DataFrame(adata.obsm["spatial"], index=adata.obs_names)
    adata.obsm["spatial"] = coords.values
    adata.obs["batch"] = dataset
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var["highly_variable"]]
    adatas.append(adata)

raw_adata = adatas[0].concatenate(
    *adatas[1:], batch_key="batch", batch_categories=datasets
)
raw_adata.obs["batch_name"] = raw_adata.obs["batch"]
new_batch_1 = raw_adata.obs["batch_name"].isin(["151673", "151674", "151675", "151676"])
new_batch_2 = raw_adata.obs["batch_name"].isin(["151669", "151670", "151671", "151672"])
new_batch_3 = raw_adata.obs["batch_name"].isin(["151507", "151508", "151509", "151510"])

raw_adata.obs["new_batch"] = (
    list(sum(new_batch_1) * ["Sample 1"])
    + list(sum(new_batch_2) * ["Sample 2"])
    + list(sum(new_batch_3) * ["Sample 3"])
)
sc.tl.pca(raw_adata, n_comps=50)
X_pca = raw_adata.obsm["X_pca"]
gmm = GaussianMixture(n_components=7, random_state=42)
raw_adata.obs["mclust"] = gmm.fit_predict(X_pca)
raw_adata.obs["mclust"] = raw_adata.obs["mclust"].astype("category")
raw_adata.write("../DATA_RAW/raw_adata_12.h5ad")


import warnings

warnings.filterwarnings("ignore")
import sys

sys.path.append("../")
import STAligner
from STAligner import ST_utils
from STAligner.ST_utils import match_cluster_labels
import os
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
import scipy.linalg
from scipy.sparse import csr_matrix
import time
import psutil
import gc
import json
import torch


def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
used_device = device

total_cells = 0
Batch_list = []
adj_list = []
datasets = [
    "151673",
    "151674",
    "151675",
    "151676",
    "151669",
    "151670",
    "151671",
    "151672",
    "151507",
    "151508",
    "151509",
    "151510",
]
file_fold = "../RAW_SLICE/DLPFC/"
adatas = []

for dataset in datasets:
    print(f"   Processing dataset: {dataset}")
    adata = sc.read_visium(
        file_fold + dataset,
        count_file=dataset + "_filtered_feature_bc_matrix.h5",
        load_images=True,
    )
    adata.var_names_make_unique()
    # read the annotation
    Ann_df = pd.read_csv(
        os.path.join(file_fold + dataset, dataset + "_truth.txt"),
        sep="\t",
        header=None,
        index_col=0,
    )
    Ann_df.columns = ["Ground Truth"]
    Ann_df[Ann_df.isna()] = "unknown"
    adata.obs["Ground Truth"] = Ann_df.loc[adata.obs_names, "Ground Truth"].astype(
        "category"
    )
    # make spot name unique
    adata.obs_names = [x + "_" + dataset for x in adata.obs_names]
    adata.obs["batch"] = dataset  # Add batch information
    # Constructing the spatial network
    STAligner.Cal_Spatial_Net(adata, rad_cutoff=150)
    STAligner.Stats_Spatial_Net(adata)  # plot the number of spatial neighbors
    # Normalization
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var["highly_variable"]]
    adj_list.append(adata.uns["adj"])
    Batch_list.append(adata)
    total_cells += adata.n_obs
    print(f"   {dataset}: {adata.n_obs} cells processed")

adata_concat = ad.concat(Batch_list, label="slice_name", keys=datasets)
adata_concat.obs["celltype"] = adata_concat.obs["Ground Truth"].astype("category")
print("adata_concat.shape: ", adata_concat.shape)
new_batch_1 = adata_concat.obs["slice_name"].isin(
    ["151673", "151674", "151675", "151676"]
)
new_batch_2 = adata_concat.obs["slice_name"].isin(
    ["151669", "151670", "151671", "151672"]
)
new_batch_3 = adata_concat.obs["slice_name"].isin(
    ["151507", "151508", "151509", "151510"]
)
adata_concat.obs["batch_name"] = (
    list(sum(new_batch_1) * ["Sample 1"])
    + list(sum(new_batch_2) * ["Sample 2"])
    + list(sum(new_batch_3) * ["Sample 3"])
)
adata_concat.obs["new_batch"] = adata_concat.obs["batch_name"].astype("category")

# adj
adj_concat = np.asarray(adj_list[0].todense())
for batch_id in range(1, len(datasets)):
    adj_concat = scipy.linalg.block_diag(
        adj_concat, np.asarray(adj_list[batch_id].todense())
    )

adata_concat.uns["edgeList"] = np.nonzero(adj_concat)

# =============== STAligner training ===============
print("Starting STAligner Core Training Benchmark...")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

memory_before = get_memory_usage()
training_start_time = time.time()

print("Initializing STAligner model...")
adata_concat = STAligner.train_STAligner(
    adata_concat, verbose=True, knn_neigh=100, device=used_device
)
edge_list = [
    [left, right]
    for left, right in zip(
        adata_concat.uns["edgeList"][0], adata_concat.uns["edgeList"][1]
    )
]
adata_concat.uns["edgeList"] = edge_list

training_end_time = time.time()
memory_after = get_memory_usage()
training_time = training_end_time - training_start_time
memory_used = memory_after - memory_before

print("Training completed!")

benchmark_results = {
    "method_name": "STAligner",
    "training_time_seconds": training_time,
    "training_time_minutes": training_time / 60,
    "training_time_hours": training_time / 3600,
    "memory_usage_mb": memory_used,
    "memory_usage_gb": memory_used / 1024,
    "total_cells": total_cells,
    "total_genes": adata_concat.n_vars,
    "embedding_dim": adata_concat.obsm["STAligner"].shape[1],
    "n_datasets": len(datasets),
    "random_seed": 50,
    "hvg_genes": 5000,
    "knn_neigh": 100,
    "rad_cutoff": 150,
    "timestamp": pd.Timestamp.now().isoformat(),
    "device": str(used_device),
}

ST_utils.mclust_R(adata_concat, num_cluster=7, used_obsm="STAligner")
adata_concat = adata_concat[adata_concat.obs["celltype"] != "unknown"]
adata_concat.write("../results/workflow/12_staligner_DLPFC.h5ad")

with open("../results/workflow/12_staligner_DLPFC.json", "w") as f:
    json.dump(benchmark_results, f, indent=2)



###########conda activate deepst_env
import scib
import scanpy as sc
import anndata as ad
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from spatial_metrics import spatialbench

adata = ad.read_h5ad("../results/12_staligner_DLPFC.h5ad")
all_results = {
    "Model": ["STAligner"],
    "MoranI": [],
    "GearyC": [],
    "SCS": [],
    "cLISI": [],
    "ASW_celltype": [],
    "GC": [],
    "iLISI": [],
    "kBET": [],
    "ASW_batch": [],
    "ARI": [],
}
res1 = spatialbench.moran_geary_batch_correction(adata, batch_key="new_batch", top_n=3)
all_results["MoranI"].append(res1[0])
all_results["GearyC"].append(res1[1])
res = spatialbench.spatial_coherence_score(adata, annotation_key="mclust", degree=4)
all_results["SCS"].append(res)
sc.pp.neighbors(adata, use_rep="STAligner")
all_results["GC"].append(scib.me.graph_connectivity(adata, label_key="celltype"))
all_results["iLISI"].append(
    scib.me.ilisi_graph(
        adata, batch_key="new_batch", type_="embed", use_rep="STAligner"
    )
)
all_results["cLISI"].append(
    scib.me.clisi_graph(adata, label_key="celltype", type_="embed", use_rep="STAligner")
)
all_results["kBET"].append(
    scib.me.kBET(
        adata,
        batch_key="new_batch",
        label_key="celltype",
        type_="embed",
        embed="STAligner",
    )
)
all_results["ASW_batch"].append(
    scib.me.silhouette_batch(
        adata, batch_key="new_batch", label_key="celltype", embed="STAligner"
    )
)
all_results["ASW_domain"].append(
    scib.me.silhouette(adata, label_key="celltype", embed="STAligner")
)
all_results["ARI"].append(
    scib.me.ari(adata, cluster_key="mclust", label_key="celltype")
)
df1 = pd.DataFrame(all_results)
df1.to_csv("../results/12_dlpfc_model_performance_results.csv", index=False)

adata = ad.read_h5ad("../DATA_RAW/raw_adata_12.h5ad")
all_results = {
    "Model": ["RAW"],
    "MoranI": [],
    "GearyC": [],
    "SCS": [],
    "cLISI": [],
    "ASW_domain": [],
    "GC": [],
    "iLISI": [],
    "kBET": [],
    "ASW_batch": [],
    "ARI": [],
}
res1 = spatialbench.moran_geary_batch_correction(adata, batch_key="new_batch", top_n=3)
all_results["MoranI"].append(res1[0])
all_results["GearyC"].append(res1[1])
res = spatialbench.spatial_coherence_score(adata, annotation_key="mclust", degree=4)
all_results["SCS"].append(res)
sc.pp.neighbors(adata, use_rep="X_pca")
all_results["GC"].append(scib.me.graph_connectivity(adata, label_key="celltype"))
all_results["iLISI"].append(
    scib.me.ilisi_graph(adata, batch_key="new_batch", type_="embed", use_rep="X_pca")
)
all_results["cLISI"].append(
    scib.me.clisi_graph(adata, label_key="celltype", type_="embed", use_rep="X_pca")
)
all_results["kBET"].append(
    scib.me.kBET(
        adata,
        batch_key="new_batch",
        label_key="celltype",
        type_="embed",
        embed="X_pca",
    )
)
all_results["ASW_batch"].append(
    scib.me.silhouette_batch(
        adata, batch_key="new_batch", label_key="celltype", embed="X_pca"
    )
)
all_results["ASW_domain"].append(
    scib.me.silhouette(adata, label_key="celltype", embed="X_pca")
)
all_results["ARI"].append(
    scib.me.ari(adata, cluster_key="mclust", label_key="celltype")
)
df2 = pd.DataFrame(all_results)
df2.to_csv("../results/12_dlpfc_model_performance_results_raw.csv", index=False)

df_combined = pd.concat([df1, df2], axis=0, ignore_index=True)
print(df_combined.head())
df_combined.to_csv("../results/workflow/model_performance_results.csv", index=False)
