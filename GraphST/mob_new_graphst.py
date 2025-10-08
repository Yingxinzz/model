import os
import torch
import scanpy as sc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
import multiprocessing as mp
import sys
sys.path.append('../')
from GraphST import GraphST
import time
import psutil
import gc
import json
import anndata as ad 
import numpy as np
from scipy import sparse

def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print("Starting data preprocessing...")
n_clusters = 8
datasets = ['10X','BGI', 'SlideV2']
file_fold = "../../RAW_SLICE/mob/"
adatas = []
total_cells = 0

for dataset in datasets:
    print(f"Processing {dataset}...")
    adata = sc.read_h5ad(file_fold + dataset + '.h5ad')
    adata.var_names_make_unique()
    adata.obs['batch'] = dataset
    print(f"  Original {dataset}: {adata.n_obs} cells, {adata.n_vars} genes")
    
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var['highly_variable']]
    adatas.append(adata)
    total_cells += adata.n_obs
    print(f"  Processed {dataset}: {adata.n_obs} cells, {adata.n_vars} genes (HVG)")

print(f"Total cells across datasets: {total_cells}")
adata = adatas[0].concatenate(adatas[1:], batch_key='batch')
adata.X = sparse.csr_matrix(adata.X)
adata.X = adata.X.astype(np.float32)
print(f"Concatenated data shape: {adata.shape}")

# =============== GraphST training ===============

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

memory_before = get_memory_usage()
training_start_time = time.time()

model = GraphST.GraphST(adata, device=device, random_seed=50)
print("Training GraphST model...")
adata = model.train()

training_end_time = time.time()
memory_after = get_memory_usage()
training_time = training_end_time - training_start_time
memory_used = memory_after - memory_before

benchmark_results = {
    'method_name': 'GraphST',
    'training_time_seconds': training_time,
    'training_time_minutes': training_time / 60,
    'training_time_hours': training_time / 3600,
    'memory_usage_mb': memory_used,
    'memory_usage_gb': memory_used / 1024,
    'total_cells': total_cells,
    'final_cells': adata.n_obs,
    'total_genes': adata.n_vars,
    'embedding_dim': adata.obsm['emb'].shape[1],
    'n_datasets': len(datasets),
    'device': str(device),
    'random_seed': 50,
    'hvg_genes': 5000,
    'timestamp': pd.Timestamp.now().isoformat()
}

print("\nContinuing with clustering...")
batch_mapping = {'0':'10X', '1':  'BGI', '2': 'SlideV2'}
from GraphST.utils import clustering
clustering(adata, n_clusters, method='mclust')
adata.obs['new_batch'] = adata.obs['batch'].replace(batch_mapping)
adata.write("../results/mob_adata.h5ad") 

with open("../results/graphst_benchmark_mob.json", "w") as f:
    json.dump(benchmark_results, f, indent=2)


# def clean_data(adata, keep_pca=True, keep_spatial=True):
#     keep_keys = ["emb"]
#     if keep_pca:
#         keep_keys.append("emb_pca")
#     if keep_spatial:
#         keep_keys.append("spatial")
#     for key in list(adata.obsm.keys()):
#         if key not in keep_keys:
#             del adata.obsm[key]
#     for col in [c for c in adata.var.columns if not c.startswith("highly_variable")]:
#         del adata.var[col]
#     for key in ["neighbors", "pca"]:
#         if key in adata.uns:
#             del adata.uns[key]
#     return adata

# adata = clean_data(adata, keep_pca=True, keep_spatial=True)
# adata.write("../results/mob_adata_clean.h5ad")
