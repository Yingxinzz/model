import sys
sys.path.append('../')
import os
import torch
import scanpy as sc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
import multiprocessing as mp
from GraphST import GraphST
import time
import psutil
import gc
import json
import anndata as ad

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print("Starting data preprocessing...")
n_clusters = 5
datasets = ['151669', '151670', '151671', '151672']
file_fold = '../../RAW_SLICE/DLPFC/'
adatas = []
total_cells = 0

for dataset in datasets:
    print(f"Processing {dataset}...")
    adata = sc.read_visium(file_fold+dataset, count_file=dataset+'_filtered_feature_bc_matrix.h5', load_images=True)
    adata.var_names_make_unique()
    adata.obs['batch'] = dataset  # Add batch information
    df_meta = pd.read_csv(file_fold+dataset+'/metadata.tsv', sep='\t')
    df_meta_layer = df_meta['layer_guess']
    adata.obs.loc[adata.obs['batch'] == dataset, 'ground_truth'] = df_meta_layer.values
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]
    total_cells += adata.n_obs
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var['highly_variable']]
    adatas.append(adata)
    print(f"  {dataset}: {adata.n_obs} cells, {adata.n_vars} genes")

print(f"Total cells across datasets: {total_cells}")

adata = adatas[0].concatenate(adatas[1:], batch_key='batch')
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
training_time = training_end_time - training_start_time
memory_after = get_memory_usage()
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

print("\nContinuing with clustering (not timed)...")

from GraphST.utils import clustering
tool = 'mclust'  # mclust, leiden, and louvain
if tool == 'mclust':
    clustering(adata, n_clusters, method=tool)
elif tool in ['leiden', 'louvain']:
    clustering(adata, n_clusters, method=tool, start=0.1, end=2.0, increment=0.01)

batch_mapping = {
    '0': '151669',
    '1': '151670',
    '2': '151671',
    '3': '151672'
}

adata.obs['new_batch'] = adata.obs['batch'].replace(batch_mapping)
adata.obs['celltype'] = adata.obs['ground_truth'].astype('category')
adata.write("../results/DLPFC_adata2.h5ad")

with open("../results/graphst_benchmark2.json", "w") as f:
    json.dump(benchmark_results, f, indent=2)

