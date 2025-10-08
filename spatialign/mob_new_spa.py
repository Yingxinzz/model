#Import packages
import sys
sys.path.append('../')
import os
import scanpy as sc
from spatialign import Spatialign
from warnings import filterwarnings
from anndata import AnnData
import h5py
import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
filterwarnings("ignore")
import torch
torch.set_default_dtype(torch.float32)
import time
import psutil
import gc
import json
from sklearn.mixture import GaussianMixture


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

data_list = [
    "../../RAW_SLICE/mob/10X.h5ad",
    "../../RAW_SLICE/mob/BGI.h5ad", 
    "../../RAW_SLICE/mob/SlideV2.h5ad"
]
dataset_names = ['10X', 'BGI', 'SlideV2']
total_cells = 0
for i, data_path in enumerate(data_list):
    temp_adata = sc.read_h5ad(data_path)
    cells_count = temp_adata.n_obs
    total_cells += cells_count
    del temp_adata  

print(f"Total cells across datasets: {total_cells}")

model = Spatialign(
    *data_list,
    batch_key='batch',
    is_norm_log=True,
    is_scale=False,
    n_neigh=15,
    is_undirected=True,
    latent_dims=100,
    seed=42,
    gpu=0,
    save_path="../results_mob",
    is_verbose=False
)
raw_merge = AnnData.concatenate(*model.dataset.data_list,
    batch_key='batch',
    batch_categories=[ '10X','BGI','SlideV2']
)

# =============== spatiAlign training ===============
print("\nStarting core training benchmarking...")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

memory_before = get_memory_usage()
training_start_time = time.time()

print("Training spatialign model...")
model.train(0.05, 1, 0.1)
model.alignment()

training_end_time = time.time()
training_time = training_end_time - training_start_time
memory_after = get_memory_usage()
memory_used = memory_after - memory_before
print("Training completed!")
correct1 = sc.read_h5ad("../results_mob/res/correct_data0.h5ad")
correct2 = sc.read_h5ad("../results_mob/res/correct_data1.h5ad")
correct3 = sc.read_h5ad("../results_mob/res/correct_data2.h5ad")
merge_data = correct1.concatenate(correct2, correct3)

benchmark_results = {
    'method_name': 'Spatialign',
    'training_time_seconds': training_time,
    'training_time_minutes': training_time / 60,
    'training_time_hours': training_time / 3600,
    'memory_usage_mb': memory_used,
    'memory_usage_gb': memory_used / 1024,
    'total_cells': total_cells,
    'final_cells': merge_data.n_obs,
    'total_genes': merge_data.n_vars,
    'embedding_dim': merge_data.obsm["correct"].shape[1],
    'n_datasets': len(data_list),
    'datasets': dataset_names,
    'device': 'GPU 0',
    'random_seed': 42,
    'latent_dims': 100,
    'timestamp': pd.Timestamp.now().isoformat()
}

# sc.pp.neighbors(merge_data, use_rep='correct',random_state=666)
# sc.tl.louvain(merge_data, random_state=666, key_added="louvain", resolution=1.15)
sc.pp.scale(merge_data)
X = merge_data.obsm['correct']
n_components = 8
gmm = GaussianMixture(n_components=n_components, random_state=42)
merge_data.obs['mclust'] = gmm.fit_predict(X)
merge_data.obs["mclust"] = merge_data.obs["mclust"].astype("category")
merge_data.obs["celltype"] = merge_data.obs["celltype"].astype("category")
merge_data.X = np.nan_to_num(merge_data.X, nan=0.0)

batch_mapping = {
    '0': '10X',
    '1': 'BGI',
    '2': 'SlideV2'
}
merge_data.X = np.nan_to_num(merge_data.X, nan=0.0)
merge_data.obs['new_batch'] = merge_data.obs['batch'].replace(batch_mapping)
merge_data.obs['new_batch'] = merge_data.obs['new_batch'].astype('category')
merge_data.write("../results_mob/multiple_adata.h5ad")

with open("../results_mob/spatialign_benchmark_mob.json", "w") as f:
    json.dump(benchmark_results, f, indent=2)

