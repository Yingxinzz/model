import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('../')
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
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
used_device = device

total_cells = 0
Batch_list = []
adj_list = []
datasets = ['151669', '151670','151671', '151672']
file_fold = '../../RAW_SLICE/DLPFC/'
adatas = []

for dataset in datasets:   
    print(f"   Processing dataset: {dataset}")
    adata = sc.read_visium(file_fold+dataset, count_file=dataset+'_filtered_feature_bc_matrix.h5', load_images=True)
    adata.var_names_make_unique()
    
    # read the annotation
    Ann_df = pd.read_csv(os.path.join(file_fold+dataset, dataset + '_truth.txt'), sep='\t', header=None, index_col=0)
    Ann_df.columns = ['Ground Truth']
    Ann_df[Ann_df.isna()] = "unknown"
    adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'Ground Truth'].astype('category')
    
    # make spot name unique
    adata.obs_names = [x+'_'+dataset for x in adata.obs_names]
    adata.obs['batch'] = dataset  # Add batch information
    
    # Constructing the spatial network
    STAligner.Cal_Spatial_Net(adata, rad_cutoff=150)
    STAligner.Stats_Spatial_Net(adata) # plot the number of spatial neighbors
    
    # Normalization
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var['highly_variable']]
    
    adj_list.append(adata.uns['adj'])
    Batch_list.append(adata)
    total_cells += adata.n_obs
    print(f"   {dataset}: {adata.n_obs} cells processed")

adata_concat = ad.concat(Batch_list, label="slice_name", keys=datasets)
adata_concat.obs['celltype'] = adata_concat.obs['Ground Truth'].astype('category')
adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')
print(f'adata_concat.shape: {adata_concat.shape}')

# adj
adj_concat = np.asarray(adj_list[0].todense())
for batch_id in range(1, len(datasets)):
    adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))

adata_concat.uns['edgeList'] = np.nonzero(adj_concat)

# =============== STAligner training ===============
print("Starting core training benchmarking...")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

memory_before = get_memory_usage()
training_start_time = time.time()

print("Training STAligner model...")
adata_concat = STAligner.train_STAligner(adata_concat, verbose=True, knn_neigh=100, device=used_device)
edge_list = [[left, right] for left, right in zip(adata_concat.uns['edgeList'][0], adata_concat.uns['edgeList'][1])]
adata_concat.uns['edgeList'] = edge_list

training_end_time = time.time()
training_time = training_end_time - training_start_time
memory_after = get_memory_usage()
memory_used = memory_after - memory_before

print("Training completed!")

benchmark_results = {
    'method_name': 'STAligner',
    'training_time_seconds': training_time,
    'training_time_minutes': training_time / 60,
    'training_time_hours': training_time / 3600,
    'memory_usage_mb': memory_used,
    'memory_usage_gb': memory_used / 1024,
    'total_cells': total_cells,
    'total_genes': adata_concat.n_vars,
    'embedding_dim': adata_concat.obsm['STAligner'].shape[1],
    'n_datasets': len(datasets),
    'random_seed': 50,
    'hvg_genes': 5000,
    'knn_neigh': 100,
    'rad_cutoff': 150,
    'timestamp': pd.Timestamp.now().isoformat(),
    'device': str(used_device)
}

ST_utils.mclust_R(adata_concat, num_cluster=5, used_obsm='STAligner')
adata_concat = adata_concat[adata_concat.obs['celltype']!='unknown']
adata_concat.obs["new_batch"] = adata_concat.obs["batch_name"].astype("category")
adata_concat.write('../results/staligner_Sample2_DLPFC.h5ad')

with open("../results/staligner_benchmark2.json", "w") as f:
    json.dump(benchmark_results, f, indent=2)

