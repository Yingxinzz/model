import scanpy as sc
import pandas as pd
import os

file_fold = '../../RAW_SLICE/DLPFC/'
datasets = ['151673', #'151674', '151675', '151676',
            '151669', #'151670','151671', '151672',
            '151507', #'151508', '151509', '151510'
            ]
save_path = "../results_dlpfc_all" 
data_list=[]
Batch_list = []
####scanpy=1.9.1 preprocessing
for dataset in datasets:  
    adata = sc.read_visium(file_fold + dataset, count_file=dataset + '_filtered_feature_bc_matrix.h5', load_images=True)
    adata.var_names_make_unique()
    Ann_df = pd.read_csv(os.path.join(file_fold+dataset, dataset + '_truth.txt'), sep='\t', header=None, index_col=0)
    Ann_df.columns = ['Ground Truth']
    Ann_df[Ann_df.isna()] = "unknown"
    adata.obs['celltype'] = Ann_df.loc[adata.obs_names, 'Ground Truth'].astype('category')
    adata = adata[adata.obs['celltype']!='unknown']
    adata.X = adata.X.astype('float32')
    if 'spatial' in adata.obsm:
        adata.obsm['spatial'] = adata.obsm['spatial'].astype('float32')
    min_gene = 20
    min_cell = 20
    sc.pp.filter_cells(adata, min_genes=min_gene)
    sc.pp.filter_genes(adata, min_cells=min_cell)
    sc.pp.normalize_total(adata, target_sum=1e4)  
    sc.pp.log1p(adata)
    h5ad_path = os.path.join(save_path, f"{dataset}.h5ad")
    adata.write_h5ad(h5ad_path)  
    data_list.append(h5ad_path)
    Batch_list.append(adata)
    print(f"Saved {h5ad_path}")

####  conda activate Spatialign   
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

print("Starting data preprocessing...")

data_list=[
    '../results_dlpfc_all/151673.h5ad',
    '../results_dlpfc_all/151669.h5ad',
    '../results_dlpfc_all/151507.h5ad'
]

dataset_names = ['151673', '151669', '151507']

print("Loading datasets for cell counting...")
total_cells = 0
for i, data_path in enumerate(data_list):
    temp_adata = sc.read_h5ad(data_path)
    cells_count = temp_adata.n_obs
    total_cells += cells_count
    print(f"  {dataset_names[i]}: {cells_count} cells")
    del temp_adata

print(f"Total cells across datasets: {total_cells}")

model = Spatialign(
    *data_list,
    batch_key='batch_name',
    is_norm_log=True,
    is_scale=False,
    n_neigh=15,
    is_undirected=True,
    latent_dims=100,
    seed=42,
    gpu=0,
    save_path="../results_dlpfc_all/",
    is_verbose=False)

merge_data = AnnData.concatenate(*model.dataset.data_list)

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

correct1 = sc.read_h5ad("../results_dlpfc_all/res/correct_data0.h5ad")
correct2 = sc.read_h5ad("../results_dlpfc_all/res/correct_data1.h5ad")
correct3 = sc.read_h5ad("../results_dlpfc_all/res/correct_data2.h5ad")
merge_data = correct1.concatenate(correct2, correct3)
new_batch_1 = merge_data.obs["batch"].isin(['0'])
new_batch_2 = merge_data.obs["batch"].isin(['1'])
new_batch_3 = merge_data.obs["batch"].isin(['2'])
merge_data.obs["sample_name"] = list(sum(new_batch_1)*['Sample 1'])+list(sum(new_batch_2)*['Sample 2'])+list(sum(new_batch_3)*['Sample 3'])
merge_data.obs["new_batch"] = merge_data.obs["sample_name"].astype('category')

# =============== 保存基准测试结果 ===============
benchmark_results = {
    'method_name': 'Spatialign',
    'training_time_seconds': training_time,
    'training_time_minutes': training_time / 60,
    'training_time_hours': training_time / 3600,
    'memory_usage_mb': memory_used,
    'memory_usage_gb': memory_used / 1024,
    'total_cells': total_cells,
    'final_cells': merge_data.n_obs,
    'embedding_dim': merge_data.obsm["correct"].shape[1],
    'n_datasets': len(data_list),
    'datasets': dataset_names,
    'device': 'GPU 0',
    'random_seed': 42,
    'latent_dims': 100,
    'timestamp': pd.Timestamp.now().isoformat()
}

batch_mapping = {
    '0': 'sample1',
    '1': 'sample2',
    '2': 'sample3',
}
merge_data.obs['new_batch'] = merge_data.obs['batch'].replace(batch_mapping)
merge_data.X = np.nan_to_num(merge_data.X, nan=0.0)
merge_data.obs['new_batch'] = merge_data.obs['new_batch'].astype('category')

print("Performing clustering...")
sc.pp.scale(merge_data)
X = merge_data.obsm['correct']
n_components = 7 
gmm = GaussianMixture(n_components=n_components, random_state=42)
merge_data.obs['mclust'] = gmm.fit_predict(X)
merge_data.obs["mclust"] = merge_data.obs["mclust"].astype("category")
merge_data.write("../results_dlpfc_all/multiple_adata_all.h5ad")

with open("../results_dlpfc_all/spatialign_benchmark_all.json", "w") as f:
    json.dump(benchmark_results, f, indent=2)

