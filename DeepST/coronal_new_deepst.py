import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('../DeepST')
import os 
from DeepST import run
import matplotlib.pyplot as plt
from pathlib import Path
import scanpy as sc
import community as louvain
import pandas as pd
import anndata as ad 
import time
import psutil
import gc
import json
import torch

def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

data_path = "../../RAW_SLICE/coronal/"
data_name_list = ['FFPE', 'DAPI', 'Normal']
save_path = "../Results" 
n_domains = 12
deepen = run(save_path = save_path, 
	task = "Integration",
	pre_epochs = 500, 
	epochs = 500, 
	use_gpu = True,
)
augement_data_list = []
graph_list = []
total_cells = 0

for i in range(len(data_name_list)):
    print(f"Processing {data_name_list[i]}...")
    adata = deepen._get_adata(platform="Visium", data_path=data_path, data_name=data_name_list[i])
    adata.var_names_make_unique()
    adata = deepen._get_image_crop(adata, data_name=data_name_list[i])
    adata = deepen._get_augment(adata, spatial_type="LinearRegress")    
    graph_dict = deepen._get_graph(adata.obsm["spatial"], distType="KDTree")
    graph_list.append(graph_dict)
    augement_data_list.append(adata)
    total_cells += adata.n_obs

print(f"Total cells across all datasets: {total_cells}")

torch.cuda.empty_cache()

multiple_adata, multiple_graph = deepen._get_multiple_adata(adata_list=augement_data_list, data_name_list=data_name_list, graph_list=graph_list)
data = deepen._data_process(multiple_adata, pca_n_comps=200)


# =============== DeepST training ===============
print("Starting DeepST Core Training Benchmark...")
gc.collect()
torch.cuda.empty_cache()

memory_before = get_memory_usage()
training_start_time = time.time()

print("Training DeepST model...")
deepst_embed = deepen._fit(
    data = data,
    graph_dict = multiple_graph,
    domains = multiple_adata.obs["batch"].values,
    n_domains = len(data_name_list)
)

training_end_time = time.time()
training_time = training_end_time - training_start_time
memory_after = get_memory_usage()
memory_used = memory_after - memory_before

print("Training completed!")
benchmark_results = {
    'method_name': 'DeepST',
    'training_time_seconds': training_time,
    'training_time_minutes': training_time / 60,
    'training_time_hours': training_time / 3600,
    'memory_usage_mb': memory_used,
    'memory_usage_gb': memory_used / 1024,
    'total_cells': total_cells,
    'final_cells': multiple_adata.n_obs,
    'total_genes': multiple_adata.n_vars,
    'embedding_dim': deepst_embed.shape[1],
    'n_datasets': len(data_name_list),
    'pre_epochs': 500,
    'epochs': 500,
    'timestamp': pd.Timestamp.now().isoformat()
}


multiple_adata.obsm["DeepST_embed"] = deepst_embed
multiple_adata = deepen._get_cluster_data(multiple_adata, n_domains=n_domains, priori = True)
multiple_adata.obs.rename(columns={'batch_name': 'new_batch'}, inplace=True)
multiple_adata.write(f"{save_path}/multiple_adata_coronal_notruth.h5ad")

with open(f"{save_path}/deepst_benchmark_coronal.json", "w") as f:
    json.dump(benchmark_results, f, indent=2)

adata = ad.read_h5ad(f"{save_path}/multiple_adata_coronal_notruth.h5ad")
print("Adding ground truth labels...")
adata.obs["new_batch"] = adata.obs["new_batch"].astype(str)
adata.obs.index = adata.obs["new_batch"].str.upper() + '-' + adata.obs.index
adata.obs.index = adata.obs.index.str.replace(r'(-\d+)+$', '-1', regex=True)
ffpe_df = pd.read_csv('../../RAW_SLICE/coronal/FFPE/FFPE_truth.csv', index_col='Unnamed: 0')
dapi_df = pd.read_csv('../../RAW_SLICE/coronal/DAPI/DAPI_truth.csv', index_col='Unnamed: 0')
normal_df = pd.read_csv('../../RAW_SLICE/coronal/Normal/Normal_truth.csv', index_col='Unnamed: 0')
ffpe_df.index = ffpe_df.index.astype(str)
dapi_df.index = dapi_df.index.astype(str)
normal_df.index = normal_df.index.astype(str)
# Replace with the actual column name that holds the ground truth data
ffpe_ground_truth = ffpe_df['celltype_new'].to_dict()  
dapi_ground_truth = dapi_df['celltype_new'].to_dict()  
normal_ground_truth = normal_df['celltype_new'].to_dict()  
combined_ground_truth = {**ffpe_ground_truth, **dapi_ground_truth, **normal_ground_truth}
adata.obs['celltype'] = adata.obs.index.map(combined_ground_truth)
adata = adata[~pd.isnull(adata.obs['celltype'])]
print("Ground truth labels added:")

adata.write(f"{save_path}/multiple_adata_coronal.h5ad")
