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
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

data_path = "../../RAW_SLICE/hbc"
data_name_list = ['section1', 'section2']
save_path = "../Results" 
n_domains = 10
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
    adata = deepen._get_adata(platform="Visium", data_path=data_path, data_name=data_name_list[i])
    adata = deepen._get_image_crop(adata, data_name=data_name_list[i])
    adata = deepen._get_augment(adata, spatial_type="LinearRegress")
    adata.obs['new_batch'] = data_name_list[i]
    graph_dict = deepen._get_graph(adata.obsm["spatial"], distType="KDTree")
    graph_list.append(graph_dict)
    augement_data_list.append(adata)
    total_cells += adata.n_obs

torch.cuda.empty_cache()

multiple_adata, multiple_graph = deepen._get_multiple_adata(adata_list=augement_data_list, data_name_list=data_name_list, graph_list=graph_list)
data = deepen._data_process(multiple_adata, pca_n_comps=200)

# =============== DeepST training ===============
print("Starting core training benchmarking...")

# Clear memory before training for fair comparison
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

# =============== Save Benchmark Results ===============
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

new_obs_names = []
for i, obs_name in enumerate(multiple_adata.obs_names):
    base_name = obs_name.split('-')[0]
    batch = multiple_adata.obs['batch'].iloc[i]
    suffix = '1' if batch == 0 else '2'
    new_obs_names.append(f"{base_name}-{suffix}")

multiple_adata.obs_names = new_obs_names
multiple_adata.obs['celltype'] = 'Unknown'
for dataset in data_name_list:
    meta = pd.read_csv(os.path.join(data_path, dataset, "metadata.csv"), index_col=0)
    common_barcodes = multiple_adata.obs_names.intersection(meta.index)
    if len(common_barcodes) > 0:
        multiple_adata.obs.loc[common_barcodes, 'celltype'] = meta.loc[common_barcodes, 'celltype']

multiple_adata.write("../Results/multiple_adata_hbc.h5ad")
with open("../Results/deepst_benchmark_hbc.json", "w") as f:
    json.dump(benchmark_results, f, indent=2)

