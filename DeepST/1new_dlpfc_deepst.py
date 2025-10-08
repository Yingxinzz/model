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

data_path = "../../RAW_SLICE/DLPFC/" 
data_name_list = ['151673', '151674', '151675', '151676']
save_path = "../Results" 
n_domains = 7
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
    df_meta = pd.read_csv(data_path + data_name_list[i] + '/metadata.tsv', sep='\t')
    df_meta_layer = df_meta['layer_guess']
    adata.obs.loc[adata.obs['new_batch'] == data_name_list[i], 'ground_truth'] = df_meta_layer.values
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]
    total_cells += adata.n_obs
    graph_dict = deepen._get_graph(adata.obsm["spatial"], distType="KDTree")
    graph_list.append(graph_dict)
    augement_data_list.append(adata)

print(f"data_name_list length: {len(data_name_list)}")
print(f"graph_list length: {len(graph_list)}")

torch.cuda.empty_cache()

multiple_adata, multiple_graph = deepen._get_multiple_adata(adata_list=augement_data_list, data_name_list=data_name_list, graph_list=graph_list)
data = deepen._data_process(multiple_adata, pca_n_comps=200)

# =============== DeepST training ===============
print("Starting core training benchmarking...")

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

# =============== Saving benchmarking results ===============
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
multiple_adata.obs['celltype'] = multiple_adata.obs['ground_truth'].astype('category')
multiple_adata.write("../Results/multiple_adata1.h5ad")

with open("../Results/deepst_benchmark1.json", "w") as f:
    json.dump(benchmark_results, f, indent=2)

