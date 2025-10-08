import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('../')
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import scipy.io
import matplotlib.pyplot as plt
import os
import sys
import STitch3D
import warnings
warnings.filterwarnings("ignore")
import time
import psutil
import gc
import json
import torch

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
used_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(1234)

mat = scipy.io.mmread("../GSE144136_GeneBarcodeMatrix_Annotated.mtx")
meta = pd.read_csv("../GSE144136_CellNames.csv", index_col=0)
meta.index = meta.x.values
group = [i.split('.')[1].split('_')[0] for i in list(meta.x.values)]
condition = [i.split('.')[1].split('_')[1] for i in list(meta.x.values)]
celltype = [i.split('.')[0] for i in list(meta.x.values)]
meta["group"] = group
meta["condition"] = condition
meta["celltype"] = celltype
genename = pd.read_csv("../GSE144136_GeneNames.csv", index_col=0)
genename.index = genename.x.values
adata_ref = ad.AnnData(X=mat.tocsr().T)
adata_ref.obs = meta
adata_ref.var = genename
adata_ref.var_names_make_unique()
adata_ref = adata_ref[adata_ref.obs.condition.values.astype(str)=="Control", :]

anno_df = pd.read_csv('../barcode_level_layer_map.tsv', sep='\t', header=None)
datasets = ['151673','151669','151507']
file_fold = '../../RAW_SLICE/DLPFC/'


adatas = []
total_cells = 0
for dataset in datasets:
    adata = sc.read_visium(file_fold + str(dataset), count_file=str(dataset) + '_filtered_feature_bc_matrix.h5', load_images=True)
    adata.var_names_make_unique()
    anno_df_slice = anno_df.iloc[anno_df[1].values.astype(str) == str(dataset)]
    anno_df_slice.columns = ["barcode", "slice_id", "layer"]
    anno_df_slice.index = anno_df_slice['barcode']
    adata.obs = adata.obs.join(anno_df_slice, how="left")
    adata = adata[adata.obs['layer'].notna()]
    adatas.append(adata)
    total_cells += adata.n_obs

adata_stitched = STitch3D.utils.align_spots(adatas, plot=True)

celltype_list_use = ['Astros_1', 'Astros_2', 'Astros_3', 'Endo', 'Micro/Macro',
                     'Oligos_1', 'Oligos_2', 'Oligos_3',
                     'Ex_1_L5_6', 'Ex_2_L5', 'Ex_3_L4_5', 'Ex_4_L_6', 'Ex_5_L5',
                     'Ex_6_L4_6', 'Ex_7_L4_6', 'Ex_8_L5_6', 'Ex_9_L5_6', 'Ex_10_L2_4']

adata, adata_basis = STitch3D.utils.preprocess(adata_stitched,
                                                  adata_ref,
                                                  celltype_ref=celltype_list_use,
                                                  sample_col="group",
                                                  slice_dist_micron=[10., 300.],
                                                  n_hvg_group=500)

# =============== STicth training ===============

gc.collect()

memory_before = get_memory_usage()
training_start_time = time.time()

print("Training STitch3D model...")
model = STitch3D.model.Model(adata, adata_basis)
model.train()

training_end_time = time.time()
training_time = training_end_time - training_start_time
memory_after = get_memory_usage()
memory_used = memory_after - memory_before

print("Training completed!")
save_path = "../results_dlpfc_all"
result = model.eval(adatas, save=True, output_path=save_path)


# =============== 保存基准测试结果 ===============
benchmark_results = {
    'method_name': 'STitch3D',
    'training_time_seconds': training_time,
    'training_time_minutes': training_time / 60,
    'training_time_hours': training_time / 3600,
    'memory_usage_mb': memory_used,
    'memory_usage_gb': memory_used / 1024,
    'total_cells': total_cells,
    'total_genes': adata.n_vars,
    'final_cells': model.adata_st.n_obs,
    'embedding_dim' : model.adata_st.obsm['latent'].shape[1],
    'n_datasets': len(datasets),
    'random_seed': 1234,
    'n_hvg_group': 500,
    'slice_dist_micron': [10., 300., 10.],
    'device': str(used_device),
    'timestamp': pd.Timestamp.now().isoformat()
}

from sklearn.mixture import GaussianMixture
np.random.seed(1234)
gm = GaussianMixture(n_components=7, covariance_type='tied', init_params='kmeans')
y = gm.fit_predict(model.adata_st.obsm['latent'], y=None)
model.adata_st.obs["GM"] = y
model.adata_st.obs["GM"].to_csv(os.path.join(save_path, "clustering_result.csv"))
adata.obs["new_batch"] = adata.obs["slice_id"].astype(str)
adata.obs['new_batch'] = adata.obs['new_batch'].str.replace('.0', '', regex=False)
adata.obs['cluster'] = adata.obs['GM'].astype('category')
adata.obs['celltype'] = adata.obs['layer'].astype('category')
adata.write("../results_dlpfc_all/DLPFC_adata_all.h5ad")

with open("../results_dlpfc_all/stitch3d_benchmark_all.json", "w") as f:
    json.dump(benchmark_results, f, indent=2)

