import scanpy as sc
import pandas as pd
import os
file_fold = '../../RAW_SLICE/hbc/'
datasets = ['section1', 'section2']
save_path = "../results_bc" 
data_list = []
Batch_list = []
####scanpy=1.9.1进行预处理
for dataset in datasets:
    adata = sc.read_visium(file_fold + dataset, load_images=True)
    adata.var_names_make_unique()
    adata.X = adata.X.astype('float32')
    if 'spatial' in adata.obsm:
        adata.obsm['spatial'] = adata.obsm['spatial'].astype('float32')
    min_gene = 20
    min_cell = 20
    sc.pp.filter_cells(adata, min_genes=min_gene)
    sc.pp.filter_genes(adata, min_cells=min_cell)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
    adata = adata[:, adata.var['highly_variable']]
    h5ad_path = os.path.join(save_path, f"{dataset}.h5ad")
    adata.write_h5ad(h5ad_path)
    data_list.append(h5ad_path)
    Batch_list.append(adata)
    print(f"Saved {h5ad_path}")

####换个scanpy版本读取处理好的h5ad文件  conda activate Spatialign   
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
data_list = [
    '../results_bc/section1.h5ad',
    '../results_bc/section2.h5ad',
]
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
    save_path="../results_bc/",
    is_verbose=False)

raw_merge = AnnData.concatenate(*model.dataset.data_list)
import time
start_time = time.time()
model.train(0.05,1,0.1)
model.alignment()
end_time = time.time()
print(f"well time: {end_time - start_time} seconds")

correct1 = sc.read_h5ad("../results_bc/res/correct_data0.h5ad")
correct2 = sc.read_h5ad("../results_bc/res/correct_data1.h5ad")
merge_data = correct1.concatenate(correct2)
# 更新 batch 列
batch_mapping = {
    '0': 'section1',
    '1': 'section2',
}
raw_merge.obs['new_batch'] = raw_merge.obs['batch'].replace(batch_mapping)
merge_data.obs['new_batch'] = merge_data.obs['batch'].replace(batch_mapping)

###########clusting
import scanpy as sc
from sklearn.mixture import GaussianMixture
sc.pp.scale(merge_data)
X = merge_data.obsm['correct']
n_components = 14 
gmm = GaussianMixture(n_components=n_components, random_state=42)
merge_data.obs['mclust'] = gmm.fit_predict(X)
merge_data.obs["mclust"] = merge_data.obs["mclust"].astype("category")

merge_data.write("../results_bc/multiple_adata.h5ad")
adata_spatiAlign = sc.read_h5ad("../results_bc/multiple_adata.h5ad") 





# # fig, ax_list = plt.subplots(1, 2, figsize=(8, 4))
# # # sc.pp.neighbors(raw_merge, use_rep='X', random_state=42)
# # # sc.tl.umap(raw_merge, random_state=42)
# # # sc.pl.umap(raw_merge, color='new_batch', title='Uncorrected', ax=ax_list[0], show=False)
# # sc.pp.neighbors(adata_spatiAlign, use_rep='correct',random_state=42) 
# # sc.tl.umap(adata_spatiAlign,random_state=42)
# # sc.pl.umap(adata_spatiAlign, color='new_batch', ax=ax_list[0], title='spatiAlign', show=False)
# # sc.pl.umap(adata_spatiAlign, color='mclust', ax=ax_list[1], title='cluster_mclust', show=False)
# # plt.tight_layout(w_pad=0.05)
# # plt.savefig('../results_bc/umap_comparison_bc.png')

# # import scib
# # sc.pp.neighbors(adata_spatiAlign, use_rep="correct")
# # scib.me.graph_connectivity(adata_spatiAlign, label_key="mclust")
# # scib.me.ilisi_graph(adata_spatiAlign, batch_key="new_batch", type_="embed", use_rep="correct")
# # scib.me.kBET(adata_spatiAlign, batch_key="new_batch", label_key="mclust", type_="embed", embed="correct")
# # scib.me.silhouette_batch(adata_spatiAlign, batch_key="new_batch", label_key="mclust", embed="correct")
