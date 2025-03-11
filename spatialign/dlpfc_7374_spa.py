import scanpy as sc
import pandas as pd
import os
file_fold = '../../RAW_SLICE/DLPFC/'
datasets = ['151673', '151674']
save_path = "../results_dlpfc1" 
data_list = []
Batch_list = []
####scanpy=1.9.1进行预处理
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
    '../results_dlpfc1/151673.h5ad',
    '../results_dlpfc1/151674.h5ad',
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
    save_path="../results_dlpfc1/",
    is_verbose=False)

raw_merge = AnnData.concatenate(*model.dataset.data_list)
import time
start_time = time.time()
model.train(0.05,1,0.1)
model.alignment()
end_time = time.time()
print(f"well time: {end_time - start_time} seconds")

correct1 = sc.read_h5ad("../results_dlpfc1/res/correct_data0.h5ad")
correct2 = sc.read_h5ad("../results_dlpfc1/res/correct_data1.h5ad")
merge_data = correct1.concatenate(correct2)
# 更新 batch 列
batch_mapping = {
    '0': '151673',
    '1': '151674',
}
raw_merge.obs['new_batch'] = raw_merge.obs['batch'].replace(batch_mapping)
merge_data.obs['new_batch'] = merge_data.obs['batch'].replace(batch_mapping)
# merge_data.obs.rename(columns={'celltype': 'ground_truth'}, inplace=True)

###########clusting
import scanpy as sc
from sklearn.mixture import GaussianMixture
sc.pp.scale(merge_data)
X = merge_data.obsm['correct']
n_components = 7 
gmm = GaussianMixture(n_components=n_components, random_state=42)
merge_data.obs['mclust'] = gmm.fit_predict(X)
merge_data.obs["mclust"] = merge_data.obs["mclust"].astype("category")

merge_data.write("../results_dlpfc1/multiple_adata_7374.h5ad")
merge_data = sc.read_h5ad("../results_dlpfc1/multiple_adata_7374.h5ad")



# # #Visualization original dataset by UMAP
# # fig, ax_list = plt.subplots(1, 3, figsize=(12, 4))
# # # sc.pp.neighbors(raw_merge, use_rep='X', random_state=42)
# # # sc.tl.umap(raw_merge, random_state=42)
# # # sc.pl.umap(raw_merge, color='new_batch', title='Uncorrected', ax=ax_list[0], show=False)
# # sc.pp.neighbors(merge_data, use_rep='correct',random_state=42) 
# # sc.tl.umap(merge_data,random_state=42)
# # sc.pl.umap(merge_data, color='new_batch', ax=ax_list[0], title='Batch corrected', show=False)
# # sc.pl.umap(merge_data, color='mclust', ax=ax_list[1], title='cluster_mclust', show=False)
# # sc.pl.umap(merge_data, color='celltype', ax=ax_list[2], title='celltype', show=False)
# # plt.tight_layout(w_pad=0.05)
# # plt.savefig('../results_dlpfc1/umap_comparison1_DLPFC.png')

# # import scib
# # sc.pp.neighbors(merge_data, use_rep="correct")
# # scib.me.graph_connectivity(merge_data, label_key="celltype")
# # scib.me.ilisi_graph(merge_data, batch_key="new_batch", type_="embed", use_rep="correct")
# # scib.me.kBET(merge_data, batch_key="new_batch", label_key="celltype", type_="embed", embed="correct")
# # scib.me.silhouette_batch(merge_data, batch_key="new_batch", label_key="celltype", embed="correct")

