#Import packages
import os
import sys
sys.path.append('../')
import scanpy as sc
from spatialign import Spatialign
from warnings import filterwarnings
from anndata import AnnData
import matplotlib.pyplot as plt
import anndata  as ad
filterwarnings("ignore")
datasets = [
    "../../RAW_SLICE/10X.h5ad",
    "../../RAW_SLICE/BGI.h5ad", 
    "../../RAW_SLICE/SlideV2.h5ad"
]
adata_list = [sc.read_h5ad(dataset) for dataset in datasets]
for i, adata in enumerate(adata_list):
    adata.X = adata.X.astype('float32')
    if 'spatial' in adata.obsm.keys():
        adata.obsm['spatial'] = adata.obsm['spatial'].astype('float32')
    output_path = datasets[i].replace(".h5ad", "_float32.h5ad")
    adata.write_h5ad(output_path)
    print(f"Saved float32 data to {output_path}")

import torch
torch.set_default_dtype(torch.float32)

datasets=[
    "../../RAW_SLICE/10X_float32.h5ad",
    "../../RAW_SLICE/BGI_float32.h5ad", 
    "../../RAW_SLICE/SlideV2_float32.h5ad"
]
data_list=[]
Batch_list = []
save_path = "../results_mob" 

for path in datasets:
    adata = sc.read(path)
    sc.pp.normalize_total(adata, target_sum=1e4) 
    sc.pp.log1p(adata)
    dataset = os.path.basename(path).split('.')[0]
    h5ad_path = os.path.join(save_path, f"{dataset}.h5ad")
    adata.write_h5ad(h5ad_path)  
    data_list.append(h5ad_path)
    Batch_list.append(adata)

model = Spatialign(
    *datasets,
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

#Training Spatialign model
#model.train(tau1=0.05, tau2=1, tau3=0.1)
model.train(0.05, 1, 0.1)
model.alignment()

#Validation inference datasets
correct1 = sc.read_h5ad("../results_mob/res/correct_data0.h5ad")
correct2 = sc.read_h5ad("../results_mob/res/correct_data1.h5ad")
correct3 = sc.read_h5ad("../results_mob/res/correct_data2.h5ad")
merge_data = correct1.concatenate(correct2, correct3)
# if hasattr(merge_data.X, "toarray"):  
#     dense_X = merge_data.X.toarray() 
# else:
#     dense_X = merge_data.X 
##########clustering
sc.pp.neighbors(merge_data, use_rep='correct',random_state=666)
sc.tl.louvain(merge_data, random_state=666, key_added="louvain", resolution=1.15)
# import scanpy as sc
# from sklearn.mixture import GaussianMixture
# sc.pp.scale(merge_data)
# X = merge_data.obsm['correct']
# n_components = 9
# gmm = GaussianMixture(n_components=n_components, random_state=42)
# merge_data.obs['mclust'] = gmm.fit_predict(X)
# merge_data.obs["mclust"] = merge_data.obs["mclust"].astype("category")
batch_mapping = {
    '0': '10X',
    '1': 'BGI',
    '2': 'SlideV2'
}
# 更新 batch 列
merge_data.obs['new_batch'] = merge_data.obs['batch'].replace(batch_mapping)
merge_data.write("../results_mob/multiple_adata.h5ad")
merge_data = sc.read_h5ad("../results_mob/multiple_adata.h5ad")




# # #Visualization original dataset by UMAP
# # fig, ax_list = plt.subplots(1, 3, figsize=(12, 4))
# # sc.tl.pca(raw_merge, n_comps=100, random_state=42)
# # sc.pp.neighbors(raw_merge, use_rep='X_pca', n_neighbors=10, n_pcs=40,random_state=42)
# # # #原始计数的最近邻（不推荐）
# # # #sc.pp.neighbors(adata, use_rep='X', n_neighbors=10, n_pcs=40,random_state=666)
# # # sc.tl.umap(raw_merge, random_state=42)
# # sc.pl.umap(raw_merge, color='new_batch', title='Uncorrected', ax=ax_list[0], show=False)
# # sc.pp.neighbors(merge_data, use_rep='correct',random_state=42) 
# # sc.tl.umap(merge_data,random_state=42)
# # sc.pl.umap(merge_data, color='new_batch', ax=ax_list[1], title='Batch corrected', show=False)
# # #sc.pl.umap(merge_data, color='celltype', ax=ax_list[1], title='celltype', show=False)
# # sc.pl.umap(merge_data, color='louvain', ax=ax_list[2], title='Colored by clusters', show=False)
# # plt.tight_layout(w_pad=0.05)
# # plt.savefig('../results_mob/umap_comparison_MOB.png')

# # import scib
# # sc.pp.neighbors(merge_data, use_rep="correct")
# # scib.me.graph_connectivity(merge_data, label_key="celltype")
# # scib.me.ilisi_graph(merge_data, batch_key="new_batch", type_="embed", use_rep="correct")
# # scib.me.kBET(merge_data, batch_key="new_batch", label_key="celltype", type_="embed", embed="correct")
# # #scib.me.kBET(adata, batch_key="batch", label_key="ceelltype", type_="knn")
# # scib.me.silhouette_batch(merge_data, batch_key="new_batch", label_key="celltype", embed="correct")
