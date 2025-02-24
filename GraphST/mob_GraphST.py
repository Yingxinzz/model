import sys
sys.path.append('../')
import os
import torch
import scanpy as sc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
import multiprocessing as mp
from GraphST import GraphST
import numpy as np
import anndata as ad
from scipy import sparse

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['R_HOME'] = '/root/anaconda3/envs/SS/lib/R'
n_clusters = 9
datasets = ['10X','BGI', 'SlideV2' ]
file_fold = "../../RAW_SLICE/"
adatas = []
for dataset in datasets:
    adata = sc.read_h5ad(file_fold + dataset + '.h5ad')
    adata.var_names_make_unique()
    adata.obs['batch'] = dataset
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)  # 降低高变基因数量
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var['highly_variable']]
    adatas.append(adata)

adata = adatas[0].concatenate(adatas[1:], batch_key='batch')
adata.X = sparse.csr_matrix(adata.X)
adata.X = adata.X.astype(np.float32)
model = GraphST.GraphST(adata, device=device, random_seed=666)
adata = model.train()

batch_mapping = {'0':'10X', '1':  'BGI', '2': 'SlideV2'}
adata.obs['new_batch'] = adata.obs['batch'].replace(batch_mapping)
# clustering 
from GraphST.utils import clustering
clustering(adata, n_clusters, method='mclust')
adata.write("../results/mob_adata.h5ad", compression='gzip')  
adata = ad.read_h5ad("../results/mob_adata.h5ad")





# # fig, ax_list = plt.subplots(1, 3, figsize=(12, 4))
# # sc.pp.pca(adata)
# # sc.pp.neighbors(adata, use_rep='X_pca', n_neighbors=15, n_pcs=40,random_state=666)
# # sc.tl.umap(adata,random_state=666)
# # sc.pl.umap(adata, color='new_batch', title='Uncorrected', ax=ax_list[0], show=False)
# # sc.pp.neighbors(adata, use_rep='emb_pca', n_neighbors=30,random_state=666)
# # sc.tl.umap(adata,random_state=666)
# # sc.pl.umap(adata, color='new_batch', ax=ax_list[1], title='Batch corrected', show=False)
# # sc.pl.umap(adata, color='domain', ax=ax_list[2], title='Colored by clusters', show=False)
# # #sc.pl.umap(adata, color='celltype', ax=ax_list[2], title='celltype', show=False)
# # plt.tight_layout(w_pad=0.05)
# # plt.savefig('../results/umap_comparison_mob.png')

# # import scib
# # sc.pp.neighbors(adata, use_rep="emb_pca")
# # scib.me.graph_connectivity(adata, label_key="celltype")
# # scib.me.ilisi_graph(adata, batch_key="new_batch", type_="embed", use_rep="emb_pca")
# # scib.me.kBET(adata, batch_key="new_batch", label_key="celltype", type_="embed", embed="emb_pca")
# # scib.me.kBET(adata, batch_key="new_batch", label_key="celltype", type_="knn")
# # scib.me.silhouette_batch(adata, batch_key="new_batch", label_key="celltype", embed="emb_pca")

# #空间域绘制
# # # plotting spatial clustering result
# # adata.obsm['spatial'][:, 1] = -1 * adata.obsm['spatial'][:, 1]
# # rgb_values = sns.color_palette("tab20", len(adata.obs['domain'].unique()))
# # color_fine = dict(zip(list(adata.obs['domain'].unique()), rgb_values))

# # plt.rcParams["figure.figsize"] = (12, 6)
# # sc.pl.embedding(adata, basis="spatial",
# #                 color="domain",
# #                 s=100,
# #                 palette=color_fine,
# #                 show=False,
# #                 title='DLPFC Neighbors')
# # plt.savefig('../results/spatial_neighbors.png')
# # plt.show()