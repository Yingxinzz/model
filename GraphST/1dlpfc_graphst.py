import os
import torch
import scanpy as sc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
import multiprocessing as mp
import sys
sys.path.append('../')
from GraphST import GraphST
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['R_HOME'] = '/root/anaconda3/envs/SS/lib/R'
n_clusters = 7
datasets = ['151673', '151674', '151675', '151676']
file_fold = '../../RAW_SLICE/DLPFC/'
adatas = []
for dataset in datasets:  
    adata = sc.read_visium(file_fold+dataset, count_file=dataset+'_filtered_feature_bc_matrix.h5', load_images=True)
    adata.var_names_make_unique()
    adata.obs['batch'] = dataset  # Add batch information
    df_meta = pd.read_csv(file_fold+dataset+'/metadata.tsv', sep='\t')
    df_meta_layer = df_meta['layer_guess']
    adata.obs.loc[adata.obs['batch'] == dataset, 'ground_truth'] = df_meta_layer.values
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var['highly_variable']]
    adatas.append(adata)

adata = adatas[0].concatenate(adatas[1:], batch_key='batch')
model = GraphST.GraphST(adata, device=device, random_seed=50)
adata = model.train()
from GraphST.utils import clustering
tool = 'mclust' # mclust, leiden, and louvain
if tool == 'mclust':
    clustering(adata, n_clusters, method=tool)
elif tool in ['leiden', 'louvain']:
    clustering(adata, n_clusters, method=tool, start=0.1, end=2.0, increment=0.01)
batch_mapping = {
    '0': '151673',
    '1': '151674',
    '2': '151675',
    '3': '151676'
}
# 更新 batch 列
adata.obs['new_batch'] = adata.obs['batch'].replace(batch_mapping)

# plotting spatial clustering result
adata.obsm['spatial'][:, 1] = -1 * adata.obsm['spatial'][:, 1]
rgb_values = sns.color_palette("tab20", len(adata.obs['domain'].unique()))
color_fine = dict(zip(list(adata.obs['domain'].unique()), rgb_values))
import anndata as ad 
adata.write("../results/DLPFC_adata.h5ad")
adata = ad.read_h5ad("../results/DLPFC_adata.h5ad")




# # #绘制批量效应校正前后的 UMAP 
# # fig, ax_list = plt.subplots(1, 3, figsize=(12, 4))
# # # sc.pp.neighbors(adata, use_rep='X', n_neighbors=10, n_pcs=40,random_state=666)
# # # sc.tl.umap(adata,random_state=666)
# # # sc.pl.umap(adata, color='new_batch', title='Uncorrected', ax=ax_list[0], show=False)
# # sc.pp.neighbors(adata, use_rep='emb_pca', n_neighbors=10,random_state=666)
# # sc.tl.umap(adata,random_state=666)
# # sc.pl.umap(adata, color='new_batch', ax=ax_list[0], title='Batch corrected', show=False)
# # sc.pl.umap(adata, color='domain', ax=ax_list[1], title='cluster_mclust', show=False)
# # sc.pl.umap(adata, color='ground_truth', ax=ax_list[2], title='celltype', show=False)
# # plt.tight_layout(w_pad=0.05)
# # plt.savefig('../results/umap_comparison_DLPFC.png')

# # import scib
# # sc.pp.neighbors(adata, use_rep="emb_pca")
# # scib.me.graph_connectivity(adata, label_key="ground_truth")
# # scib.me.ilisi_graph(adata, batch_key="new_batch", type_="embed", use_rep="emb_pca")
# # scib.me.kBET(adata, batch_key="new_batch", label_key="ground_truth", type_="embed", embed="emb_pca")
# # scib.me.kBET(adata, batch_key="new_batch", label_key="ground_truth", type_="knn")
# # scib.me.silhouette_batch(adata, batch_key="new_batch", label_key="ground_truth", embed="emb_pca")

# # #空间域绘制
# # plt.rcParams["figure.figsize"] = (12, 6)
# # sc.pl.embedding(adata, basis="spatial",
# #                 color="domain",
# #                 s=100,
# #                 palette=color_fine,
# #                 show=False,
# #                 title='DLPFC Neighbors')
# # plt.savefig('../results/spatial_neighbors.png')
# # plt.show()
