import warnings
warnings.filterwarnings("ignore")

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
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import anndata as ad 
os.environ['R_HOME'] = '/root/anaconda3/envs/SS/lib/R'
n_clusters = 7
datasets=['151673','151669','151507']
file_fold = '../../RAW_SLICE/DLPFC/'
adatas = []
for dataset in datasets:  
    adata = sc.read_visium(file_fold+dataset, count_file=dataset+'_filtered_feature_bc_matrix.h5', load_images=True)
    adata.var_names_make_unique()
    adata.obs['batch'] = dataset  
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
# define model
model = GraphST.GraphST(adata, device=device, random_seed=50)
# run model
import time
start_time = time.time()
adata = model.train()
end_time = time.time()
print(f"well time: {end_time - start_time} seconds")
# clustering
from GraphST.utils import clustering
tool = 'mclust' # mclust, leiden, and louvain
if tool == 'mclust':
    clustering(adata, n_clusters, method=tool)
elif tool in ['leiden', 'louvain']:
    clustering(adata, n_clusters, method=tool, start=0.1, end=2.0, increment=0.01)

### Concat the scanpy objects for multiple slices
adata.obs["new_batch"] = adata.obs["batch"]  
adata.obs["batch_name"] = adata.obs["new_batch"].astype('category')
print('adata.shape: ', adata.shape)
###combine silce names into sample name
new_batch_1 = adata.obs["new_batch"].isin(['0'])
new_batch_2 = adata.obs["new_batch"].isin(['1'])
new_batch_3 = adata.obs["new_batch"].isin(['2'])
adata.obs["sample_name"] = list(sum(new_batch_1)*['Sample 1'])+list(sum(new_batch_2)*['Sample 2'])+list(sum(new_batch_3)*['Sample 3'])
adata.obs["sample_name"] = adata.obs["sample_name"].astype('category')
adata.obs["batch_name"] = adata.obs["sample_name"].copy()


# plotting spatial clustering result
adata.obsm['spatial'][:, 1] = -1 * adata.obsm['spatial'][:, 1]
rgb_values = sns.color_palette("tab20", len(adata.obs['domain'].unique()))
color_fine = dict(zip(list(adata.obs['domain'].unique()), rgb_values))
import anndata as ad 
adata.write("../results/all_DLPFC_adata.h5ad")
adata = ad.read_h5ad("../results/all_DLPFC_adata.h5ad")




# # #绘制批量效应校正前后的 UMAP 
# # fig, ax_list = plt.subplots(1, 3, figsize=(12, 4))
# # # #sc.pp.normalize_total(adata)
# # # #sc.pp.log1p(adata)
# # # #sc.pp.pca(adata)
# # # #sc.pp.neighbors(adata, use_rep='X_pca', n_neighbors=10, n_pcs=40) 不明显
# # # sc.pp.neighbors(adata, use_rep='X', n_neighbors=10, n_pcs=40,random_state=666)
# # # sc.tl.umap(adata,random_state=666)
# # # sc.pl.umap(adata, color='batch_name', title='Uncorrected', ax=ax_list[0], show=False)
# # sc.pp.neighbors(adata, use_rep='emb_pca', n_neighbors=10,random_state=666)
# # sc.tl.umap(adata,random_state=666)
# # sc.pl.umap(adata, color='batch_name', ax=ax_list[0], title='Batch corrected', show=False)
# # sc.pl.umap(adata, color='domain', ax=ax_list[1], title='Colored by clusters', show=False)
# # sc.pl.umap(adata, color='ground_truth', ax=ax_list[2], title='truth', show=False)
# # plt.tight_layout(w_pad=0.05)
# # plt.savefig('../results/umap_all_comparison_DLPFC.png')

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
