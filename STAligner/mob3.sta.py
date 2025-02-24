import sys
sys.path.append('../')
import STAligner
from STAligner import ST_utils
import os
os.environ['R_HOME'] = '/root/anaconda3/envs/SS/lib/R'
os.environ['R_LIBS'] = '/root/anaconda3/envs/SS/lib/R/library'
os.environ['R_USER'] = "/root/anaconda3/envs/SG/lib/python3.8/site-packages/rpy2"
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri

import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
import scipy.linalg
from scipy.sparse import csr_matrix

import torch
used_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import warnings
warnings.filterwarnings("ignore")

#Load Data
Batch_list = []
adj_list = []
datasets = ['10X','BGI','SlideV2']

for dataset in datasets:
    adata = sc.read_h5ad(os.path.join('../../RAW_SLICE', dataset + '.h5ad'))
    adata.X = csr_matrix(adata.X)
    adata.var_names_make_unique()
    print('Before flitering: ', adata.shape)
    sc.pp.filter_genes(adata, min_cells=50)
    print('After flitering: ', adata.shape)
    # make spot name unique
    adata.obs_names = [x+'_'+dataset for x in adata.obs_names]
    # Constructing the spatial network
    STAligner.Cal_Spatial_Net(adata, rad_cutoff=150)
    STAligner.Stats_Spatial_Net(adata) # plot the number of spatial neighbors
    # Normalization
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3",n_top_genes=5000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var['highly_variable']]
    adj_list.append(adata.uns['adj'])  
    Batch_list.append(adata)    

adata_concat = ad.concat(Batch_list, label="slice_name", keys=datasets)
adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')
print('adata_concat.shape: ', adata_concat.shape)
adj_concat = np.asarray(adj_list[0].todense())
for batch_id in range(1,len(datasets)):
    adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))

adata_concat.uns['edgeList'] = np.nonzero(adj_concat)

#Running STAligner
import time
start_time = time.time()
adata_concat = STAligner.train_STAligner(adata_concat, verbose=True, knn_neigh=100, device=used_device)  # epochs = 1500
end_time = time.time()
print(f"well time: {end_time - start_time} seconds")

edge_list = [[left, right] for left, right in zip(adata_concat.uns['edgeList'][0], adata_concat.uns['edgeList'][1])]
adata_concat.uns['edgeList'] = edge_list

Clustering
import scanpy as sc
from sklearn.mixture import GaussianMixture
sc.pp.scale(adata_concat)
X = adata_concat.obsm['STAligner']
n_components =9
gmm = GaussianMixture(n_components=n_components, random_state=42)
adata_concat.obs['mclust'] = gmm.fit_predict(X)
adata_concat.obs["mclust"] = adata_concat.obs["mclust"].astype("category")

sc.pp.neighbors(adata_concat, use_rep='STAligner',random_state=666)
sc.tl.louvain(adata_concat, random_state=666, key_added="louvain", resolution=0.4)
adata_concat.write('../results/staligner_mob3.h5ad')








# # import seaborn as sns
# # from sklearn import preprocessing
# # from matplotlib.colors import ListedColormap
# # import numpy as np
# # import scanpy as sc
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import warnings
# # warnings.filterwarnings("ignore")
# # adata = sc.read_h5ad('../results/staligner_mob3.h5ad')

# # fig, ax_list = plt.subplots(1, 3, figsize=(12, 4))
# # # #sc.pp.neighbors(adata, use_rep='X', random_state=666)  killed 内存不足 将维了
# # sc.pp.pca(adata)
# # sc.pp.neighbors(adata, use_rep='X_pca', n_neighbors=15, n_pcs=40,random_state=666)
# # sc.tl.umap(adata, random_state=666)
# # sc.pl.umap(adata, color='batch_name', title='uncorrected', ax=ax_list[0],show=False)
# # sc.pp.neighbors(adata, use_rep='STAligner', random_state=666)
# # sc.tl.umap(adata,random_state=666)
# # sc.pl.umap(adata, color='batch_name', title='corrected', ax=ax_list[1],show=False)
# # sc.pl.umap(adata, color='louvain', title='louvain',ax=ax_list[2], show=False)
# # #sc.pl.umap(adata, color='celltype', title='celltype',ax=ax_list[2], show=False)

# # plt.tight_layout(w_pad=0.05)
# # plt.savefig('../results/umap_comparison_mob3.png')

# # import scib
# # sc.pp.neighbors(adata, use_rep="STAligner")
# # scib.me.graph_connectivity(adata, label_key="celltype")
# # scib.me.ilisi_graph(adata, batch_key="batch_name", type_="embed", use_rep="STAligner")
# # scib.me.kBET(adata, batch_key="batch_name", label_key="celltype", type_="embed", embed="STAligner")
# # scib.me.kBET(adata, batch_key="batch_name", label_key="celltype", type_="knn")
# # scib.me.silhouette_batch(adata, batch_key="batch_name", label_key="celltype", embed="STAligner")


# # # 绘制 UMAP 图，按批次显示不同颜色
# # p1=sc.pl.umap(adata, color="batch_name", title="STAligner UMAP")
# # plt.savefig("../results/STAligner_UMAP.png")
# # sc.pp.neighbors(adata, use_rep='STAligner', n_neighbors=30, random_state=666)
# # sc.tl.umap(adata, random_state=666,min_dist=0.5)
# # sc.pl.umap(adata, color='batch_name', title='corrected')



