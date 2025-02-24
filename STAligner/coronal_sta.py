import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('../')
import STAligner
from STAligner import ST_utils
from STAligner.ST_utils import match_cluster_labels
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


Batch_list = []
adj_list = []
datasets = ['FFPE', 'DAPI', 'Normal']
file_fold = '../../RAW_SLICE/coronal/'
for dataset in datasets:   
    adata = sc.read_visium(file_fold+dataset, count_file='filtered_feature_bc_matrix.h5', load_images=True)
    adata.var_names_make_unique()
    # read the annotation
    Ann_df = pd.read_csv(os.path.join(file_fold, dataset, dataset + '_truth.csv'), sep=',', header=0, index_col=0)
    Ann_df.index = Ann_df.index.str.replace(r'^[A-Za-z]+-', '', regex=True)
    Ann_df.index = Ann_df.index.str.strip()  
    missing_barcodes = set(adata.obs_names) - set(Ann_df.index)
    if missing_barcodes:
        print(f"Missing barcodes: {missing_barcodes}")
    
    adata.obs_names = adata.obs_names.str.strip()  
    Ann_df.index = Ann_df.index.str.strip()
    common_barcodes = adata.obs_names[adata.obs_names.isin(Ann_df.index)]
    cell_info_new = Ann_df.loc[common_barcodes, 'celltype_new']
    adata.obs['ground_truth'] = cell_info_new
    adata = adata[adata.obs['ground_truth'] != 'unknown']
    # make spot name unique
    adata.obs_names = [x+'_'+dataset for x in adata.obs_names]
    adata.obs['batch'] = dataset  # Add batch information
    # Constructing the spatial network
    STAligner.Cal_Spatial_Net(adata, rad_cutoff=300)
    STAligner.Stats_Spatial_Net(adata)
    # Normalization
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
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
import time
start_time = time.time()
adata_concat = STAligner.train_STAligner(adata_concat, verbose=True, knn_neigh=100, device=used_device)  # epochs = 1500,
end_time = time.time()
print(f"well time: {end_time - start_time} seconds")
edge_list = [[left, right] for left, right in zip(adata_concat.uns['edgeList'][0], adata_concat.uns['edgeList'][1])]
adata_concat.uns['edgeList'] = edge_list
adata_concat.obs.rename(columns={'batch_name': 'new_batch'}, inplace=True)

#######Clustering
ST_utils.mclust_R(adata_concat, num_cluster=12, used_obsm='STAligner')
adata_concat = adata_concat[~adata_concat.obs['ground_truth'].isna() & ~adata_concat.obs['mclust'].isna()]
from sklearn.metrics import adjusted_rand_score as ari_score
print('mclust, ARI = %01.3f' % ari_score(adata_concat.obs['ground_truth'], adata_concat.obs['mclust']))
adata_concat.write('../results/staligner_coronal.h5ad')





# # import seaborn as sns
# # from sklearn import preprocessing
# # from matplotlib.colors import ListedColormap
# # import numpy as np
# # import scanpy as sc
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # adata_STAligner = sc.read_h5ad('../results/staligner_coronal.h5ad')

# # #画出STAligner之前umap图
# # fig, ax_list = plt.subplots(1, 3, figsize=(12, 4))
# # sc.pp.neighbors(adata_STAligner, use_rep='X', random_state=666)
# # sc.tl.umap(adata_STAligner, random_state=666)
# # sc.pl.umap(adata_STAligner, color='new_batch', title='RAW_uncorrected', ax=ax_list[0],show=False)
# # sc.pp.neighbors(adata_STAligner, use_rep='STAligner', random_state=666)
# # sc.tl.umap(adata_STAligner,random_state=22)
# # sc.pl.umap(adata_STAligner, color='new_batch', title='corrected', ax=ax_list[1],show=False)
# # sc.pl.umap(adata_STAligner, color='mclust', title='Colored by clusters', ax=ax_list[2], show=False)
# # plt.tight_layout(w_pad=0.05)
# # plt.savefig('../results/umap_comparison_coronal.png')

# # import scib
# # sc.pp.neighbors(adata, use_rep="STAligner")
# # scib.me.graph_connectivity(adata_STAligner, label_key="louvain")
# # scib.me.ilisi_graph(adata_STAligner, batch_key="new_batch", type_="embed", use_rep="STAligner")
# # scib.me.kBET(adata_STAligner, batch_key="new_batch", label_key="louvain", type_="embed", embed="STAligner")
# # scib.me.silhouette_batch(adata_STAligner, batch_key="new_batch", label_key="louvain", embed="emb_pca")
