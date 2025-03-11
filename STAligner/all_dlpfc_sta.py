import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('../')
import STAligner
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
datasets = ['151673','151669','151507']
print(datasets)
file_fold = '../../RAW_SLICE/DLPFC/'
for dataset in datasets:
    print(dataset)
    adata = sc.read_visium(file_fold+dataset, count_file=dataset+'_filtered_feature_bc_matrix.h5', load_images=True)
    adata.var_names_make_unique()
    # read the annotation
    Ann_df = pd.read_csv(os.path.join(file_fold+dataset, dataset + '_truth.txt'), sep='\t', header=None, index_col=0)
    Ann_df.columns = ['Ground Truth']
    Ann_df[Ann_df.isna()] = "unknown"
    adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'Ground Truth'].astype('category')
    # make spot name unique
    adata.obs_names = [x+'_'+dataset for x in adata.obs_names]
    # Add batch information
    adata.obs['batch'] = dataset  
    # Constructing the spatial network
    STAligner.Cal_Spatial_Net(adata, rad_cutoff=150) 
    STAligner.Stats_Spatial_Net(adata)
    ## Normalization
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=10000) #ensure enough common HVGs in the combined matrix
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var['highly_variable']]
    adj_list.append(adata.uns['adj'])
    Batch_list.append(adata)

adata_concat = ad.concat(Batch_list, label="slice_name", keys=datasets)
adata_concat.obs['Ground_Truth'] = adata_concat.obs['Ground Truth'].astype('category')
adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')
print('adata_concat.shape: ', adata_concat.shape)
new_batch_1 = adata_concat.obs["slice_name"].isin(['151673'])
new_batch_2 = adata_concat.obs["slice_name"].isin(['151669'])
new_batch_3 = adata_concat.obs["slice_name"].isin(['151507'])
adata_concat.obs["sample_name"] = list(sum(new_batch_1)*['Sample 1'])+list(sum(new_batch_2)*['Sample 2'])+list(sum(new_batch_3)*['Sample 3'])
adata_concat.obs["sample_name"] = adata_concat.obs["sample_name"].astype('category')
adata_concat.obs["batch_name"] = adata_concat.obs["sample_name"].copy()

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

STAligner.mclust_R(adata_concat, num_cluster=7, used_obsm='STAligner')
adata_concat = adata_concat[adata_concat.obs['Ground Truth']!='unknown']

from sklearn.metrics import adjusted_rand_score as ari_score
print('mclust, ARI = %01.3f' % ari_score(adata_concat.obs['Ground Truth'], adata_concat.obs['mclust']))
adata_concat.write('../results/staligner_all_DLPFC.h5ad')





#可视化
# # import seaborn as sns
# # from sklearn import preprocessing
# # from matplotlib.colors import ListedColormap
# # import numpy as np
# # import scanpy as sc
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import warnings
# # warnings.filterwarnings("ignore")
# # from STAligner import ST_utils
# # from STAligner.ST_utils import match_cluster_labels
# # adata = sc.read_h5ad('../results/staligner_all_DLPFC.h5ad')

# # #画出STAligner之前umap图
# # adata.obs['mclust'] = pd.Series(ST_utils.match_cluster_labels(adata.obs['Ground Truth'], adata.obs['mclust'].values),index=adata.obs.index, dtype='category')
# # fig, ax_list = plt.subplots(1, 3, figsize=(12, 4))
# # # sc.pp.neighbors(adata, use_rep='X', random_state=666)
# # # sc.tl.umap(adata, random_state=666)
# # # sc.pl.umap(adata, color='batch_name', title='uncorrected', ax=ax_list[0],show=False)
# # sc.pp.neighbors(adata, use_rep='STAligner', random_state=666)
# # sc.tl.umap(adata,random_state=22)
# # sc.pl.umap(adata, color='batch_name', title='corrected', ax=ax_list[0],show=False)
# # sc.pl.umap(adata, color='Ground_Truth', title='Ground_Truth', ax=ax_list[1], show=False)
# # sc.pl.umap(adata, color='mclust', title='Colored by clusters', ax=ax_list[2], show=False)
# # plt.tight_layout(w_pad=0.05)
# # plt.savefig('../results/umap_comparison_all_DLPFC.png')

# # import scib
# # sc.pp.neighbors(adata, use_rep="STAligner")
# # scib.me.graph_connectivity(adata, label_key="Ground_Truth")
# # scib.me.ilisi_graph(adata, batch_key="batch_name", type_="embed", use_rep="STAligner")
# # scib.me.kBET(adata, batch_key="batch_name", label_key="Ground_Truth", type_="embed", embed="STAligner")
# # scib.me.kBET(adata, batch_key="batch_name", label_key="Ground_Truth", type_="knn")
# # scib.me.silhouette_batch(adata, batch_key="batch_name", label_key="Ground_Truth", embed="STAligner")



