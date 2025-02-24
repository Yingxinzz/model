import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('../DeepST')
import os 
from DeepST import run
import matplotlib.pyplot as plt
from pathlib import Path
import scanpy as sc
import community as louvain
import pandas as pd
import anndata as ad 
import scipy.sparse
data_path = "../../RAW_SLICE/coronal/"
data_name_list=['FFPE', 'DAPI', 'Normal']
save_path = "../results_coronal" 
n_domains = 12
#初始化 DeepST 对象
deepen = run(save_path = save_path, 
	task = "Integration",
	pre_epochs = 800, 
	epochs = 1000, 
	use_gpu = True,
)
###### Generate an augmented list of multiple datasets
augement_data_list = []
graph_list = []
for i in range(len(data_name_list)):
    adata = deepen._get_adata(platform="Visium", data_path=data_path, data_name=data_name_list[i])
    adata.var_names_make_unique()
    adata = deepen._get_image_crop(adata, data_name=data_name_list[i])
    adata = deepen._get_augment(adata, spatial_type="LinearRegress")    
    graph_dict = deepen._get_graph(adata.obsm["spatial"], distType="KDTree")
    graph_list.append(graph_dict)
    augement_data_list.append(adata)

multiple_adata, multiple_graph = deepen._get_multiple_adata(adata_list=augement_data_list, data_name_list=data_name_list, graph_list=graph_list)
data = deepen._data_process(multiple_adata, pca_n_comps=200)
deepst_embed = deepen._fit(
		data = data,
		graph_dict = multiple_graph,
		domains = multiple_adata.obs["batch"].values,  ##### Input to Domain Adversarial Model
		n_domains = len(data_name_list)
		)

multiple_adata.obsm["DeepST_embed"] = deepst_embed
multiple_adata = deepen._get_cluster_data(multiple_adata, n_domains=n_domains, priori = True)
multiple_adata.obs.rename(columns={'batch_name': 'new_batch'}, inplace=True)
multiple_adata.write("../results_coronal/adata_DeepST_notruth.h5ad")

adata = ad.read_h5ad("../results_coronal/adata_DeepST_notruth.h5ad")
# 读取数据并将 'Unnamed: 0' 列设置为索引
ffpe_df = pd.read_csv('../../RAW_SLICE/coronal/FFPE/FFPE_truth.csv', index_col='Unnamed: 0')
dapi_df = pd.read_csv('../../RAW_SLICE/coronal/DAPI/DAPI_truth.csv', index_col='Unnamed: 0')
normal_df = pd.read_csv('../../RAW_SLICE/coronal/Normal/Normal_truth.csv', index_col='Unnamed: 0')
# ffpe_df.index = ffpe_df.index.astype(str).str.replace(r'^FFPE-', '', regex=True).str.replace(r'-1$', '', regex=True)
# dapi_df.index = dapi_df.index.astype(str).str.replace(r'^DAPI-', '', regex=True).str.replace(r'-1$', '', regex=True)
# normal_df.index = normal_df.index.astype(str).str.replace(r'^Normal-', '', regex=True).str.replace(r'-1$', '', regex=True)
ffpe_df.index = ffpe_df.index.astype(str).str.replace(r'^FFPE-', '', regex=True).str.replace(r'-1$', '-1-0-0', regex=True)
dapi_df.index = dapi_df.index.astype(str).str.replace(r'^DAPI-', '', regex=True).str.replace(r'-1$', '-1-1-0', regex=True)
normal_df.index = normal_df.index.astype(str).str.replace(r'^Normal-', '', regex=True).str.replace(r'-1$', '-1-1', regex=True)
print("After FFPE Index:", ffpe_df.index[:5])
print("After DAPI Index:", dapi_df.index[:5])
print("After Normal Index:", normal_df.index[:5])
# Replace with the actual column name that holds the ground truth data
ffpe_ground_truth = ffpe_df['celltype_new'].to_dict()  
dapi_ground_truth = dapi_df['celltype_new'].to_dict()  
normal_ground_truth = normal_df['celltype_new'].to_dict()  
combined_ground_truth = {**ffpe_ground_truth, **dapi_ground_truth, **normal_ground_truth}
adata.obs['ground_truth'] = adata.obs.index.map(combined_ground_truth)
adata = adata[~pd.isnull(adata.obs['ground_truth'])]
print(adata.obs['ground_truth'].head())
adata.write("../results_coronal/adata_DeepST.h5ad")
adata_DeepST = ad.read_h5ad("../results_coronal/adata_DeepST.h5ad")


fig, ax_list = plt.subplots(1, 3, figsize=(12, 4))
sc.pp.neighbors(adata_DeepST, use_rep='X', n_neighbors=10, n_pcs=40,random_state=666)
sc.tl.umap(adata_DeepST,random_state=666)
sc.pl.umap(adata_DeepST, color='new_batch', ax=ax_list[0], title='RAW_uncorrected', show=False)
sc.pp.neighbors(adata_DeepST, use_rep='DeepST_embed',random_state=666)
sc.tl.umap(adata_DeepST,random_state=666)
sc.pl.umap(adata_DeepST, color='new_batch', ax=ax_list[1], title='Batch corrected', show=False)
sc.pl.umap(adata_DeepST, color='DeepST_refine_domain', ax=ax_list[2], title='cluster_leiden', show=False)
plt.tight_layout(w_pad=0.05)
plt.savefig(os.path.join(save_path, f'{"_".join(data_name_list)}_umap.pdf'), bbox_inches='tight', dpi=600)

import scib
sc.pp.neighbors(adata_DeepST, use_rep='DeepST_embed')
scib.me.graph_connectivity(adata_DeepST, label_key="DeepST_refine_domain")
scib.me.ilisi_graph(adata_DeepST, batch_key="new_batch",type_="embed",use_rep="DeepST_embed")
scib.me.kBET(adata_DeepST, batch_key="new_batch", label_key="DeepST_refine_domain", type_="embed", embed="DeepST_embed")
scib.me.silhouette_batch(adata_DeepST, batch_key="new_batch", label_key="DeepST_refine_domain", embed="DeepST_embed")#绘制每个数据集的空间图


