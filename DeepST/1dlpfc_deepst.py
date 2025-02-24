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
data_path = "../data/DLPFC/" 
data_name_list = ['151673', '151674', '151675', '151676']
save_path = "../Results" 
n_domains = 7
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
    adata = deepen._get_image_crop(adata, data_name=data_name_list[i])
    adata = deepen._get_augment(adata, spatial_type="LinearRegress")
    adata.obs['new_batch'] = data_name_list[i]
    df_meta = pd.read_csv(data_path + data_name_list[i] + '/metadata.tsv', sep='\t')
    df_meta_layer = df_meta['layer_guess']
    adata.obs.loc[adata.obs['new_batch'] == data_name_list[i], 'ground_truth'] = df_meta_layer.values
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]
    graph_dict = deepen._get_graph(adata.obsm["spatial"], distType="KDTree")
    graph_list.append(graph_dict)
    augement_data_list.append(adata)

print(f"data_name_list length: {len(data_name_list)}")
print(f"graph_list length: {len(graph_list)}")

import torch
torch.cuda.empty_cache()
######## Synthetic Datasets and Graphs 生成合成数据集和图谱
multiple_adata, multiple_graph = deepen._get_multiple_adata(adata_list=augement_data_list, data_name_list=data_name_list, graph_list=graph_list)
###### Enhanced data preprocessing增强数据预处理
data = deepen._data_process(multiple_adata, pca_n_comps=200)
####训练 DeepST 模型并获取嵌入
deepst_embed = deepen._fit(
		data = data,
		graph_dict = multiple_graph,
		domains = multiple_adata.obs["batch"].values,  ##### Input to Domain Adversarial Model
		n_domains = len(data_name_list)
		)
multiple_adata.obsm["DeepST_embed"] = deepst_embed
multiple_adata = deepen._get_cluster_data(multiple_adata, n_domains=n_domains, priori = True)
multiple_adata.write("../Results/multiple_adata.h5ad")
multiple_adata = ad.read_h5ad("../Results/multiple_adata.h5ad")


#绘制批次校正后的umap图
fig, ax_list = plt.subplots(1, 3, figsize=(12, 4))
# sc.pp.neighbors(multiple_adata, use_rep='X', n_neighbors=10, n_pcs=40,random_state=666)
# sc.tl.umap(multiple_adata,random_state=666)
# sc.pl.umap(multiple_adata, color='batch_name', title='Uncorrected', ax=ax_list[0], show=False)
sc.pp.neighbors(multiple_adata, use_rep='DeepST_embed',random_state=666) 
sc.tl.umap(multiple_adata,random_state=666)
sc.pl.umap(multiple_adata, color='batch_name', ax=ax_list[0], title='Batch corrected', show=False)
sc.pl.umap(multiple_adata, color='ground_truth', ax=ax_list[1], title='celltype', show=False)
sc.pl.umap(multiple_adata, color='DeepST_refine_domain', ax=ax_list[2], title='cluster_leiden', show=False)
plt.tight_layout(w_pad=0.05)
plt.savefig(os.path.join(save_path, f'{"_".join(data_name_list)}_umap.pdf'), bbox_inches='tight', dpi=600)



#用scib进行基准测试
import scib
sc.pp.neighbors(multiple_adata, use_rep='DeepST_embed')
scib.me.graph_connectivity(multiple_adata, label_key="ground_truth")
scib.me.ilisi_graph(multiple_adata, batch_key="batch_name",type_="embed",use_rep="DeepST_embed")
scib.me.kBET(multiple_adata, batch_key="batch_name", label_key="ground_truth", type_="embed", embed="DeepST_embed")
scib.me.silhouette_batch(multiple_adata, batch_key="batch_name", label_key="ground_truth", embed="DeepST_embed")#绘制每个数据集的空间图



# # embedding output放弃这个指标 需要未集成的adata

#for data_name in data_name_list:
	#adata = multiple_adata[multiple_adata.obs["batch_name"]==data_name]
	#sc.pl.spatial(adata, color='DeepST_refine_domain', frameon = False, spot_size=150)
	#plt.savefig(os.path.join(save_path, f'{data_name}_domains.pdf'), bbox_inches='tight', dpi=300)
