import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('../')
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import scipy.io
import matplotlib.pyplot as plt
import os
import sys
import STitch3D
import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
np.random.seed(1234)

#加载基因条形码矩阵：
mat = scipy.io.mmread("../GSE144136_GeneBarcodeMatrix_Annotated.mtx")
#加载细胞元数据：
meta = pd.read_csv("../GSE144136_CellNames.csv", index_col=0)
meta.index = meta.x.values
#提取并添加分组、条件和细胞类型信息：
group = [i.split('.')[1].split('_')[0] for i in list(meta.x.values)]
condition = [i.split('.')[1].split('_')[1] for i in list(meta.x.values)]
celltype = [i.split('.')[0] for i in list(meta.x.values)]
meta["group"] = group
meta["condition"] = condition
meta["celltype"] = celltype
#加载基因名称数据
genename = pd.read_csv("../GSE144136_GeneNames.csv", index_col=0)
genename.index = genename.x.values
#构建 AnnData 对象：
adata_ref = ad.AnnData(X=mat.tocsr().T)
adata_ref.obs = meta
adata_ref.var = genename
adata_ref.var_names_make_unique()
#过滤掉非对照条件的细胞：
adata_ref = adata_ref[adata_ref.obs.condition.values.astype(str)=="Control", :]

# 加载数据和注释
anno_df = pd.read_csv('../10X/barcode_level_layer_map.tsv', sep='\t', header=None)
datasets = ['151507', '151508', '151509', '151510']
file_fold = '../../RAW_SLICE/DLPFC/'

adatas = []
for dataset in datasets:
    adata = sc.read_visium(file_fold + str(dataset), count_file=str(dataset) + '_filtered_feature_bc_matrix.h5', load_images=True)
    adata.var_names_make_unique()
    anno_df_slice = anno_df.iloc[anno_df[1].values.astype(str) == str(dataset)]
    anno_df_slice.columns = ["barcode", "slice_id", "layer"]
    anno_df_slice.index = anno_df_slice['barcode']
    adata.obs = adata.obs.join(anno_df_slice, how="left")
    adata = adata[adata.obs['layer'].notna()]
    adatas.append(adata)

# 加载和预处理数据的函数
##spatial data
# adata_st1 = sc.read_visium(file_fold+datasets[0],count_file=datasets[0]+'_filtered_feature_bc_matrix.h5', load_images=True)
# anno_df1 = anno_df.iloc[anno_df[1].values.astype(str) == str(datasets[0])]
# anno_df1.columns = ["barcode", "slice_id", "layer"]
# anno_df1.index = anno_df1['barcode']
# adata_st1.obs = adata_st1.obs.join(anno_df1, how="left")
# adata_st1 = adata_st1[adata_st1.obs['layer'].notna()]

# adata_st2 = sc.read_visium(file_fold+datasets[0],count_file=datasets[0]+'_filtered_feature_bc_matrix.h5', load_images=True)
# anno_df2 = anno_df.iloc[anno_df[1].values.astype(str) == str(datasets[1])]
# anno_df2.columns = ["barcode", "slice_id", "layer"]
# anno_df2.index = anno_df2['barcode']
# adata_st2.obs = adata_st2.obs.join(anno_df2, how="left")
# adata_st2 = adata_st2[adata_st2.obs['layer'].notna()]

# adata_st3 = sc.read_visium(file_fold+datasets[0],count_file=datasets[0]+'_filtered_feature_bc_matrix.h5', load_images=True)
# anno_df3 = anno_df.iloc[anno_df[1].values.astype(str) == str(datasets[2])]
# anno_df3.columns = ["barcode", "slice_id", "layer"]
# anno_df3.index = anno_df3['barcode']
# adata_st3.obs = adata_st3.obs.join(anno_df3, how="left")
# adata_st3 = adata_st3[adata_st3.obs['layer'].notna()]

# adata_st4 = sc.read_visium(file_fold+datasets[0],count_file=datasets[0]+'_filtered_feature_bc_matrix.h5', load_images=True)
# anno_df4 = anno_df.iloc[anno_df[1].values.astype(str) == str(datasets[3])]
# anno_df4.columns = ["barcode", "slice_id", "layer"]
# anno_df4.index = anno_df4['barcode']
# adata_st4.obs = adata_st4.obs.join(anno_df4, how="left")
# adata_st4 = adata_st4[adata_st4.obs['layer'].notna()]

# adatas= [adata_st1, adata_st2, adata_st3, adata_st4]
adata_stitched = STitch3D.utils.align_spots(adatas, plot=True)

# 选择高度可变的基因并构建 3D 空间图
celltype_list_use = ['Astros_1', 'Astros_2', 'Astros_3', 'Endo', 'Micro/Macro',
                     'Oligos_1', 'Oligos_2', 'Oligos_3',
                     'Ex_1_L5_6', 'Ex_2_L5', 'Ex_3_L4_5', 'Ex_4_L_6', 'Ex_5_L5',
                     'Ex_6_L4_6', 'Ex_7_L4_6', 'Ex_8_L5_6', 'Ex_9_L5_6', 'Ex_10_L2_4']

adata, adata_basis = STitch3D.utils.preprocess(adata_stitched,
                                                  adata_ref,
                                                  celltype_ref=celltype_list_use,
                                                  sample_col="group",
                                                  slice_dist_micron=[10., 300., 10.],
                                                  n_hvg_group=500)

# 运行 STitch3D 模型
import time
start_time = time.time()
model = STitch3D.model.Model(adata, adata_basis)
model.train()
end_time = time.time()
# 保存 STitch3D 结果
save_path = "../results_dlpfc3/"
result = model.eval(adatas, save=True, output_path=save_path)


#clustering
from sklearn.mixture import GaussianMixture
np.random.seed(1234)
gm = GaussianMixture(n_components=7, covariance_type='tied', init_params='kmeans')
y = gm.fit_predict(adata.obsm['latent'], y=None)
adata.obs["GM"] = y
adata.obs["GM"].to_csv(os.path.join(save_path, "clustering_result.csv"))
# Restoring clustering labels to result
order = [2,4,6,0,3,5,1] # reordering cluster labels
adata.obs["Cluster"] = [order[label] for label in adata.obs["GM"].values]
for i in range(len(result)):
    result[i].obs["GM"] =adata.obs.loc[result[i].obs_names, ]["GM"]
    result[i].obs["Cluster"] = adata.obs.loc[result[i].obs_names, ]["Cluster"]
adata.obs['Cluster'] = adata.obs['Cluster'].astype('category')
adata.obs["slice_id"] = adata.obs["slice_id"].astype(str)
adata.obs['slice_id'] = adata.obs['slice_id'].str.replace('.0', '', regex=False)
adata.obs["layer"] = adata.obs["layer"].astype("category")
import anndata as ad 
adata.write("../results_dlpfc3/DLPFC_adata3.h5ad")
adata = ad.read_h5ad("../results_dlpfc3/DLPFC_adata3.h5ad")





# # import umap
# # # Step 1: 手动计算 UMAP 嵌入并应用自定义 UMAP 参数
# # reducer = umap.UMAP(n_neighbors=30,
# #                     n_components=2,
# #                     metric="correlation",
# #                     learning_rate=1.0,
# #                     min_dist=0.3,
# #                     spread=1.0,
# #                     set_op_mix_ratio=1.0,
# #                     local_connectivity=1,
# #                     repulsion_strength=1,
# #                     negative_sample_rate=5,
# #                     random_state=1234,
# #                     verbose=True)
# # fig, ax_list = plt.subplots(1, 3, figsize=(12, 4))
# # # 计算 UMAP 并将结果放入 adata.obsm['X_umap']
# # # adata.obsm['X_umap'] = reducer.fit_transform(adata.X)
# # # n_spots = adata.obsm['X_umap'].shape[0]
# # # sc.pl.umap(adata, color='slice_id', title='uncorrected', ax=ax_list[0], show=False)
# # adata.obsm['X_umap'] = reducer.fit_transform(adata.obsm['latent'])  # 使用校正后的表示
# # adata.obsm['X_latent'] = adata.obsm['X_umap']  # 使用校正后的表示
# # n_spots_corrected = adata.obsm['X_latent'].shape[0]
# # sc.pl.umap(adata, color='slice_id', title='corrected', ax=ax_list[0], show=False)
# # sc.pl.umap(adata, color='layer', title='celltype', ax=ax_list[1], show=False)
# # sc.pl.umap(adata, color='Cluster', title='clusters_GM', ax=ax_list[2], show=False)
# # plt.tight_layout(w_pad=0.05)
# # plt.savefig('../results_dlpfc3/umap_comparison3_DLPFC.png')

# # import scib
# # adata.obs["slice_id"] = adata.obs["slice_id"].astype("category")
# # adata.obs['Cluster'] = adata.obs['Cluster'].astype('category')

# # sc.pp.neighbors(adata, use_rep="latent")
# # scib.me.graph_connectivity(adata, label_key="layer")#真实细胞标签
# # scib.me.ilisi_graph(adata, batch_key="slice_id", type_="embed", use_rep="latent")
# # scib.me.kBET(adata, batch_key="slice_id", label_key="layer", type_="embed", embed="latent")
# # scib.me.kBET(adata, batch_key="slice_id", label_key="layer", type_="knn")
# # scib.me.silhouette_batch(adata, batch_key="slice_id", label_key="layer", embed="latent")


