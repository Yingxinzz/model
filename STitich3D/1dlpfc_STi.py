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
anno_df = pd.read_csv('../barcode_level_layer_map.tsv', sep='\t', header=None)
datasets = ['151673', '151674', '151675', '151676']
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


# 对齐 ST 组织切片（原始的切片）
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
save_path = "../results_dlpfc1/"
result = model.eval(adatas, save=True, output_path=save_path)

#clustering
from sklearn.mixture import GaussianMixture
np.random.seed(1234)
gm = GaussianMixture(n_components=7, covariance_type='tied', init_params='kmeans')
y = gm.fit_predict(model.adata_st.obsm['latent'], y=None)
model.adata_st.obs["GM"] = y
model.adata_st.obs["GM"].to_csv(os.path.join(save_path, "clustering_result.csv"))

order = [2,4,6,0,3,5,1] # reordering cluster labels
model.adata_st.obs["Cluster"] = model.adata_st.obs["GM"].map(lambda x: order[x])
for i in range(len(result)):
    result[i].obs["GM"] = model.adata_st.obs.loc[result[i].obs_names, "GM"]
    result[i].obs["Cluster"] = model.adata_st.obs.loc[result[i].obs_names, "Cluster"]

# Reorder cluster labels
order = [2, 4, 6, 0, 3, 5, 1]  # Desired reordering
gm_to_int = {gm: i for i, gm in enumerate(sorted(model.adata_st.obs["GM"].unique()))}
model.adata_st.obs["GM"] = model.adata_st.obs["GM"].map(gm_to_int)
model.adata_st.obs["Cluster"] = model.adata_st.obs["GM"].map(lambda x: order[x])
for i in range(len(result)):
    result[i].obs["GM"] = model.adata_st.obs.loc[result[i].obs_names, "GM"]
    result[i].obs["Cluster"] = model.adata_st.obs.loc[result[i].obs_names, "Cluster"]

# 去掉 .0
adata.obs["slice_id"] = adata.obs["slice_id"].astype(str)
adata.obs['slice_id'] = adata.obs['slice_id'].str.replace('.0', '', regex=False)
adata.obs['Cluster'] = adata.obs['Cluster'].astype('category')

import anndata as ad 
adata.write("../results_dlpfc1/DLPFC_adata.h5ad")
adata = ad.read_h5ad("../results_dlpfc1/DLPFC_adata.h5ad")






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
# # adata.obsm['X_latent']=adata.obsm['X_umap']
# # n_spots_corrected = adata.obsm['X_latent'].shape[0]
# # sc.pl.umap(adata, color='slice_id', title='corrected', ax=ax_list[0], show=False)
# # sc.pl.umap(adata, color='layer', title='celltype', ax=ax_list[1],  palette=layer_palette, show=False)
# # sc.pl.umap(adata, color='Cluster', title='clusters_GM', ax=ax_list[2], show=False)
# # plt.tight_layout(w_pad=0.05)
# # plt.savefig('../results_dlpfc1/umap_comparison_DLPFC.png')


# # import scib
# # adata.obs["layer"] = adata.obs["layer"].astype("category")
# # adata.obs["slice_id"] = adata.obs["slice_id"].astype("category")

# # sc.pp.neighbors(adata, use_rep="latent")
# # scib.me.graph_connectivity(adata, label_key="layer")
# # scib.me.ilisi_graph(adata, batch_key="slice_id", type_="embed", use_rep="latent")
# # scib.me.kBET(adata, batch_key="slice_id", label_key="layer", type_="embed", embed="latent")
# # scib.me.kBET(adata, batch_key="slice_id", label_key="layer", type_="knn")
# # scib.me.silhouette_batch(adata, batch_key="slice_id", label_key="layer", embed="latent")


