suppressPackageStartupMessages(library(Seurat))
suppressPackageStartupMessages(library(SingleCellExperiment))
name_ID4 <- as.character(c(151669, 151670,151671, 151672))

### Read data in an online manner
n_ID <- length(name_ID4)
url_brainA <- "https://github.com/feiyoung/DR-SC.Analysis/raw/main/data/DLPFC_data/"
url_brainB <- ".rds"
seuList <- list()
if (!require(ProFAST)) {
    remotes::install_github("feiyoung/ProFAST")
}
options(timeout = 300)
for (i in 1:n_ID) {
    # i <- 1
    cat("input brain data", i, "\n")
    # load and read data
    dlpfc <- readRDS(url(paste0(url_brainA, name_ID4[i], url_brainB)))
    count <- dlpfc@assays@data$counts
    row.names(count) <- make.unique(ProFAST::transferGeneNames(row.names(count), species = "Human"))
    seu1 <- CreateSeuratObject(counts = count, meta.data = as.data.frame(colData(dlpfc)), min.cells = 10,
        min.features = 10)
    seuList[[i]] <- seu1
}
# for (i in 1:n_ID) {
#     cat("input brain data", i, "\n")
#     # Load and read data
#     dlpfc <- readRDS(url(paste0(url_brainA, name_ID4[i], url_brainB)))
#     count <- dlpfc@assays@data$counts
#     row.names(count) <- make.unique(ProFAST::transferGeneNames(row.names(count), species = "Human"))
#     # Extract only annotation information
#     meta_data <- data.frame(annotation = dlpfc$annotation)
#     row.names(meta_data) <- colnames(count)
#     seu1 <- CreateSeuratObject(counts = count, meta.data = meta_data, min.cells = 10, min.features = 10)
#     seuList[[i]] <- seu1
# }

saveRDS(seuList, file = "results/2dlpfc_seuList4.RDS")
library(Seurat)
library(PRECAST)
seuList <- readRDS("results/2dlpfc_seuList4.RDS")

metadataList <- lapply(seuList, function(x) x@meta.data)

for (r in seq_along(metadataList)) {
    meta_data <- metadataList[[r]]
    cat(all(c("row", "col") %in% colnames(meta_data)), "\n")
}
set.seed(2023)
#preobj <- CreatePRECASTObject(seuList = seuList, selectGenesMethod = "HVGs", gene.number = 2000,rawData.preserve = TRUE)
PRECASTObj <- CreatePRECASTObject(seuList = seuList, selectGenesMethod = "HVGs", gene.number = 2000)

#Add the model setting
PRECASTObj@seulist
PRECASTObj <- AddAdjList(PRECASTObj, platform = "Visium")
#PRECASTObj <- AddParSetting(PRECASTObj, Sigma_equal = TRUE, coreNum = 1, maxIter = 30, verbose = TRUE)
PRECASTObj <- AddParSetting(PRECASTObj, Sigma_equal = FALSE, verbose = TRUE, maxIter = 30)
#Fit
PRECASTObj <- PRECAST(PRECASTObj, K = 5)
# 保存 PRECASTObj 对象到文件
saveRDS(PRECASTObj, file = "results/2dlpfc_PRECASTObj.rds")

#使用函数SelectModel()重新组织 PRECASTObj 中的拟合结果 对使用 PRECAST 模型的聚类结果进行评估
## backup the fitting results in resList 
resList <- PRECASTObj@resList

PRECASTObj <- SelectModel(PRECASTObj)## 从 PRECASTObj 中选择模型
ari_precast <- sapply(1:length(seuList), function(r) mclust::adjustedRandIndex(PRECASTObj@resList$cluster[[r]],PRECASTObj@seulist[[r]]$layer_guess_reordered))## 计算每个数据集的调整兰德指数（ARI）
mat <- matrix(round(ari_precast, 2), nrow = 1)# 将 ARI 结果整理成矩阵
name_ID4 <- as.character(c(151669, 151670,151671, 151672))
colnames(mat) <- name_ID4
DT::datatable(mat)
# # 保存矩阵到 CSV 文件
write.csv(mat, file = "results/ari2_results.csv", row.names = FALSE)
print(PRECASTObj@seuList)
seuInt <- IntegrateSpaData(PRECASTObj, species = "Human")
seuInt
saveRDS(seuInt, file = "results/2dlpfc_seuInt.rds")
seuInt=readRDS("results/2dlpfc_seuInt.rds")
#######显示聚类的空间散点图
cols_cluster <- chooseColors(palettes_name = "Classic 20", n_colors = 7, plot_colors = TRUE)
library(ggplot2)
p12 <- SpaPlot(seuInt, item = "cluster", batch = NULL, point_size = 1, cols = cols_cluster, combine = TRUE, nrow.legend = 7)
p12
ggsave("cluster2-.png", plot = p12, width = 8, height = 6, dpi = 600)
library(Seurat)
library(ggplot2)
seuInt <- AddTSNE(seuInt, n_comp = 2)
p1 <- dimPlot(seuInt, item = "cluster", point_size = 0.5, font_family = "serif", cols = cols_cluster, border_col = "gray10", nrow.legend = 14, legend_pos = "right")
p2 <- dimPlot(seuInt, item = "batch", point_size = 0.5, font_family = "serif", legend_pos = "right")
drawFigs(list(p1, p2), layout.dim = c(1, 2), legend.position = "right", align = "hv")
combined_plot <- drawFigs(list(p1, p2), layout.dim = c(1, 2), legend.position = "right", align = "hv")
ggsave("batch_cluster2.png", plot = combined_plot, width = 20, height = 8)

# 将 Seurat 对象转换为 SingleCellExperiment 对象
library(zellkonverter)
library(SingleCellExperiment)
library(Seurat)
seuInt <- ScaleData(seuInt)
sce <- as.SingleCellExperiment(seuInt)
zellkonverter::writeH5AD(sce, "results/2dlpfc_seuInt.h5ad")

import scib
import anndata
import scanpy as sc
import pandas as pd
import os
import numpy as np
adata = sc.read_h5ad('results/2dlpfc_seuInt.h5ad')
datasets = ['151669', '151670','151671', '151672']
file_fold = '../../RAW_SLICE/DLPFC/'
ground_truth_list = []
for i, dataset in enumerate(datasets, start=1):  # 从1开始计数
    Ann_df = pd.read_csv(os.path.join(file_fold, dataset, dataset + '_truth.txt'), sep='\t', header=None, index_col=0)
    Ann_df.columns = ['Ground_Truth']
    Ann_df['Ground_Truth'].fillna("unknown", inplace=True)
    Ann_df.index = [f"{cell[:-2]}-{i + 10}" for cell in Ann_df.index]  # 取前部分并加后缀
    ground_truth_list.append(Ann_df)
combined_truth = pd.concat(ground_truth_list)
combined_truth = combined_truth[~combined_truth.index.duplicated(keep='first')]
combined_truth = combined_truth[combined_truth['Ground_Truth'] != "unknown"]
adata.obs['Ground_Truth'] = combined_truth['Ground_Truth'].reindex(adata.obs.index)
adata = adata[~adata.obs['Ground_Truth'].isnull()]
adata.obs['Ground_Truth'] = adata.obs['Ground_Truth'].astype('category')
print(combined_truth.head())
batch_mapping = {
    '1': '151669',
    '2': '151670',
    '3': '151671',
    '4': '151672'
}
adata.obs['new_batch'] = adata.obs['batch'].replace(batch_mapping)
print(adata.obs['new_batch'])
datasets = ['151669', '151670','151671', '151672']
spatial_dict = {}
file_fold = '../../RAW_SLICE/DLPFC/'
all_spatial_coords = []
matching_adata_idx_all = []
for i, dataset in enumerate(datasets, start=1):  
    spatial_file = os.path.join(file_fold, dataset, 'spatial', 'tissue_positions_list.csv')
    if os.path.exists(spatial_file):
        spatial_data = pd.read_csv(spatial_file, header=None)
        original_cell_ids = spatial_data.iloc[:, 0].values
        modified_cell_ids = [cell_id[:-2] + '-11' for cell_id in original_cell_ids]
        spatial_data.iloc[:, 0] = modified_cell_ids
        spatial_coords = spatial_data.iloc[:, -2:].values  
        spatial_dict[dataset] = {'coords': spatial_coords, 'cell_ids': modified_cell_ids}
        adata_cell_ids = adata.obs.index.values
        matching_idx = [i for i, cell_id in enumerate(modified_cell_ids) if cell_id in adata_cell_ids]
        print(f"数据集 {dataset} 匹配的细胞数量：{len(matching_idx)}")
        if len(matching_idx) > 0:
            matched_coords = spatial_coords[matching_idx]
            all_spatial_coords.append(matched_coords)            
            matching_adata_idx_all.extend([np.where(adata_cell_ids == modified_cell_ids[idx])[0][0] for idx in matching_idx])
    else:
        print(f"警告：文件 {spatial_file} 不存在")

all_spatial_coords = np.vstack(all_spatial_coords)
all_coords = np.full((adata.shape[0], 2), np.nan)
all_coords[matching_adata_idx_all] = all_spatial_coords
adata.obsm['spatial'] = all_coords
adata.write('results/2dlpfc_seuInt_with_all_spatial.h5ad')
adata = sc.read_h5ad('results/2dlpfc_seuInt_with_all_spatial.h5ad')
print(adata.obsm.keys())

adata.write('results/2dlpfc_seuInt.h5ad')
adata = sc.read_h5ad('results/2dlpfc_seuInt.h5ad')

# # import matplotlib.pyplot as plt
# # fig, ax_list = plt.subplots(1, 3, figsize=(12, 4))
# # # sc.pp.neighbors(adata, use_rep='X', n_neighbors=10, n_pcs=40,random_state=666)
# # # sc.tl.umap(adata,random_state=666)
# # # sc.pl.umap(adata, color='batch', title='Uncorrected', ax=ax_list[0], show=False)
# # sc.pp.neighbors(adata, use_rep='PRECAST',random_state=666) 
# # sc.tl.umap(adata,random_state=666)
# # sc.pl.umap(adata, color='new_batch', ax=ax_list[0], title='Batch corrected', show=False)
# # sc.pl.umap(adata, color='Ground_Truth', ax=ax_list[1], title='celltype', show=False)
# # sc.pl.umap(adata, color='cluster', ax=ax_list[2], title='cluster_leiden', show=False)
# # plt.tight_layout(w_pad=0.05)
# # plt.savefig('results/umap_comparison_DLPFC2.png')

# # sc.pp.neighbors(adata, use_rep="PRECAST")
# # scib.me.graph_connectivity(adata, label_key="cluster")
# # scib.me.ilisi_graph(adata, batch_key="new_batch", type_="embed", use_rep="PRECAST")
# # scib.me.kBET(adata, batch_key="new_batch", label_key="Ground_Truth", type_="embed", embed="PRECAST")
# # scib.me.silhouette_batch(adata, batch_key="new_batch", label_key="Ground_Truth", embed="PRECAST")
