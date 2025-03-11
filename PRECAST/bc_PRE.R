dir.file <- "../../RAW_SLICE/hbc/section"  ## the folders Section1 and Section2, and each includes two folders spatial and filtered_feature_bc_matrix
seuList <- list()
for (r in 1:2) {
    message("r = ", r)
    seuList[[r]] <- DR.SC::read10XVisium(paste0(dir.file, r))
}
bc2 <- seuList
library(PRECAST)
library(Seurat)

head(bc2[[1]])
## Get the gene-by-spot read count matrices countList <- lapply(bc2, function(x)
## x[['RNA']]@counts)
countList <- lapply(bc2, function(x) {
    assay <- DefaultAssay(x)
    GetAssayData(x, assay = assay, layer = "counts")
})

M <- length(countList)
## Get the meta data of each spot for each data batch
metadataList <- lapply(bc2, function(x) x@meta.data)
for (r in 1:M) {
    meta_data <- metadataList[[r]]
    all(c("row", "col") %in% colnames(meta_data))  ## the names are correct!
    head(meta_data[, c("row", "col")])
}


## ensure the row.names of metadata in metaList are the same as that of colnames count matrix
## in countList

for (r in 1:M) {
    row.names(metadataList[[r]]) <- colnames(countList[[r]])
}

## Create the Seurat list object
seuList <- list()
for (r in 1:M) {
    seuList[[r]] <- CreateSeuratObject(counts = countList[[r]], meta.data = metadataList[[r]], project = "BreastCancerPRECAST")
}

bc2 <- seuList
rm(seuList)
head(meta_data[, c("row", "col")])
saveRDS(bc2, file = "results/seuList4.RDS")
library(Seurat)
library(PRECAST)
#seuList <- readRDS("results/seuList4.RDS")
## Create PRECASTObject.
set.seed(2022)
PRECASTObj <- CreatePRECASTObject(bc2, project = "BC2", gene.number = 2000, selectGenesMethod = "SPARK-X",
    premin.spots = 20, premin.features = 20, postmin.spots = 1, postmin.features = 10)

## User can retain the raw seuList by the following commond.  
#PRECASTObj <-CreatePRECASTObject(seuList, customGenelist=row.names(seuList[[1]]), rawData.preserve =TRUE)

#Add the model setting 
## check the number of genes/features after filtering step
PRECASTObj@seulist

## seuList is null since the default value `rawData.preserve` is FALSE.
PRECASTObj@seuList

## Add adjacency matrix list for a PRECASTObj object to prepare for PRECAST model fitting.
PRECASTObj <- AddAdjList(PRECASTObj, platform = "Visium")

## Add a model setting in advance for a PRECASTObj object. verbose =TRUE helps outputing the
## information in the algorithm.
PRECASTObj <- AddParSetting(PRECASTObj, Sigma_equal = FALSE, verbose = TRUE, maxIter = 30)

#Fit PRECAST
### Given K
PRECASTObj <- PRECAST(PRECASTObj, K = 14)
## backup the fitting results in resList
resList <- PRECASTObj@resList
PRECASTObj <- SelectModel(PRECASTObj)
print(PRECASTObj@seuList)
seuInt <- IntegrateSpaData(PRECASTObj, species = "Human")
seuInt
saveRDS(seuInt, file = "results/seuInt.rds")
## The low-dimensional embeddings obtained by PRECAST are saved in PRECAST reduction slot.
cols_cluster <- chooseColors(palettes_name = "Classic 20", n_colors = 14, plot_colors = TRUE)
library(ggplot2)
p12 <- SpaPlot(seuInt, item = "cluster", batch = NULL, point_size = 1, cols = cols_cluster, combine = TRUE,
    nrow.legend = 7)
p12
ggsave("cluster-hb.png", plot = p12, width = 8, height = 6, dpi = 600)

pList <- SpaPlot(seuInt, item = "cluster", batch = NULL, point_size = 1, cols = cols_cluster, combine = FALSE,
    nrow.legend = 7)
drawFigs(pList, layout.dim = c(1, 2), common.legend = TRUE, legend.position = "right", align = "hv")
ggsave("pList-hb.png", plot = arrangeGrob(grobs = pList), width = 12, height = 12, dpi = 600)

seuInt <- AddTSNE(seuInt, n_comp = 2)
p1 <- dimPlot(seuInt, item = "cluster", point_size = 0.5, font_family = "serif", cols = cols_cluster,
    border_col = "gray10", nrow.legend = 14, legend_pos = "right"+theme(plot.margin = margin(10, 10, 10, 30)))  # Times New Roman
p2 <- dimPlot(seuInt, item = "batch", point_size = 0.5, font_family = "serif", legend_pos = "right"+theme(plot.margin = margin(10, 10, 10, 30)))
ggsave("dimPlot-cluster_hbc.png", plot = p1, width = 12, height = 8, dpi = 600)  # 增加高度
ggsave("dimPlot-batch_hbc.png", plot = p2, width = 12, height = 8, dpi = 600)  # 增加高度

library(Seurat)
dat_deg <- FindAllMarkers(seuInt)
library(dplyr)
n <- 10
dat_deg %>%
    group_by(cluster) %>%
    top_n(n = n, wt = avg_log2FC) -> top10

seuInt <- ScaleData(seuInt)
seus <- subset(seuInt, downsample = 400)
color_id <- as.numeric(levels(Idents(seus)))
library(ggplot2)
## HeatMap
p1 <- doHeatmap(seus, features = top10$gene, cell_label = "Domain", grp_label = F, grp_color = cols_cluster[color_id],
    pt_size = 6, slot = "scale.data") + theme(legend.text = element_text(size = 10), legend.title = element_text(size = 13,
    face = "bold"), axis.text.y = element_text(size = 5, face = "italic", family = "serif"))
ggsave("HeatMap-hb.png", plot = p1, width = 8, height = 6, dpi = 600)

# 将 Seurat 对象转换为 SingleCellExperiment 对象
library(zellkonverter)
library(SingleCellExperiment)
library(Seurat)
sce <- as.SingleCellExperiment(seuInt)
zellkonverter::writeH5AD(sce, "results/bc_seuInt.h5ad")

import scib
import anndata
import scanpy as sc
import pandas as pd
import os
import numpy as np
adata = sc.read_h5ad('results/bc_seuInt.h5ad')

batch_mapping = {
    '1': 'section1',
    '2': 'section2',
}
adata_PRECAST.obs['new_batch'] = adata_PRECAST.obs['batch'].replace(batch_mapping)
print(adata_PRECAST.obs['new_batch'])
datasets = ['section1', 'section2']
spatial_dict = {}
file_fold = '../../RAW_SLICE/hbc/'
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
adata.write ('results/bc_seuInt_with_spatial.h5ad')



#adata_PRECAST.write ('results/bc_seuInt.h5ad')
# # import scib
# # import anndata
# # import scanpy as sc
# # import pandas as pd
# # import os
# # import matplotlib.pyplot as plt
# # fig, ax_list = plt.subplots(1, 3, figsize=(12, 4))
# # sc.pp.neighbors(adata_PRECAST, use_rep='X', n_neighbors=10, n_pcs=40,random_state=666)
# # sc.tl.umap(adata_PRECAST,random_state=666)
# # sc.pl.umap(adata_PRECAST, color='new_batch', title='RAW_Uncorrected', ax=ax_list[0], show=False)
# # sc.pp.neighbors(adata_PRECAST, use_rep='PRECAST',random_state=666) 
# # sc.tl.umap(adata_PRECAST,random_state=666)
# # sc.pl.umap(adata_PRECAST, color='new_batch', ax=ax_list[1], title='PRECAST', show=False)
# # sc.pl.umap(adata_PRECAST, color='cluster', ax=ax_list[2], title='cluster_leiden', show=False)
# # plt.tight_layout(w_pad=0.05)
# # plt.savefig('results/umap_comparison_BC.png')

# # sc.pp.neighbors(adata_PRECAST, use_rep="PRECAST")
# # scib.me.graph_connectivity(adata_PRECAST, label_key="cluster")
# # scib.me.ilisi_graph(adata_PRECAST, batch_key="new_batch", type_="embed", use_rep="PRECAST")
# # scib.me.kBET(adata_PRECAST, batch_key="new_batch", label_key="cluster", type_="embed", embed="PRECAST")
# # scib.me.silhouette_batch(adata_PRECAST, batch_key="new_batch", label_key="cluster", embed="PRECAST")