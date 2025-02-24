dir.file <- "../../RAW_SLICE/coronal/"
file_names <- c('FFPE', 'DAPI', 'Normal')  # 文件夹名
seuList <- list()
for (r in file_names) {
    message("Reading files from r: ", r)
    file_path <- paste0(dir.file, r)
    seuList[[r]] <- DR.SC::read10XVisium(file_path)
}
library(PRECAST)
library(Seurat)

coronal <- seuList
head(coronal[[1]])
countList <- lapply(coronal, function(x) {
    assay <- DefaultAssay(x)
    GetAssayData(x, assay = assay, layer = "counts")
})

M <- length(countList)
metadataList <- lapply(coronal, function(x) x@meta.data)
for (r in 1:M) {
    meta_data <- metadataList[[r]]
    all(c("row", "col") %in% colnames(meta_data))  ## the names are correct!
    head(meta_data[, c("row", "col")])
}
for (r in 1:M) {
    row.names(metadataList[[r]]) <- colnames(countList[[r]])
}

## Create the Seurat list object
seuList <- list()
for (r in 1:M) {
    seuList[[r]] <- CreateSeuratObject(counts = countList[[r]], meta.data = metadataList[[r]], project = "coronalPRECAST")
}
coronal <- seuList
rm(seuList)
head(meta_data[, c("row", "col")])
saveRDS(coronal, file = "results/seuList3.RDS")


set.seed(2024)
PRECASTObj <- CreatePRECASTObject(coronal, project = "coronal", gene.number = 2000, selectGenesMethod = "SPARK-X",
    premin.spots = 20, premin.features = 20, postmin.spots = 1, postmin.features = 10)
#PRECASTObj <-CreatePRECASTObject(seuList, customGenelist=row.names(seuList[[1]]), rawData.preserve =TRUE)

## seuList is null since the default value `rawData.preserve` is FALSE.
PRECASTObj@seuList

## Add adjacency matrix list for a PRECASTObj object to prepare for PRECAST model fitting.
PRECASTObj <- AddAdjList(PRECASTObj, platform = "Visium")

## Add a model setting in advance for a PRECASTObj object. verbose =TRUE helps outputing the
## information in the algorithm.
PRECASTObj <- AddParSetting(PRECASTObj, Sigma_equal = FALSE, verbose = TRUE, maxIter = 30)

#Fit PRECAST
### Given K
PRECASTObj <- PRECAST(PRECASTObj, K = 12)
## backup the fitting results in resList
resList <- PRECASTObj@resList
PRECASTObj <- SelectModel(PRECASTObj)
print(PRECASTObj@seuList)
seuInt <- IntegrateSpaData(PRECASTObj, species = "Mouse")
seuInt
saveRDS(seuInt, file = "results/seuInt.rds")


# 将 Seurat 对象转换为 SingleCellExperiment 对象
library(zellkonverter)
library(SingleCellExperiment)
library(Seurat)
sce <- as.SingleCellExperiment(seuInt)
zellkonverter::writeH5AD(sce, "results/coronal_seuInt.h5ad")

import scib
import anndata
import scanpy as sc
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

adata_PRECAST = sc.read_h5ad('results/coronal_seuInt.h5ad')
adata = sc.read_h5ad('results/coronal_seuInt.h5ad')
datasets = ['FFPE', 'DAPI', 'Normal']
suffixes = ['1', '2', '3']  # 对应每个数据集的后缀

ground_truth_list = []

for i, dataset in enumerate(datasets):
    truth_file = os.path.join(file_fold, dataset, dataset + '_truth.csv')
    Ann_df = pd.read_csv(truth_file, sep=',', header=0, index_col=0)
    Ann_df.index = Ann_df.index.str.replace(f'{dataset}-', '', regex=False) + suffixes[i]
    ground_truth_list.append(Ann_df['celltype_new'])

ground_truth_combined = pd.concat(ground_truth_list)

if ground_truth_combined.index.duplicated().any():
    print("重复的索引:", ground_truth_combined.index[ground_truth_combined.index.duplicated()])
    ground_truth_combined = ground_truth_combined[~ground_truth_combined.index.duplicated(keep='first')]

adata.obs['ground_truth'] = ground_truth_combined
print(adata.obs.head())


datasets = ['FFPE', 'DAPI', 'Normal']
spatial_dict = {}
file_fold = '../../RAW_SLICE/coronal/'
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
batch_mapping = {
    '1': 'FFPE',
    '2': 'DAPI',
    '3': 'Normal'
}
adata.obs['new_batch'] = adata.obs['batch'].replace(batch_mapping)
print(adata.obs['new_batch'])
adata.write ('results/bc_seuInt_with_spatial.h5ad')

adata.write('results/coronal_seuInt_truth.h5ad')
#adata_PRECAST.write ('results/bc_seuInt.h5ad')



# # fig, ax_list = plt.subplots(1, 3, figsize=(12, 4))
# # sc.pp.neighbors(adata_PRECAST, use_rep='X', n_neighbors=10, n_pcs=40,random_state=666)
# # sc.tl.umap(adata_PRECAST,random_state=666)
# # sc.pl.umap(adata_PRECAST, color='new_batch', title='RAW_Uncorrected', ax=ax_list[0], show=False)
# # sc.pp.neighbors(adata_PRECAST, use_rep='PRECAST',random_state=666)
# # sc.tl.umap(adata_PRECAST,random_state=666)
# # sc.pl.umap(adata_PRECAST, color='new_batch', ax=ax_list[1], title='PRECAST', show=False)
# # sc.pl.umap(adata_PRECAST, color='cluster', ax=ax_list[2], title='cluster_leiden', show=False)
# # plt.tight_layout(w_pad=0.05)
# # plt.savefig('results/umap_comparison_coronal.png')

# # sc.pp.neighbors(adata_PRECAST, use_rep="PRECAST")
# # scib.me.graph_connectivity(adata_PRECAST, label_key="cluster")
# # scib.me.ilisi_graph(adata_PRECAST, batch_key="new_batch", type_="embed", use_rep="PRECAST")
# # scib.me.kBET(adata_PRECAST, batch_key="new_batch", label_key="cluster", type_="embed", embed="PRECAST")
# # scib.me.silhouette_batch(adata_PRECAST, batch_key="new_batch", label_key="cluster", embed="PRECAST")



