suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(PRECAST))
suppressPackageStartupMessages(library(Seurat))
suppressPackageStartupMessages(library(DR.SC))

# =============== Memory Monitoring Function ===============
get_memory_usage <- function() {
    # Get memory usage in MB
    mem_info <- gc(verbose = FALSE)
    used_memory <- sum(mem_info[, 2]) # Ncells + Vcells used
    return(used_memory * 8 / 1024^2)  # Convert to MB (8 bytes per cell/vcell)
}

# =============== Data Setup ===============
dir.file <- "../../RAW_SLICE/coronal/"
file_names <- c('FFPE', 'DAPI', 'Normal')
seuList <- list()

print("Loading data...")
data_loading_start <- Sys.time()

for (r in file_names) {
    message("Reading files from r: ", r)
    file_path <- paste0(dir.file, r)
    seuList[[r]] <- DR.SC::read10XVisium(file_path)
}
coronal <- seuList
head(coronal[[1]])

countList <- lapply(coronal, function(x) {
    assay <- DefaultAssay(x)
    GetAssayData(x, assay = assay, layer = "counts")
})

M <- length(countList)
metadataList <- lapply(coronal, function(x) x@meta.data)

## Create the Seurat list object
seuList <- list()
total_cells <- 0
total_genes <- 0
for (r in 1:M) {
    seu <- CreateSeuratObject(
        counts = countList[[r]],
        meta.data = metadataList[[r]],
        project = "coronal_PRECAST"
    )
    seuList[[r]] <- seu
    
    # Count cells and genes for benchmarking
    total_cells <- total_cells + ncol(seu)
    if (r == 1) total_genes <- nrow(seu)  # Only count genes once
}

coronal <- seuList
rm(seuList, countList, metadataList)  # Clean up intermediate variables

data_loading_end <- Sys.time()
data_loading_time <- as.numeric(difftime(data_loading_end, data_loading_start, units = "secs"))

saveRDS(coronal, file = "results/coronal_seuList.RDS")

# =============== PRECAST Training with Benchmarking ===============
print("Starting PRECAST training benchmarking...")

# Clear memory before training for fair comparison
gc(verbose = FALSE)
memory_before <- get_memory_usage()
training_start_time <- Sys.time()

print("Training PRECAST model...")

set.seed(2024)
PRECASTObj <- CreatePRECASTObject(coronal, project = "coronal", gene.number = 2000, selectGenesMethod = "SPARK-X",
    premin.spots = 20, premin.features = 20, postmin.spots = 1, postmin.features = 10)
PRECASTObj@seuList
PRECASTObj <- AddAdjList(PRECASTObj, platform = "Visium")
PRECASTObj <- AddParSetting(PRECASTObj, Sigma_equal = FALSE, verbose = TRUE, maxIter = 30)
PRECASTObj <- PRECAST(PRECASTObj, K = 12)
training_end_time <- Sys.time()
training_time <- as.numeric(difftime(training_end_time, training_start_time, units = "secs"))
memory_after <- get_memory_usage()
memory_used <- memory_after - memory_before
print("Training completed!")
saveRDS(PRECASTObj, file = "results/coronal_PRECASTObj.rds")

# =============== Post-processing (not included in benchmark) ===============
print("Performing post-processing...")

resList <- PRECASTObj@resList
print(PRECASTObj@resList)
PRECASTObj <- SelectModel(PRECASTObj)
print(PRECASTObj@seuList)
seuInt <- IntegrateSpaData(PRECASTObj, species = "Mouse")
saveRDS(seuInt, file = "results/coronal_seuInt.rds")

# =============== Calculate final statistics ===============
final_cells <- sum(unlist(lapply(PRECASTObj@seuList, function(x) ncol(x))))
final_genes <- nrow(PRECASTObj@seuList[[1]])
embedding_dim <- ncol(PRECASTObj@resList$hZ[[1]])
# =============== Save Benchmark Results ===============
benchmark_results <- list(
    method_name = "PRECAST",
    training_time_seconds = training_time,
    training_time_minutes = training_time / 60,
    training_time_hours = training_time / 3600,
    memory_usage_mb = memory_used,
    memory_usage_gb = memory_used / 1024,
    total_cells = total_cells,
    final_cells = final_cells,
    total_genes = total_genes,
    n_datasets = M,
    max_iterations = 30,
    gene_number = 2000,
    dataset_type = "coronal",
    dataset_names = file_names,
    embedding_dim = embedding_dim,
    timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%S")
)

# Create results directory if it doesn't exist
if (!dir.exists("results")) {
    dir.create("results", recursive = TRUE)
}

# Save benchmark results as JSON
write_json(benchmark_results, "results/precast_benchmark_coronal.json", pretty = TRUE, auto_unbox = TRUE)

# =============== Final Memory Cleanup ===============
gc(verbose = FALSE)
final_memory <- get_memory_usage()
cat(sprintf("Final memory usage: %.1f MB\n", final_memory))



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

adata = sc.read_h5ad('results/coronal_seuInt.h5ad')
file_fold = '../../RAW_SLICE/coronal/'
suffixes = ['11', '12', '13']
datasets = ['FFPE', 'DAPI', 'Normal']
all_spatial_ids = []
all_spatial_coords = []

for i, dataset in enumerate(datasets):
    spatial_file = os.path.join(file_fold, dataset, 'spatial', 'tissue_positions_list.csv')
    if os.path.exists(spatial_file):
        spatial_data = pd.read_csv(spatial_file, header=None)
        original_cell_ids = spatial_data.iloc[:, 0].values
        modified_cell_ids = [cell_id.replace('-1', f'-{suffixes[i]}') for cell_id in original_cell_ids]
        coords = spatial_data.iloc[:, -2:].values
        all_spatial_ids.extend(modified_cell_ids)  
        all_spatial_coords.append(coords)          
        matched = len(set(modified_cell_ids) & set(adata.obs.index))
        print(f"{dataset}: matched {matched}/{len(modified_cell_ids)} cells")
    else:
        print(f"warning: {spatial_file} not found")

all_spatial_coords = np.vstack(all_spatial_coords)
all_spatial_ids = np.concatenate([np.array(ids) for ids in np.array_split(all_spatial_ids, len(datasets))])
coords_dict = dict(zip(all_spatial_ids, all_spatial_coords))
adata.obsm['spatial'] = np.array([coords_dict.get(idx, [np.nan, np.nan]) for idx in adata.obs.index])

batch_mapping = {
    '1': 'FFPE',
    '2': 'DAPI',
    '3': 'Normal'
}
adata.obs['new_batch'] = adata.obs['batch'].replace(batch_mapping)
adata.obs["new_batch"] = adata.obs["new_batch"].astype(str)
adata.obs_names = adata.obs["new_batch"].astype(str) + '-' + adata.obs_names.str.replace(r'-\d+$', '-1', regex=True)
adata.obs_names = adata.obs_names.str.strip()
ffpe_df = pd.read_csv('../../RAW_SLICE/coronal/FFPE/FFPE_truth.csv', index_col='Unnamed: 0')
dapi_df = pd.read_csv('../../RAW_SLICE/coronal/DAPI/DAPI_truth.csv', index_col='Unnamed: 0')
normal_df = pd.read_csv('../../RAW_SLICE/coronal/Normal/Normal_truth.csv', index_col='Unnamed: 0')
ffpe_df.index = ffpe_df.index.str.strip()
dapi_df.index = dapi_df.index.str.strip()
normal_df.index = normal_df.index.str.strip()
ffpe_ground_truth = ffpe_df["celltype_new"].to_dict()
dapi_ground_truth = dapi_df["celltype_new"].to_dict()
normal_ground_truth = normal_df["celltype_new"].to_dict()
combined_ground_truth = {**ffpe_ground_truth, **dapi_ground_truth, **normal_ground_truth}
adata.obs['celltype'] = adata.obs.index.map(combined_ground_truth)

adata.write ('results/coronal_seuInt_with_spatial.h5ad')

