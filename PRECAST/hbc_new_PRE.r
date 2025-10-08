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
dir.file <- "../../RAW_SLICE/hbc/section"  ## the folders Section1 and Section2
seuList <- list()

print("Loading data...")
data_loading_start <- Sys.time()

for (r in 1:2) {
    message("r = ", r)
    seuList[[r]] <- DR.SC::read10XVisium(paste0(dir.file, r))
}
bc2 <- seuList

# Extract count matrices and metadata
countList <- lapply(bc2, function(x) {
    assay <- DefaultAssay(x)
    GetAssayData(x, assay = assay, layer = "counts")
})

# Extract metadata from original objects
metadataList <- lapply(bc2, function(x) x@meta.data)

M <- length(countList)
seuList <- list()
total_cells <- 0
total_genes <- 0

for (r in 1:M) {
    seu <- CreateSeuratObject(
        counts = countList[[r]],
        meta.data = metadataList[[r]],
        project = "hbc_PRECAST"
    )
    seuList[[r]] <- seu
    
    # Count cells and genes for benchmarking
    total_cells <- total_cells + ncol(seu)
    if (r == 1) total_genes <- nrow(seu)  # Only count genes once
}

bc2 <- seuList
rm(seuList, countList, metadataList)  # Clean up intermediate variables

data_loading_end <- Sys.time()
data_loading_time <- as.numeric(difftime(data_loading_end, data_loading_start, units = "secs"))

saveRDS(bc2, file = "results/hbc_seuList.RDS")

# =============== PRECAST Training with Benchmarking ===============
print("Starting PRECAST training benchmarking...")

# Clear memory before training for fair comparison
gc(verbose = FALSE)
memory_before <- get_memory_usage()
training_start_time <- Sys.time()

print("Training PRECAST model...")

# Create PRECAST object and train
set.seed(2022)
PRECASTObj <- CreatePRECASTObject(bc2, project = "BC2", gene.number = 2000, selectGenesMethod = "SPARK-X",
    premin.spots = 20, premin.features = 20, postmin.spots = 1, postmin.features = 10)
PRECASTObj <- AddAdjList(PRECASTObj, platform = "Visium")
PRECASTObj <- AddParSetting(PRECASTObj, Sigma_equal = FALSE, verbose = TRUE, maxIter = 30)
PRECASTObj <- PRECAST(PRECASTObj, K = 10)
training_end_time <- Sys.time()
training_time <- as.numeric(difftime(training_end_time, training_start_time, units = "secs"))
memory_after <- get_memory_usage()
memory_used <- memory_after - memory_before

print("Training completed!")

saveRDS(PRECASTObj, file = "results/hbc_PRECASTObj.rds")

# =============== Post-processing (not included in benchmark) ===============
print("Performing post-processing...")

resList <- PRECASTObj@resList
print(PRECASTObj@resList)
PRECASTObj <- SelectModel(PRECASTObj)
print(PRECASTObj@seuList)
seuInt <- IntegrateSpaData(PRECASTObj, species = "Human")
print(seuInt)
saveRDS(seuInt, file = "results/hbc_seuInt.rds")

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
    embedding_dim = embedding_dim,
    timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%S")
)

# Create results directory if it doesn't exist
if (!dir.exists("results")) {
    dir.create("results", recursive = TRUE)
}

# Save benchmark results as JSON
write_json(benchmark_results, "results/precast_benchmark_hbc.json", pretty = TRUE, auto_unbox = TRUE)

# =============== Final Memory Cleanup ===============
gc(verbose = FALSE)
final_memory <- get_memory_usage()
cat(sprintf("Final memory usage: %.1f MB\n", final_memory))


library(zellkonverter)
library(SingleCellExperiment)
library(Seurat)
sce <- as.SingleCellExperiment(seuInt)
zellkonverter::writeH5AD(sce, "results/hbc_seuInt.h5ad")

import scib
import anndata
import scanpy as sc
import pandas as pd
import os
import numpy as np

adata = sc.read_h5ad('results/hbc_seuInt.h5ad')
batch_mapping = {'1': 'section1', '2': 'section2'}
adata.obs['new_batch'] = adata.obs['batch'].replace(batch_mapping)
datasets = ['section1', 'section2']
file_fold = '../../RAW_SLICE/hbc/'
all_spatial_coords = []
matching_adata_idx_all = []

new_obs_names = []
for i, obs_name in enumerate(adata.obs_names):
    base_name = obs_name.rsplit('-', 1)[0]
    batch = adata.obs['batch'].iloc[i]
    suffix = '-1' if batch == '1' else '-2'
    new_obs_names.append(base_name + suffix)

adata.obs_names = new_obs_names
adata.obs['celltype'] = 'Unknown'

for dataset in datasets:
    suffix = '-1' if dataset == 'section1' else '-2'
    meta = pd.read_csv(os.path.join(file_fold, dataset, "metadata.csv"), index_col=0)
    common_barcodes = adata.obs_names.intersection(meta.index)
    for barcode in common_barcodes:
        adata.obs.loc[barcode, 'celltype'] = meta.loc[barcode, 'celltype']
    
    spatial_file = os.path.join(file_fold, dataset, 'spatial', 'tissue_positions_list.csv')
    if os.path.exists(spatial_file):
        spatial_data = pd.read_csv(spatial_file, header=None)
        spatial_coords = spatial_data.iloc[:, -2:].values 
        modified_cell_ids = [cell_id.rsplit('-', 1)[0] + suffix for cell_id in spatial_data.iloc[:, 0]]
        
        for j, cell_id in enumerate(modified_cell_ids):
            if cell_id in adata.obs_names:
                adata_idx = adata.obs_names.get_loc(cell_id)
                all_spatial_coords.append(spatial_coords[j])
                matching_adata_idx_all.append(adata_idx)

if all_spatial_coords:
    all_coords = np.full((adata.shape[0], 2), np.nan)
    all_coords[matching_adata_idx_all] = np.array(all_spatial_coords)
    adata.obsm['spatial'] = all_coords

adata.write('results/hbc_seuInt_with_spatial.h5ad')