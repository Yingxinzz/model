suppressPackageStartupMessages(library(Seurat))
suppressPackageStartupMessages(library(SingleCellExperiment))
suppressPackageStartupMessages(library(PRECAST))
suppressPackageStartupMessages(library(jsonlite))

# =============== Memory Monitoring Function ===============
get_memory_usage <- function() {
    # Get memory usage in MB
    mem_info <- gc(verbose = FALSE)
    used_memory <- sum(mem_info[, 2]) # Ncells + Vcells used
    return(used_memory * 8 / 1024^2)  # Convert to MB (8 bytes per cell/vcell)
}

# =============== Data Setup ===============
name_ID4 <- as.character(c(151673, 151674))

### Read data in an online manner
n_ID <- length(name_ID4)
url_brainA <- "https://github.com/feiyoung/DR-SC.Analysis/raw/main/data/DLPFC_data/"
url_brainB <- ".rds"
seuList <- list()

if (!require(ProFAST)) {
    remotes::install_github("feiyoung/ProFAST")
}

options(timeout = 300)

# Count total cells for benchmarking
total_cells <- 0
total_genes <- 0

print("Loading data...")
data_loading_start <- Sys.time()

for (i in 1:n_ID) {
    cat("input brain data", i, "\n")
    # load and read data
    dlpfc <- readRDS(url(paste0(url_brainA, name_ID4[i], url_brainB)))
    count <- dlpfc@assays@data$counts
    row.names(count) <- make.unique(ProFAST::transferGeneNames(row.names(count), species = "Human"))
    seu1 <- CreateSeuratObject(counts = count, meta.data = as.data.frame(colData(dlpfc)), 
                               min.cells = 10, min.features = 10)
    
    # Count cells and genes
    total_cells <- total_cells + ncol(seu1)
    if (i == 1) total_genes <- nrow(seu1)
    
    seuList[[i]] <- seu1
}

data_loading_end <- Sys.time()
data_loading_time <- as.numeric(difftime(data_loading_end, data_loading_start, units = "secs"))

saveRDS(seuList, file = "results/all_dlpfc_seuList.RDS")

# =============== PRECAST Training with Benchmarking ===============
print("Starting PRECAST training benchmarking...")

# Clear memory before training for fair comparison
gc(verbose = FALSE)
memory_before <- get_memory_usage()
training_start_time <- Sys.time()

print("Training PRECAST model...")

# Create PRECAST object
set.seed(2023)
PRECASTObj <- CreatePRECASTObject(seuList = seuList, selectGenesMethod = "HVGs", gene.number = 2000)
PRECASTObj <- AddAdjList(PRECASTObj, platform = "Visium")
PRECASTObj <- AddParSetting(PRECASTObj, Sigma_equal = FALSE, verbose = TRUE, maxIter = 30)
PRECASTObj <- PRECAST(PRECASTObj, K = 7)
training_end_time <- Sys.time()
training_time <- as.numeric(difftime(training_end_time, training_start_time, units = "secs"))
memory_after <- get_memory_usage()
memory_used <- memory_after - memory_before

print("Training completed!")

saveRDS(PRECASTObj, file = "results/all_dlpfc_PRECASTObj.rds")

# =============== Post-processing (not included in benchmark) ===============
print("Performing post-processing...")

## backup the fitting results in resList 
resList <- PRECASTObj@resList
print(PRECASTObj@resList)
PRECASTObj <- SelectModel(PRECASTObj)

print(PRECASTObj@seuList)
seuInt <- IntegrateSpaData(PRECASTObj, species = "Human")
print(seuInt)
saveRDS(seuInt, file = "results/all_dlpfc_seuInt.rds")

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
    data_loading_time_seconds = data_loading_time,
    data_loading_time_minutes = data_loading_time / 60,
    memory_usage_mb = memory_used,
    memory_usage_gb = memory_used / 1024,
    total_cells = total_cells,
    final_cells = final_cells,
    total_genes = total_genes,
    n_datasets = length(name_ID4),
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
write_json(benchmark_results, "results/precast_benchmark_all.json", pretty = TRUE, auto_unbox = TRUE)

# =============== Final Memory Cleanup ===============
gc(verbose = FALSE)
final_memory <- get_memory_usage()
cat(sprintf("Final memory usage: %.1f MB\n", final_memory))

library(zellkonverter)
library(SingleCellExperiment)
library(Seurat)
sce <- as.SingleCellExperiment(seuInt)
zellkonverter::writeH5AD(sce, "results/all_dlpfc_seuInt.h5ad")


import scib
import anndata
import scanpy as sc
import pandas as pd
import os
import numpy as np
adata = sc.read_h5ad('results/all_dlpfc_seuInt.h5ad')
datasets=['151673','151669','151507']
file_fold = '../../RAW_SLICE/DLPFC/'
ground_truth_list = []
spatial_dict = {}

for i, dataset in enumerate(datasets, start=1):  
    truth_file = os.path.join(file_fold, dataset, dataset + '_truth.txt')
    Ann_df = pd.read_csv(truth_file, sep='\t', header=None, index_col=0)
    Ann_df.columns = ['Ground_Truth']
    Ann_df['Ground_Truth'] = Ann_df['Ground_Truth'].fillna("unknown")
    Ann_df.index = [f"{cell[:-2]}-{i+10}" for cell in Ann_df.index]  
    ground_truth_list.append(Ann_df)
    spatial_file = os.path.join(file_fold, dataset, 'spatial', 'tissue_positions_list.csv')
    if os.path.exists(spatial_file):
        spatial_data = pd.read_csv(spatial_file, header=None)
        original_cell_ids = spatial_data.iloc[:, 0].values
        modified_cell_ids = [cell_id[:-2] + f'-{i+10}' for cell_id in original_cell_ids]
        spatial_coords = spatial_data.iloc[:, -2:].values
        spatial_dict.update(dict(zip(modified_cell_ids, spatial_coords)))
    else:
        print(f"warning: file {spatial_file} does not exist")

combined_truth = pd.concat(ground_truth_list)
combined_truth = combined_truth[~combined_truth.index.duplicated(keep='first')]
combined_truth = combined_truth[combined_truth['Ground_Truth'] != "unknown"]
adata.obs['celltype'] = combined_truth['Ground_Truth'].reindex(adata.obs.index)
adata = adata[~adata.obs['celltype'].isnull()].copy()
adata.obs['celltype'] = adata.obs['celltype'].astype('category')
adata_cell_ids = adata.obs.index.values
all_coords = np.full((adata.shape[0], 2), 0.0)

matched = 0
for j, cid in enumerate(adata_cell_ids):
    if cid in spatial_dict:
        all_coords[j] = spatial_dict[cid]
        matched += 1

adata.obsm['spatial'] = all_coords
batch_mapping = {
    '1': '151673',
    '2': '151669',
    '3': '151507',
}
adata.obs['new_batch'] = adata.obs['batch'].replace(batch_mapping)
adata.X = np.nan_to_num(adata.X, nan=0.0)
for k in adata.obsm.keys():
    adata.obsm[k] = np.nan_to_num(adata.obsm[k], nan=0.0)

adata.write('results/all_dlpfc_seuInt_with_all_spatial.h5ad')


