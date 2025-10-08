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

# =============== Data Loading Function ===============
load_spatial_data <- function(data_path, dataset_name) {
    cat("Loading", dataset_name, "data...\n")
    
    meta_file <- file.path(data_path, paste0(dataset_name, "_meta.csv"))
    mat_file <- file.path(data_path, paste0(dataset_name, "_mat.csv"))
    coord_file <- file.path(data_path, paste0(dataset_name, "_coord.csv"))
    
    if (!all(file.exists(c(meta_file, mat_file, coord_file)))) {
        stop(paste("Missing files for dataset:", dataset_name))
    }
    
    meta_data <- read.csv(meta_file, row.names = 1, stringsAsFactors = FALSE)
    count_matrix <- read.csv(mat_file, row.names = 1, check.names = FALSE)
    count_matrix <- t(count_matrix)

    coordinates <- read.csv(coord_file, row.names = 1)
    
    common_spots <- intersect(intersect(rownames(meta_data), colnames(count_matrix)), 
                             rownames(coordinates))
    
    meta_data <- meta_data[common_spots, , drop = FALSE]
    count_matrix <- count_matrix[, common_spots, drop = FALSE]
    coordinates <- coordinates[common_spots, , drop = FALSE]
    
    if ("x" %in% colnames(coordinates) && "y" %in% colnames(coordinates)) {
        meta_data$row <- coordinates$x
        meta_data$col <- coordinates$y
    } else if ("row" %in% colnames(coordinates) && "col" %in% colnames(coordinates)) {
        meta_data$row <- coordinates$row
        meta_data$col <- coordinates$col
    } else {
        meta_data$row <- coordinates[, 1]
        meta_data$col <- coordinates[, 2]
    }
    
    meta_data$dataset <- dataset_name
    
    seu <- CreateSeuratObject(
        counts = as.matrix(count_matrix),
        meta.data = meta_data,
        project = paste0("MOB_", dataset_name)
    )
    
    cat("Dataset", dataset_name, "loaded:", ncol(seu), "spots,", nrow(seu), "genes\n")
    return(seu)
}



# =============== Data Setup ===============
input_dir <- "../../SPIRAL/data/mouse_olfactory_bulb/processed/35um/"
results_dir <- "results/"
datasets <- c( 'BGI', 'SlideV2', '10X')

if (!dir.exists(results_dir)) {
    dir.create(results_dir, recursive = TRUE)
}

print("Loading data...")

seuList <- list()
for (i in seq_along(datasets)) {
    dataset <- datasets[i]
    message("Loading dataset ", i, ": ", dataset)
    seuList[[i]] <- load_spatial_data(input_dir, dataset)
}

total_cells <- sum(sapply(seuList, ncol))
total_genes <- nrow(seuList[[1]])  
M <- length(seuList)

saveRDS(seuList, file = file.path(results_dir, "MOB_seuList.rds"))


# =============== Data Preprocessing ===============
print("Performing quality control...")
for (i in seq_along(seuList)) {
    seuList[[i]]$nFeature_RNA <- Matrix::colSums(GetAssayData(seuList[[i]], slot = "counts") > 0)
    seuList[[i]]$nCount_RNA   <- Matrix::colSums(GetAssayData(seuList[[i]], slot = "counts"))
    
    seuList[[i]] <- subset(seuList[[i]], 
                          subset = nFeature_RNA > 200 & nCount_RNA > 300)
    
    cat("Dataset", datasets[i], ": After QC -", ncol(seuList[[i]]), "spots\n")
}
print("Finding common genes...")
common_genes <- Reduce(intersect, lapply(seuList, rownames))
seuList <- lapply(seuList, function(x) x[common_genes, ])
cat("Common genes across all datasets:", length(common_genes), "\n")

# =============== PRECAST Training with Benchmarking ===============
print("Starting PRECAST training benchmarking...")

gc(verbose = FALSE)
memory_before <- get_memory_usage()
training_start_time <- Sys.time()

print("Creating PRECAST object and training...")

set.seed(2025)
PRECASTObj <- CreatePRECASTObject(
    seuList, 
    project = "MOB_MultiPlatform", 
    gene.number = 2000, 
    selectGenesMethod = "SPARK-X",
    premin.spots = 20, 
    premin.features = 20, 
    postmin.spots = 1, 
    postmin.features = 10
)

print("Adding adjacency matrices...")
PRECASTObj <- AddAdjList(PRECASTObj, platform = "Other")

PRECASTObj <- AddParSetting(
    PRECASTObj, 
    Sigma_equal = FALSE, 
    verbose = TRUE, 
    maxIter = 30)

print("Running main PRECAST algorithm...")
PRECASTObj <- PRECAST(PRECASTObj, K = 8)  

training_end_time <- Sys.time()
training_time <- as.numeric(difftime(training_end_time, training_start_time, units = "secs"))
memory_after <- get_memory_usage()
memory_used <- memory_after - memory_before

print("Training completed!")

saveRDS(PRECASTObj, file = file.path(results_dir, "MOB_PRECASTObj.rds"))

# =============== Post-processing ===============
print("Performing post-processing...")
resList <- PRECASTObj@resList
PRECASTObj <- SelectModel(PRECASTObj)

print("Integrating spatial data...")
seuInt <- IntegrateSpaData(PRECASTObj, species = "Mouse")

saveRDS(seuInt, file = file.path(results_dir, "MOB_seuInt.rds"))

# =============== Calculate Final Statistics ===============
final_spots <- sum(unlist(lapply(PRECASTObj@seuList, function(x) ncol(x))))
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
    total_spots = total_cells,
    final_spots = final_spots,
    total_genes = total_genes,
    n_datasets = M,
    max_iterations = 30,
    gene_number = 2000,
    embedding_dim = embedding_dim,
    timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%S")
)

write_json(benchmark_results, 
          file.path(results_dir, "precast_benchmark_mob.json"), 
          pretty = TRUE, auto_unbox = TRUE)


gc(verbose = FALSE)
final_memory <- get_memory_usage()


library(zellkonverter)
library(SingleCellExperiment)
library(Seurat)
sce <- as.SingleCellExperiment(seuInt)
zellkonverter::writeH5AD(sce, "results/MOB_seuInt.h5ad")

import anndata
import scanpy as sc
import pandas as pd
import os
import numpy as np
adata = sc.read_h5ad('results/MOB_seuInt.h5ad')
input_dir = "../../SPIRAL/data/mouse_olfactory_bulb/processed/35um/"

batch_mapping = {
    '1': 'BGI',
    '2': 'SlideV2',
    '3': '10X'
}
adata.obs["new_batch"] = adata.obs["batch"].astype(str).map(batch_mapping)
celltype_mapping = {
    'BGI': {
        'ONL': 'ONL',
        'GL': 'GL',
        'GCL': 'GCL',
        'MCL': 'MCL',
        'OPL': 'OPL',
        'SEZ': 'SEZ',
        'Meninges': 'Meninges',
        'Low_Quality': None
    },
    'SlideV2': {
        'ONL': 'ONL',
        'GL': 'GL',
        'GCL': 'GCL',
        'MCL': 'MCL',
        'EPL': 'EPL',
        'SEZ': 'SEZ'
    },
    '10X': {
        'ONL': 'ONL',
        'GL_1': 'GL',
        'GL_2': 'GL',
        'GCL_1': 'GCL',
        'GCL_2': 'GCL',
        'MCL': 'MCL',
        'EPL': 'EPL'
    }
}

for dataset in adata.obs["new_batch"].unique():
    meta_file = os.path.join(input_dir, f"{dataset}_meta.csv")
    meta = pd.read_csv(meta_file, index_col=0)
    if "celltype" not in meta.columns:
        raise ValueError(f"{dataset}_meta.csv haven't celltype column")
    mapping = celltype_mapping[dataset]
    new_celltypes = meta["celltype"].map(mapping)
    cells_dataset = adata.obs[adata.obs["new_batch"] == dataset].index
    common_cells = meta.index.intersection(cells_dataset)
    adata.obs.loc[common_cells, "celltype"] = new_celltypes.loc[common_cells]

common_cells = adata.obs_names.intersection(adata.obsm['POSITION'].index)
adata = adata[common_cells].copy()
adata.obsm['spatial'] = adata.obsm['POSITION'].loc[adata.obs_names, ["POSITION_1", "POSITION_2"]].to_numpy()
adata.uns["spatial"] = {
    "MOB_dataset": {
        "images": {"hires": np.zeros((1, 1, 3), dtype=np.uint8)},
        "scalefactors": {
            "tissue_hires_scalef": 1.0,
            "spot_diameter_fullres": 50 
        }
    }
}
adata.write ('results/MOB_seuInt_with_spatial.h5ad')
