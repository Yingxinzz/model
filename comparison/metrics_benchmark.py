import os
import pandas as pd
import numpy as np
import scanpy as sc
import squidpy as sq
import scib
from tqdm import tqdm
import sys
import scipy.sparse as sp

sys.path.append("../")
from spatial_metrics import spatialbench
import warnings

warnings.filterwarnings("ignore")

import psutil
import os
import gc
import torch

os.makedirs("../results/individual_models", exist_ok=True)

METHOD_BASE_CONFIG = {
    "RAW": {
        "cluster_key": "mclust",
        "batch_key": "new_batch",
        "celltype_key": "celltype",
        "use_rep": "X_pca",
    },
    "PRECAST": {
        "cluster_key": "cluster",
        "batch_key": "new_batch",
        "celltype_key": "celltype",
        "use_rep": "PRECAST",
    },
    "DeepST": {
        "cluster_key": "DeepST_refine_domain",
        "batch_key": "new_batch",
        "celltype_key": "celltype",
        "use_rep": "DeepST_embed",
    },
    "STAligner": {
        "cluster_key": "mclust",
        "batch_key": "new_batch",
        "celltype_key": "celltype",
        "use_rep": "STAligner",
    },
    "GraphST": {
        "cluster_key": "domain",
        "batch_key": "new_batch",
        "celltype_key": "celltype",
        "use_rep": "emb_pca",
    },
    "SPIRAL": {
        "cluster_key": "mclust",
        "batch_key": "new_batch",
        "celltype_key": "celltype",
        "use_rep": "spiral",
    },
    "STitch3D": {
        "cluster_key": "cluster",
        "batch_key": "new_batch",
        "celltype_key": "celltype",
        "use_rep": "latent",
    },
    "Spatialign": {
        "cluster_key": "mclust",
        "batch_key": "new_batch",
        "celltype_key": "celltype",
        "use_rep": "correct",
    },
}

DATASET_PATH_CONFIGS = {
    "dlpfc": {
        "base_samples": list(range(1, 4)),
        "path_templates": {
            "RAW": "../DATA_RAW/raw_adata{}.h5ad",
            "PRECAST": "../PRECAST/dlpfc/results/{}dlpfc_seuInt_with_all_spatial.h5ad",
            "DeepST": "../DeepST/Results/multiple_adata{}.h5ad",
            "STAligner": "../STAligner/results/staligner_Sample{}_DLPFC.h5ad",
            "GraphST": "../GraphST/results/DLPFC_adata{}.h5ad",
            "SPIRAL": "../SPIRAL/results/spiral_Sample{}_DLPFC.h5ad",
            "STitch3D": "../STitch3D/results_dlpfc{0}/DLPFC_adata{0}.h5ad",
            "Spatialign": "../Spatialign/results_dlpfc{0}/multiple_adata{0}.h5ad",
        },
    },
    "dlpfc_7374": {
        "RAW": "../DATA_RAW/raw_adata_7374.h5ad",
        "PRECAST": "../PRECAST/dlpfc/results/7374_dlpfc_seuInt_with_all_spatial.h5ad",
        "DeepST": "../DeepST/Results/multiple_adata_7374.h5ad",
        "STAligner": "../STAligner/results/staligner_Sample_7374_DLPFC.h5ad",
        "GraphST": "../GraphST/results/DLPFC_adata_7374.h5ad",
        "SPIRAL": "../SPIRAL/results/spiral_Sample_DLPFC_7374.h5ad",
        "STitch3D": "../STitch3D/results_dlpfc_7374/DLPFC_adata7374.h5ad",
        "Spatialign": "../Spatialign/results_dlpfc1/multiple_adata_7374.h5ad",
    },
    "dlpfc_all": {
        "RAW": "../DATA_RAW/raw_adata_all.h5ad",
        "PRECAST": "../PRECAST/dlpfc/results/all_dlpfc_seuInt_with_all_spatial.h5ad",
        "DeepST": "../DeepST/Results/multiple_adata_all.h5ad",
        "STAligner": "../STAligner/results/staligner_all_DLPFC.h5ad",
        "GraphST": "../GraphST/results/DLPFC_adata_all.h5ad",
        "SPIRAL": "../SPIRAL/results/spiral_Sample_DLPFC_all.h5ad",
        "STitch3D": "../STitch3D/results_dlpfc_all/DLPFC_adata_all.h5ad",
        "Spatialign": "../Spatialign/results_dlpfc_all/multiple_adata_all.h5ad",
    },
    "hbc": {
        "RAW": "../DATA_RAW/raw_adata_hbc.h5ad",
        "PRECAST": "../PRECAST/hbc/results/hbc_seuInt_with_spatial.h5ad",
        "DeepST": "../DeepST/Results/multiple_adata_hbc.h5ad",
        "STAligner": "../STAligner/results/staligner_hbc.h5ad",
        "GraphST": "../GraphST/results/hbc_adata.h5ad",
        "SPIRAL": "../SPIRAL/results/spiral_hbc.h5ad",
        "Spatialign": "../Spatialign/results_hbc/multiple_adata.h5ad",
    },
    "coronal": {
        "RAW": "../DATA_RAW/raw_adata_coronal.h5ad",
        "PRECAST": "../PRECAST/coronal/results/coronal_seuInt_with_spatial.h5ad",
        "DeepST": "../DeepST/Results/multiple_adata_coronal.h5ad",
        "STAligner": "../STAligner/results/staligner_coronal.h5ad",
        "GraphST": "../GraphST/results/coronal_adata.h5ad",
        "SPIRAL": "../SPIRAL/results/spiral_coronal.h5ad",
        "Spatialign": "../Spatialign/results_coronal/multiple_adata.h5ad",
    },
    "mob": {
        "RAW": "../DATA_RAW/raw_adata_mob.h5ad",
        "PRECAST": "../PRECAST/mob/results/MOB_seuInt_with_spatial.h5ad",
        "STAligner": "../STAligner/results/staligner_mob3.h5ad",
        "GraphST": "../GraphST/results/mob_adata_clean.h5ad",
        "SPIRAL": "../SPIRAL/results/spiral_mob3.h5ad",
        "Spatialign": "../Spatialign/results_mob/multiple_adata.h5ad",
    },
}

BATCH_METRICS = [
    "GC",
    "iLISI",
    "kBET",
    "ASW_batch",
]
_METRICS = ["SCS", "MoranI", "GearyC", "cLISI", "ASW_domain", "ARI"]
ALL_METRICS = [
    "SCS",
    "MoranI",
    "GearyC",
    "cLISI",
    "ASW_domain",
    "GC",
    "iLISI",
    "kBET",
    "ASW_batch",
    "ARI",
]


def clear_memory():
    import gc

    gc.collect()
    print("    Memory cleared")


def compute_metrics(adata, config, degree=4, seed=0, top_n=3, safe_mode=True):

    def clean_matrix(mat):

        if sp.issparse(mat):
            mat = mat.astype(np.float32).tocoo(copy=True)
            mat.data = np.nan_to_num(mat.data, nan=0.0, posinf=0.0, neginf=0.0)
            return mat.tocsr()
        else:
            mat = np.array(mat, dtype=np.float32, copy=True)
            mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
            return mat

    if safe_mode:
        adata.X = clean_matrix(adata.X)

    results = {}

    if adata.raw is None:
        adata.raw = adata.copy()
    else:
        adata.raw._X = clean_matrix(adata.raw.X)

    adata_scib = adata.copy()
    adata_raw_clean = adata.raw.to_adata()
    adata_raw_clean.X = clean_matrix(adata_raw_clean.X)

    use_rep = config.get("use_rep", "X_pca")
    if use_rep in adata_scib.obsm:
        adata_scib.obsm[use_rep] = clean_matrix(adata_scib.obsm[use_rep])
    try:
        print("  Computing SCS...")
        scs = spatialbench.spatial_coherence_score(
            adata_raw_clean,
            annotation_key=config["cluster_key"],
            degree=degree,
            seed=seed,
        )
        results["SCS"] = scs
        clear_memory()
    except Exception as e:
        print(f"    SCS failed: {e}")
        results["SCS"] = np.nan

    try:
        print("  Computing  Moran I & Geary C...")
        moranI, gearyC = spatialbench.moran_geary_preservation(
            adata_raw_clean, celltype_key=config["celltype_key"], top_n=top_n
        )
        results["MoranI"] = moranI
        results["GearyC"] = gearyC
        print("DEBUG  MoranI:", moranI)
        print("DEBUG  GearyC:", gearyC)
        clear_memory()
    except Exception as e:
        print(f"     MoranI & GearyC failed: {e}")
        results["MoranI"] = np.nan
        results["GearyC"] = np.nan

    try:
        sc.pp.neighbors(adata_scib, use_rep=use_rep, n_neighbors=15, random_state=seed)
        scib_metrics = [
            (
                "ARI",
                lambda: scib.me.ari(
                    adata_scib,
                    cluster_key=config["cluster_key"],
                    label_key=config["celltype_key"],
                ),
            ),
            (
                "GC",
                lambda: scib.me.graph_connectivity(
                    adata_scib, label_key=config["celltype_key"]
                ),
            ),
            (
                "ASW_domain",
                lambda: scib.me.silhouette(
                    adata_scib, label_key=config["celltype_key"], embed=use_rep
                ),
            ),
            (
                "ASW_batch",
                lambda: scib.me.silhouette_batch(
                    adata_scib,
                    batch_key=config["batch_key"],
                    label_key=config["celltype_key"],
                    embed=use_rep,
                ),
            ),
            (
                "iLISI",
                lambda: scib.me.ilisi_graph(
                    adata_scib,
                    batch_key=config["batch_key"],
                    type_="embed",
                    use_rep=use_rep,
                ),
            ),
            (
                "cLISI",
                lambda: scib.me.clisi_graph(
                    adata_scib,
                    label_key=config["celltype_key"],
                    type_="embed",
                    use_rep=use_rep,
                ),
            ),
            (
                "kBET",
                lambda: scib.me.kBET(
                    adata_scib,
                    batch_key=config["batch_key"],
                    label_key=config["celltype_key"],
                    type_="embed",
                    embed=use_rep,
                ),
            ),
        ]
        for metric_name, compute_func in scib_metrics:
            try:
                print(f"  Computing {metric_name}...")
                results[metric_name] = compute_func()
                clear_memory()
            except Exception as e:
                print(f"    {metric_name} failed: {e}")
                results[metric_name] = np.nan
    except Exception as e:
        print(f"Neighbors graph construction failed: {e}")
        for name in [
            "GC",
            "iLISI",
            "cLISI",
            "ASW_domain",
            "ASW_batch",
            "ARI",
            "kBET",
        ]:
            results[name] = np.nan

    return results


def process_dlpfc_sample(sample_num, method=None):
    print(f"\n{'='*50}")
    print(f"Processing DLPFC Sample {sample_num}")
    print(f"{'='*50}")
    dataset_config = DATASET_PATH_CONFIGS["dlpfc"]
    results = []
    methods_to_run = (
        {method: METHOD_BASE_CONFIG[method]} if method else METHOD_BASE_CONFIG
    )

    for method_name, method_config in methods_to_run.items():
        print(f"\nProcessing method: {method_name}")
        path_template = dataset_config["path_templates"][method_name]
        file_path = path_template.format(sample_num)

        if os.path.exists(file_path):
            print(f"  Loading: {file_path}")
            try:
                adata = sc.read_h5ad(file_path)
                print(f"  Data shape: {adata.shape}")
                adata = adata.copy()
                metrics = compute_metrics(adata, method_config)
                metrics["Method"] = method_name
                metrics["Sample"] = f"sample_{sample_num}"
                results.append(metrics)

                del adata
                clear_memory()

            except Exception as e:
                print(f"  Error processing {file_path}: {e}")
        else:
            print(f"  File not found: {file_path}")

    if results:
        df = pd.DataFrame(results)
        os.makedirs("../results/individual_models", exist_ok=True)

        mode = "a" if method else "w"

        all_metrics_df = df[["Method", "Sample"] + ALL_METRICS]
        all_metrics_path = (
            f"../results/individual_models/dlpfc_sample{sample_num}_all_metrics.csv"
        )
        all_metrics_df.to_csv(
            all_metrics_path,
            mode=mode,
            header=not (os.path.exists(all_metrics_path) and method),
            index=False,
        )
        print(f"\nSaved all metrics to: {all_metrics_path}")

        batch_metrics_df = df[["Method", "Sample"] + BATCH_METRICS]
        batch_metrics_path = (
            f"../results/individual_models/dlpfc_sample{sample_num}_batch_metrics.csv"
        )
        batch_metrics_df.to_csv(
            batch_metrics_path,
            mode=mode,
            header=not (os.path.exists(batch_metrics_path) and method),
            index=False,
        )
        print(f"Saved batch metrics to: {batch_metrics_path}")

        _metrics_df = df[["Method", "Sample"] + _METRICS]
        _metrics_path = (
            f"../results/individual_models/dlpfc_sample{sample_num}__metrics.csv"
        )
        _metrics_df.to_csv(
            _metrics_path,
            mode=mode,
            header=not (os.path.exists(_metrics_path) and method),
            index=False,
        )
        print(f"Saved  metrics to: {_metrics_path}")

    clear_memory()
    print(f"\n DLPFC Sample {sample_num} completed successfully!")


def process_other_dataset(dataset_name, method=None):
    print(f"\n{'='*50}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'='*50}")

    dataset_config = DATASET_PATH_CONFIGS[dataset_name]
    results = []
    methods_to_run = (
        {method: METHOD_BASE_CONFIG[method]} if method else METHOD_BASE_CONFIG
    )

    for method_name, method_config in methods_to_run.items():
        print(f"\nProcessing method: {method_name}")

        if method_name in dataset_config:
            file_path = dataset_config[method_name]
            if os.path.exists(file_path):
                print(f"  Loading: {file_path}")
                try:
                    adata = sc.read_h5ad(file_path)
                    print(f"  Data shape: {adata.shape}")
                    adata = adata.copy()
                    metrics = compute_metrics(adata, method_config)
                    metrics["Method"] = method_name
                    metrics["Sample"] = dataset_name
                    results.append(metrics)

                    del adata
                    clear_memory()

                except Exception as e:
                    print(f"  Error processing {file_path}: {e}")
            else:
                print(f"  File not found: {file_path}")
        else:
            print(f"  Method {method_name} not available for {dataset_name}")

    if results:
        df = pd.DataFrame(results)
        os.makedirs("../results/individual_models", exist_ok=True)
        mode = "a" if method else "w"
        all_metrics_df = df[["Method"] + ALL_METRICS]
        all_metrics_path = (
            f"../results/individual_models/{dataset_name}_all_metrics.csv"
        )
        all_metrics_df.to_csv(
            all_metrics_path,
            mode=mode,
            header=not (os.path.exists(all_metrics_path) and method),
            index=False,
        )
        print(f"\nSaved all metrics to: {all_metrics_path}")

        batch_metrics_df = df[["Method"] + BATCH_METRICS]
        batch_metrics_path = (
            f"../results/individual_models/{dataset_name}_batch_metrics.csv"
        )
        batch_metrics_df.to_csv(
            batch_metrics_path,
            mode=mode,
            header=not (os.path.exists(batch_metrics_path) and method),
            index=False,
        )
        print(f"Saved batch metrics to: {batch_metrics_path}")

        _metrics_df = df[["Method"] + _METRICS]
        _metrics_path = (
            f"../results/individual_models/{dataset_name}__metrics.csv"
        )
        _metrics_df.to_csv(
            _metrics_path,
            mode=mode,
            header=not (os.path.exists(_metrics_path) and method),
            index=False,
        )
        print(f"Saved  metrics to: {_metrics_path}")

    clear_memory()
    print(f"\n Dataset {dataset_name} completed successfully!")


def process_all_datasets_safely():
    for dataset_name in DATASET_PATH_CONFIGS.keys():
        try:
            if dataset_name == "dlpfc":
                for i in range(1, 4):
                    process_dlpfc_sample(i)
                    print(f"\n DLPFC Sample {i} finished! Results saved to:")
                    print(
                        f"   - ../results/individual_models/dlpfc_sample{i}_all_metrics.csv"
                    )
                    print(
                        f"   - ../results/individual_models/dlpfc_sample{i}_batch_metrics.csv"
                    )
                    print(
                        f"   - ../results/individual_models/dlpfc_sample{i}__metrics.csv"
                    )
            else:
                process_other_dataset(dataset_name)
                print(f"\n Dataset {dataset_name} finished! Results saved to:")
                print(
                    f"   - ../results/individual_models/{dataset_name}_all_metrics.csv"
                )
                print(
                    f"   - ../results/individual_models/{dataset_name}_batch_metrics.csv"
                )
                print(
                    f"   - ../results/individual_models/{dataset_name}__metrics.csv"
                )
        except Exception as e:
            print(f"\n Error processing {dataset_name}: {e}")
            continue


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Spatial Transcriptomics Benchmark Tool"
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        help="Dataset name to process (e.g., dlpfc_sample1, hbc, coronal)",
    )
    parser.add_argument(
        "--method",
        "-m",
        type=str,
        help="Specify a single method to run (e.g., PRECAST, DeepST, GraphST)",
    )
    parser.add_argument(
        "--list", "-l", action="store_true", help="List all available datasets"
    )
    parser.add_argument("--all", "-a", action="store_true", help="Process all datasets")
    parser.add_argument(
        "--summary",
        "-s",
        type=str,
        help="Show dataset configuration paths (e.g., -s dlpfc)",
    )

    args = parser.parse_args()

    available_datasets = sorted(DATASET_PATH_CONFIGS.keys())

    if args.list:
        print("Available datasets:")
        for i, dataset in enumerate(available_datasets, 1):
            print(f"  {i}. {dataset}")

    elif args.summary:
        dataset_name = args.summary.lower()
        if dataset_name in DATASET_PATH_CONFIGS:
            print(f"\nSummary of dataset: {dataset_name}")
            for method, path in DATASET_PATH_CONFIGS[dataset_name].items():
                print(f"  {method:10s} : {path}")
        else:
            print(f"Error: Unknown dataset '{dataset_name}'")
            print("Use --list to see available datasets")

    elif args.dataset:
        dataset_name = args.dataset.lower()
        method_name = args.method

        if dataset_name.startswith("dlpfc_sample"):
            try:
                sample_num = int(dataset_name.replace("dlpfc_sample", ""))
                if method_name:
                    process_dlpfc_sample(sample_num, method=method_name)
                    print(f"Successfully completed: {dataset_name} with {method_name}")
                else:
                    process_dlpfc_sample(sample_num)
                    print(f"Successfully completed: {dataset_name}(all methods)")
            except Exception as e:
                print(f"Error processing {dataset_name}: {e}")

        elif dataset_name in available_datasets:
            try:
                if method_name:
                    process_other_dataset(dataset_name, method=method_name)
                else:
                    process_other_dataset(dataset_name)

                print(f"Successfully completed: {dataset_name}")
            except Exception as e:
                print(f"Error processing {dataset_name}: {e}")
        else:
            print(f"Error: Unknown dataset '{dataset_name}'")
            print("Use --list to see available datasets")

    elif args.all:
        print("Processing all datasets...")
        try:
            process_all_datasets_safely()
            print("Successfully completed all datasets!")
        except Exception as e:
            print(f"Error during batch processing: {e}")


if __name__ == "__main__":
    main()
