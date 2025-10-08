"""
Usage Guide
===========

Run with:
    python comparison_umap.py --modes <mode_id> [mode_id2 mode_id3 ...]

Examples:
    python comparison_umap.py --modes 1
    python comparison_umap.py --modes 0 3 5

Available Modes:
    0: Show dataset configuration overview only
    1: Show configuration overview + check all file paths (recommended first run)
    2: Check file paths by dataset type (DLPFC/HBC/Coronal/MOB)

    === Process by dataset type ===
    3: Process DLPFC samples 1–3
    4: Process DLPFC special datasets (7374, all)
    5: Process HBC dataset
    6: Process Coronal dataset
    7: Process MOB dataset

    === Process all ===
    8: Process all datasets

    === Testing & optimized versions ===
    9: Test run (one subset per type)
    10: Memory-efficient run by dataset type
"""

import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

sc.settings.verbosity = 0
sc.settings.set_figure_params(dpi=80, facecolor="white")


def create_layer_palette(adata):
    if "layer" in adata.obs:
        return sns.color_palette("Set2", n_colors=len(adata.obs["layer"].unique()))
    else:
        return sns.color_palette("Set2", n_colors=10)


def load_adata_fast(file_path):
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found, skipping...")
        return None
    try:
        adata = sc.read_h5ad(file_path)
        return adata
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def generate_dlpfc_configs(sample_range=range(1, 4)):
    configs = {}
    for i in sample_range:
        configs[f"dlpfc{i}"] = {
            "raw": f"../DATA_RAW//raw_adata{i}.h5ad",
            "PRECAST": f"../PRECAST/dlpfc/results/{i}dlpfc_seuInt_with_all_spatial.h5ad",
            "DeepST": f"../DeepST/Results/multiple_adata{i}.h5ad",
            "STAligner": f"../STAligner/results/staligner_Sample{i}_DLPFC.h5ad",
            "GraphST": f"../GraphST/results/DLPFC_adata{i}.h5ad",
            "SPIRAL": f"../SPIRAL/results/spiral_Sample{i}_DLPFC.h5ad",
            "STitch3D": f"../STitch3D/results_dlpfc{i}/DLPFC_adata{i}.h5ad",
            "Spatialign": f"../Spatialign/results_dlpfc{i}/multiple_adata{i}.h5ad",
        }
    return configs


DLPFC_CONFIGS = generate_dlpfc_configs(range(1, 4))

DLPFC_SPECIAL_CONFIGS = {
    "dlpfc_7374": {
        "raw": "../DATA_RAW/raw_adata_7374.h5ad",
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
}

HBC_CONFIGS = {
    "hbc": {
        "RAW": "../DATA_RAW/raw_adata_hbc.h5ad",
        "PRECAST": "../PRECAST/hbc/results/hbc_seuInt_with_spatial.h5ad",
        "DeepST": "../DeepST/Results/multiple_adata_hbc.h5ad",
        "STAligner": "../STAligner/results/staligner_hbc.h5ad",
        "GraphST": "../GraphST/results/hbc_adata.h5ad",
        "SPIRAL": "../SPIRAL/results/spiral_hbc.h5ad",
        "Spatialign": "../Spatialign/results_hbc/multiple_adata.h5ad",
    }
}

CORONAL_CONFIGS = {
    "coronal": {
        "RAW": "../DATA_RAW/raw_adata_coronal.h5ad",
        "PRECAST": "../PRECAST/coronal/results/coronal_seuInt_with_spatial.h5ad",
        "DeepST": "../DeepST/Results/multiple_adata_coronal.h5ad",
        "STAligner": "../STAligner/results/staligner_coronal.h5ad",
        "GraphST": "../GraphST/results/coronal_adata.h5ad",
        "SPIRAL": "../SPIRAL/results/spiral_coronal.h5ad",
        "Spatialign": "../Spatialign/results_coronal/multiple_adata.h5ad",
    }
}

MOB_CONFIGS = {
    "mob": {
        "RAW": "../DATA_RAW/raw_adata_mob.h5ad",
        "PRECAST": "../PRECAST/mob/results/MOB_seuInt_with_spatial.h5ad",
        "STAligner": "../STAligner/results/staligner_mob3.h5ad",
        "GraphST": "../GraphST/results/mob_adata_clean.h5ad",
        "SPIRAL": "../SPIRAL/results/spiral_mob3.h5ad",
        "Spatialign": "../Spatialign/results_mob/multiple_adata.h5ad",
    }
}

DATASET_CONFIGS = {}
DATASET_CONFIGS.update(DLPFC_CONFIGS)
DATASET_CONFIGS.update(DLPFC_SPECIAL_CONFIGS)
DATASET_CONFIGS.update(HBC_CONFIGS)
DATASET_CONFIGS.update(CORONAL_CONFIGS)
DATASET_CONFIGS.update(MOB_CONFIGS)


def show_all_configs():
    print("=" * 60)
    print("ALL DATASET CONFIGURATIONS")
    print("=" * 60)

    print(f" DLPFC_CONFIGS ({len(DLPFC_CONFIGS)} datasets):")
    print(f"   {list(DLPFC_CONFIGS.keys())}")

    print(f" DLPFC_SPECIAL_CONFIGS ({len(DLPFC_SPECIAL_CONFIGS)} datasets):")
    print(f"   {list(DLPFC_SPECIAL_CONFIGS.keys())}")

    print(f" HBC_CONFIGS ({len(HBC_CONFIGS)} datasets):")
    print(f"   {list(HBC_CONFIGS.keys())}")

    print(f" CORONAL_CONFIGS ({len(CORONAL_CONFIGS)} datasets):")
    print(f"   {list(CORONAL_CONFIGS.keys())}")

    print(f" MOB_CONFIGS ({len(MOB_CONFIGS)} datasets):")
    print(f"   {list(MOB_CONFIGS.keys())}")

    print(f" Total: {len(DATASET_CONFIGS)} datasets")
    print("=" * 60)


def process_dataset(dataset_name):
    print(f"Processing dataset: {dataset_name}...")

    if dataset_name not in DATASET_CONFIGS:
        print(f"Dataset {dataset_name} not found in configurations")
        return

    file_paths = DATASET_CONFIGS[dataset_name]

    adatas = {}
    for method, path in file_paths.items():
        adata = load_adata_fast(path)
        if adata is not None:
            adatas[method] = adata

    if not adatas:
        print(f"No valid data found for dataset {dataset_name}")
        return

    plot_dataset_umaps(adatas, dataset_name)


def process_single_sample(sample_idx):
    print(f"Processing DLPFC sample {sample_idx}...")

    dataset_name = f"dlpfc{sample_idx}"
    if dataset_name in DLPFC_CONFIGS:
        process_dataset(dataset_name)
    else:
        print(f"DLPFC sample {sample_idx} not found in configurations")


def process_dataset_type(dataset_type):
    type_configs = {
        "dlpfc": DLPFC_CONFIGS,
        "dlpfc_special": DLPFC_SPECIAL_CONFIGS,
        "hbc": HBC_CONFIGS,
        "coronal": CORONAL_CONFIGS,
        "mob": MOB_CONFIGS,
    }

    if dataset_type not in type_configs:
        print(f"Available types: {list(type_configs.keys())}")
        return

    configs = type_configs[dataset_type]

    print(f"Processing {dataset_type.upper()} datasets...")
    for dataset_name in configs.keys():
        try:
            process_dataset(dataset_name)
        except Exception as e:
            print(f"Dataset {dataset_name} failed: {e}")

        import gc

        gc.collect()


def compute_umap_fast(adata, use_rep, n_neighbors=15, random_state=666):
    try:
        sc.pp.neighbors(
            adata,
            use_rep=use_rep,
            n_neighbors=n_neighbors,
            random_state=random_state,
            n_pcs=None,
        )
        sc.tl.umap(adata, random_state=random_state, min_dist=0.5, spread=1.0)
    except Exception as e:
        print(f"UMAP computation failed: {e}")


def plot_dataset_umaps(adatas, dataset_name):
    methods = list(adatas.keys())
    n_methods = len(methods)

    if n_methods == 0:
        return

    fig, axes = plt.subplots(4, n_methods, figsize=(8 * n_methods, 28), dpi=300)
    if n_methods == 1:
        axes = axes.reshape(-1, 1)
    elif len(axes.shape) == 1:
        axes = axes.reshape(-1, 1)

    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    method_params = {
        "raw": {
            "use_rep": "X_pca",
            "batch_col": "new_batch",
            "cell_col": "celltype",
            "cluster_col": "mclust",
        },
        "PRECAST": {
            "use_rep": "PRECAST",
            "batch_col": "new_batch",
            "cell_col": "celltype",
            "cluster_col": "cluster",
        },
        "DeepST": {
            "use_rep": "DeepST_embed",
            "batch_col": "new_batch",
            "cell_col": "celltype",
            "cluster_col": "DeepST_refine_domain",
        },
        "STAligner": {
            "use_rep": "STAligner",
            "batch_col": "new_batch",
            "cell_col": "celltype",
            "cluster_col": "mclust",
        },
        "GraphST": {
            "use_rep": "emb_pca",
            "batch_col": "new_batch",
            "cell_col": "celltype",
            "cluster_col": "domain",
        },
        "SPIRAL": {
            "use_rep": "spiral",
            "batch_col": "new_batch",
            "cell_col": "celltype",
            "cluster_col": "mclust",
        },
        "STitch3D": {
            "use_rep": "latent",
            "batch_col": "new_batch",
            "cell_col": "celltype",
            "cluster_col": "cluster",
        },
        "Spatialign": {
            "use_rep": "correct",
            "batch_col": "new_batch",
            "cell_col": "celltype",
            "cluster_col": "mclust",
        },
    }

    for i, method in enumerate(methods):
        adata = adatas[method]
        params = method_params.get(method, method_params["raw"])

        try:
            if "X_umap" not in adata.obsm:
                print(f"Computing UMAP for {method}...")
                compute_umap_fast(adata, params["use_rep"])

            if params["batch_col"] in adata.obs:
                sc.pl.umap(
                    adata,
                    color=params["batch_col"],
                    ax=axes[0, i],
                    title=f"{method}_Batch",
                    show=False,
                    frameon=False,
                    size=60,
                )
                axes[0, i].title.set_fontsize(28)
            else:
                axes[0, i].set_title(f"{method}_Batch (No batch info)", fontsize=28)
                axes[0, i].axis("off")

            if params["cell_col"] in adata.obs:
                layer_palette = create_layer_palette(adata)
                sc.pl.umap(
                    adata,
                    color=params["cell_col"],
                    ax=axes[1, i],
                    palette=layer_palette,
                    title=f"{method}_Domain",
                    show=False,
                    frameon=False,
                    size=60,
                )
                axes[1, i].title.set_fontsize(28)
            else:
                axes[1, i].set_title(
                    f"{method}_CellType (No celltype info)", fontsize=28
                )
                axes[1, i].axis("off")

            if params["cluster_col"] in adata.obs:
                sc.pl.umap(
                    adata,
                    color=params["cluster_col"],
                    ax=axes[2, i],
                    title=f"{method}_Cluster",
                    show=False,
                    frameon=False,
                    size=60,
                )
                axes[2, i].title.set_fontsize(28)
            else:
                axes[2, i].set_title(f"{method}_Cluster (No cluster info)", fontsize=28)
                axes[2, i].axis("off")

            if "spatial" in adata.obsm and params["cluster_col"] in adata.obs:
                sc.pl.spatial(
                    adata,
                    color=params["cluster_col"],
                    ax=axes[3, i],
                    title=f"{method}_Spatial",
                    spot_size=150,
                    frameon=False,
                    cmap="viridis",
                )
                axes[3, i].title.set_fontsize(28)
            else:
                axes[3, i].set_title(f"{method}_Spatial (No spatial info)", fontsize=28)
                axes[3, i].axis("off")

        except Exception as e:
            print(f"Error plotting {method}: {e}")
            for row in range(4):
                axes[row, i].set_title(f"{method} (Error)", fontsize=28)
                axes[row, i].axis("off")
            continue

    os.makedirs("../results/umap", exist_ok=True)
    plt.savefig(
        f"../results/umap/umap_comparison_{dataset_name}.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()
    print(f"Saved comparison for dataset {dataset_name}")


def batch_process_dataset_type(dataset_type, n_workers=2):
    type_configs = {
        "dlpfc": DLPFC_CONFIGS,
        "dlpfc_special": DLPFC_SPECIAL_CONFIGS,
        "hbc": HBC_CONFIGS,
        "coronal": CORONAL_CONFIGS,
        "mob": MOB_CONFIGS,
    }

    if dataset_type not in type_configs:
        print(f"Available types: {list(type_configs.keys())}")
        return

    configs = type_configs[dataset_type]
    dataset_list = list(configs.keys())

    print(f"Batch processing {dataset_type.upper()}: {dataset_list}")
    batch_process_all_datasets(dataset_list, n_workers)


def batch_process_all_datasets(dataset_list=None, n_workers=4):
    if dataset_list is None:
        dataset_list = list(DATASET_CONFIGS.keys())

    print(f"Starting batch processing for datasets: {dataset_list}")
    print(f"Using {n_workers} workers")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(process_dataset, dataset_name): dataset_name
            for dataset_name in dataset_list
        }

        for future in futures:
            try:
                future.result()
            except Exception as e:
                dataset_name = futures[future]
                print(f"Dataset {dataset_name} failed: {e}")

    print("All datasets processed!")


def quick_run_dlpfc_samples():
    batch_process_dataset_type("dlpfc", n_workers=1)


def quick_run_dlpfc_special():
    batch_process_dataset_type("dlpfc_special", n_workers=2)


def quick_run_hbc():
    batch_process_dataset_type("hbc", n_workers=1)


def quick_run_coronal():
    batch_process_dataset_type("coronal", n_workers=1)


def quick_run_mob():
    batch_process_dataset_type("mob", n_workers=1)

def check_file_paths():
    print("\nChecking file paths...")
    for dataset_name, paths in DATASET_CONFIGS.items():
        print(f"\n=== {dataset_name.upper()} ===")
        for method, path in paths.items():
            exists = "yes" if os.path.exists(path) else "no"
            print(f"{exists} {method}: {path}")


def check_file_paths_by_type():
    show_all_configs()
    check_file_paths()
    type_configs = {
        "DLPFC": DLPFC_CONFIGS,
        "DLPFC_SPECIAL": DLPFC_SPECIAL_CONFIGS,
        "HBC": HBC_CONFIGS,
        "CORONAL": CORONAL_CONFIGS,
        "MOB": MOB_CONFIGS,
    }

    print("Checking file paths by dataset type...")
    for type_name, configs in type_configs.items():
        print(f"\n{'='*50}")
        print(f"=== {type_name} ===")
        print(f"{'='*50}")
        for dataset_name, paths in configs.items():
            print(f"\n--- {dataset_name.upper()} ---")
            for method, path in paths.items():
                exists = "✓" if os.path.exists(path) else "✗"
                print(f"{exists} {method}: {path}")


def memory_efficient_run_by_type():
    print("=== Processing DLPFC Samples ===")
    process_dataset_type("dlpfc")

    print("=== Processing DLPFC Special ===")
    process_dataset_type("dlpfc_special")

    print("=== Processing HBC ===")
    process_dataset_type("hbc")

    print("=== Processing Coronal ===")
    process_dataset_type("coronal")

    print("=== Processing MOB ===")
    process_dataset_type("mob")


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="choose modes to run")
    parser.add_argument(
        "--modes", type=int, nargs="+", required=True, help=" --modes 1 2 3"
    )
    args = parser.parse_args()

    for mode in args.modes:
        print(f"\n=== start {mode} ===")
        if mode == 1:
            memory_efficient_run_by_type()
        elif mode == 2:
            check_file_paths_by_type()
        elif mode == 3:
            quick_run_dlpfc_samples()
        elif mode == 4:
            quick_run_dlpfc_special()
        elif mode == 5:
            quick_run_hbc()
        elif mode == 6:
            quick_run_coronal()
        elif mode == 7:
            quick_run_mob()
        else:
            print(f"no: {mode}")
