import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings

warnings.filterwarnings("ignore")

plt.style.use("default")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10

os.makedirs("../results/figures", exist_ok=True)
DATASETS = [
    "dlpfc_sample1",
    "dlpfc_sample2",
    "dlpfc_sample3",
    "dlpfc_7374",
    "dlpfc_all",
    "hbc",
    "coronal",
    "mob",
]
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
BATCH_METRICS = [
    "GC",
    "iLISI",
    "kBET",
    "ASW_batch",
]
BIO_METRICS = ["MoranI", "GearyC", "cLISI", "ASW_domain", "ARI"]
METHODS = [
    "RAW",
    "PRECAST",
    "DeepST",
    "STAligner",
    "GraphST",
    "SPIRAL",
    "STitch3D",
    "Spatialign",
]

METHOD_COLORS = {
    "RAW": "#2B6CB0",
    "PRECAST": "#3182CE",
    "DeepST": "#00B5D8",
    "STAligner": "#38A169",
    "GraphST": "#ED8936",
    "SPIRAL": "#805AD5",
    "STitch3D": "#319795",
    "Spatialign": "#FF6B6B",
}


def load_dataset_results(dataset_name):
    base_path = "../results/individual_models"

    try:
        all_metrics_path = f"{base_path}/{dataset_name}_all_metrics.csv"
        batch_metrics_path = f"{base_path}/{dataset_name}_batch_metrics.csv"
        bio_metrics_path = f"{base_path}/{dataset_name}_bio_metrics.csv"

        all_df = pd.read_csv(all_metrics_path)
        batch_df = pd.read_csv(batch_metrics_path)
        bio_df = pd.read_csv(bio_metrics_path)

        print(f"Loaded {dataset_name}: {all_df.shape[0]} records")
        return all_df, batch_df, bio_df, True

    except FileNotFoundError as e:
        print(f"Warning: Files for {dataset_name} not found: {e}")
        return None, None, None, False
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        return None, None, None, False


def plot_individual_metrics_lines(df, dataset_name, save_dir):
    if df is None or df.empty:
        return

    df_plot = df.copy()

    n_metrics = len(ALL_METRICS)
    n_rows = 2
    n_cols = 5

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 10))
    
    axes = axes.flatten()

    for i, metric in enumerate(ALL_METRICS):
        ax = axes[i]

        valid_data = df_plot[df_plot[metric].notna()]
        if valid_data.empty:
            ax.set_title(f"{metric}\n(No Data)")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        methods = valid_data["Method"].values
        scores = valid_data[metric].values

        ax.plot(methods, scores, "o-", linewidth=2.5, markersize=8, alpha=0.8)

        for j, (method, score) in enumerate(zip(methods, scores)):
            ax.scatter(
                method,
                score,
                color=METHOD_COLORS.get(method, "#333333"),
                s=100,
                zorder=5,
                edgecolor="white",
                linewidth=1,
            )

        ax.set_title(f"{metric}", fontweight="bold", fontsize=12)
        ax.set_ylabel("Score", fontsize=10)
        ax.tick_params(axis="x", rotation=45, labelsize=9)
        ax.grid(True, alpha=0.3)

        if not np.isnan(scores).all():
            y_min, y_max = np.nanmin(scores), np.nanmax(scores)
            y_range = y_max - y_min
            if y_range > 0:
                ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    plt.tight_layout()

    save_path = f"{save_dir}/{dataset_name}_individual_metrics.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {save_path}")

def plot_metrics_grouped(df, metrics_list, title, dataset_name, save_name, save_dir):
    if df is None or df.empty:
        return

    methods = [m for m in METHOD_COLORS.keys() if m in df["Method"].unique()]
    data_matrix = []
    valid_metrics = []

    for metric in metrics_list:
        metric_values = []
        has_data = False

        for method in methods:
            row = df[df["Method"] == method]
            if not row.empty and metric in row.columns:
                value = row[metric].values[0]
                if pd.notna(value):
                    metric_values.append(value)
                    has_data = True
                else:
                    metric_values.append(None)
            else:
                metric_values.append(None)

        if has_data:
            data_matrix.append(metric_values)
            valid_metrics.append(metric)

    if not data_matrix:
        print(f"No valid data found for {dataset_name}")
        return

    n_methods = len(methods)
    n_metrics = len(valid_metrics)

    fig, ax = plt.subplots(figsize=(max(12, n_metrics * 1.8), 8))

    x_pos = np.arange(n_metrics)

    for j, metric in enumerate(valid_metrics):
        metric_values = data_matrix[j]
        methods_with_data_for_metric = []
        scores = []

        for i, value in enumerate(metric_values):
            if value is not None:
                methods_with_data_for_metric.append(methods[i])
                scores.append(value)

        n_valid_methods = len(methods_with_data_for_metric)
        if n_valid_methods == 0:
            continue

        current_bar_width = 0.8 / max(n_valid_methods, 1)

        for i, (method, score) in enumerate(zip(methods_with_data_for_metric, scores)):
            position = x_pos[j] + (i - (n_valid_methods - 1) / 2) * current_bar_width
            color = METHOD_COLORS.get(method, "#999999")

            ax.bar([position], [score], current_bar_width, color=color, alpha=0.9)

    ax.set_xlabel("Metrics", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(valid_metrics, rotation=30, ha="right", fontsize=12)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color("black")

    ax.grid(False)
    legend_handles = []
    legend_labels = []
    used_methods = set()
    for row in data_matrix:
        for i, value in enumerate(row):
            if value is not None:
                used_methods.add(methods[i])

    for method in methods:
        if method in used_methods:
            color = METHOD_COLORS.get(method, "#999999")
            legend_handles.append(
                plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.9)
            )
            legend_labels.append(method)

    ax.legend(
        legend_handles,
        legend_labels,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=11,
        frameon=False,
    )

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin * 1.12 if ymin < 0 else 0, ymax * 1.12)
    ax.axhline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.7)

    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{dataset_name}_{save_name}.png"
    plt.savefig(
        save_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    plt.close()
    print(f"Saved: {save_path}")


def process_single_dataset_visualization(dataset_name):
    print(f"\n{'='*50}")
    print(f"Creating visualizations for: {dataset_name}")
    print(f"{'='*50}")

    save_dir = f"../results/figures/{dataset_name}"
    os.makedirs(save_dir, exist_ok=True)

    all_df, batch_df, bio_df, success = load_dataset_results(dataset_name)

    if not success:
        print(f"Skipping {dataset_name} due to missing data")
        return

    try:
        print("Creating individual metrics line plots...")
        plot_individual_metrics_lines(all_df, dataset_name, save_dir)

        print("Creating batch metrics grouped bar plot...")
        plot_metrics_grouped(
            batch_df,
            BATCH_METRICS,
            "Batch-related Metrics",
            dataset_name,
            "batch_metrics",
            save_dir,
        )

        print("Creating biological metrics grouped bar plot...")
        plot_metrics_grouped(
            bio_df,
            BIO_METRICS,
            "Biological Heterogeneity Metrics",
            dataset_name,
            "bio_metrics",
            save_dir,
        )

        print(f" Successfully created all plots for {dataset_name}")

    except Exception as e:
        print(f" Error creating plots for {dataset_name}: {e}")


def show_visualization_menu():
    print("\n" + "=" * 60)
    print("         SPATIAL BENCHMARK VISUALIZATION TOOL")
    print("=" * 60)
    print("\nAvailable options:")
    print("0. Exit")

    for i, dataset in enumerate(DATASETS, 1):
        print(f"{i}. Create plots for dataset: {dataset}")

    print(f"{len(DATASETS) + 1}. Create plots for ALL datasets")
    print(f"{len(DATASETS) + 2}. Show available datasets and files")
    print("=" * 60)


def check_available_files():
    print("\n" + "-" * 50)
    print("AVAILABLE CSV FILES:")
    print("-" * 50)

    base_path = "../results/individual_models"
    for dataset in DATASETS:
        print(f"\n{dataset}:")
        files = [
            f"{dataset}_all_metrics.csv",
            f"{dataset}_batch_metrics.csv",
            f"{dataset}_bio_metrics.csv",
        ]

        for file in files:
            file_path = f"{base_path}/{file}"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                print(f"{file} ({df.shape[0]} records)")
            else:
                print(f" {file} (missing)")


def process_all_visualizations():
    print(f"{'#'*60}")
    print("Creating visualizations for ALL datasets")
    print(f"{'#'*60}")

    success_count = 0
    for i, dataset in enumerate(DATASETS):
        print(f"Processing {i+1}/{len(DATASETS)}: {dataset}")
        try:
            process_single_dataset_visualization(dataset)
            success_count += 1
        except Exception as e:
            print(f" Error processing {dataset}: {e}")

    print(f" Successfully processed {success_count}/{len(DATASETS)} datasets")


import argparse
def main():
    parser = argparse.ArgumentParser(description="Spatial Benchmark Visualization Tool")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default=None,
        help="Dataset to visualize (e.g., dlpfc_sample1, hbc). "
        "Use 'all' for all datasets. "
        "Prefix like 'dlpfc' will match all starting datasets.",
    )
    args = parser.parse_args()
    if args.dataset is None:
        print("Please specify a dataset with -d. Available: "
              f"{', '.join(DATASETS)}")
        return

    elif args.dataset.lower() == "all":
        process_all_visualizations()

    elif args.dataset in DATASETS:
        process_single_dataset_visualization(args.dataset)

    else:
        matched = [ds for ds in DATASETS if ds.startswith(args.dataset)]
        if matched:
            print(f" Matched datasets: {', '.join(matched)}")
            for ds in matched:
                process_single_dataset_visualization(ds)
        else:
            print(
                f" Dataset {args.dataset} not recognized. Available: {', '.join(DATASETS)}"
            )


if __name__ == "__main__":
    main()
