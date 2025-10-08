import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from collections import defaultdict
import glob
from scipy.stats import friedmanchisquare
from adjustText import adjust_text
import sys

BATCH_METRICS = [
    "GC",
    "iLISI",
    "kBET",
    "ASW_batch",
]
BIO_METRICS = ["SCS", "MoranI", "GearyC", "cLISI", "ASW_domain", "ARI"]
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

METHOD_COLORS = {
    "RAW": "#1B77D8",
    "PRECAST": "#A1CEE8",
    "DeepST": "#00B5D8",
    "STAligner": "#38A169",
    "GraphST": "#ED8936",
    "SPIRAL": "#805AD5",
    "STitch3D": "#319795",
    "Spatialign": "#FF6B6B",
}

def rank_metrics(df, metrics_list):
    ranking_df = df.copy()

    for metric in metrics_list:
        if metric not in df.columns:
            print(f"Warning: {metric} not found in dataframe")
            continue

        values = df[metric].copy()

        if metric == "SCS":
            values_filled = values.fillna(values.max() + 1)
            ranking_df[f"{metric}_rank"] = values_filled.rank(ascending=True)

        elif metric == "MoranI":
            values_filled = values.fillna(values.min() - 1)
            ranking_df[f"{metric}_rank"] = values_filled.rank(ascending=False)

        elif metric == "GearyC":
            values_filled = values.fillna(values.max() + 1)
            ranking_df[f"{metric}_rank"] = values_filled.rank(ascending=True)
        else:
            values_filled = values.fillna(values.min() - 1)
            ranking_df[f"{metric}_rank"] = values_filled.rank(ascending=False)

    return ranking_df


def calculate_dataset_rankings(data_dict, save_dir="../results/rankings"):
    os.makedirs(save_dir, exist_ok=True)
    all_dataset_rankings = {}

    for dataset in DATASETS:
        if dataset not in data_dict:
            print(f"Warning: Dataset {dataset} not found in data")
            continue

        df = data_dict[dataset]
        print(f"process dataset: {dataset}")

        dataset_rankings = {}

        # Batch metrics ranking
        batch_ranking_df = rank_metrics(df, BATCH_METRICS)
        batch_rank_cols = [
            f"{metric}_rank"
            for metric in BATCH_METRICS
            if f"{metric}_rank" in batch_ranking_df.columns
        ]

        if batch_rank_cols:
            batch_ranking_df["batch_avg_rank"] = batch_ranking_df[batch_rank_cols].mean(
                axis=1
            )
            dataset_rankings["batch"] = batch_ranking_df[
                ["Method"] + batch_rank_cols + ["batch_avg_rank"]
            ].copy()

            dataset_rankings["batch"].to_csv(
                os.path.join(save_dir, f"{dataset}_batch_ranking.csv"), index=False
            )

        # Bio metrics ranking
        bio_ranking_df = rank_metrics(df, BIO_METRICS)
        bio_rank_cols = [
            f"{metric}_rank"
            for metric in BIO_METRICS
            if f"{metric}_rank" in bio_ranking_df.columns
        ]

        if bio_rank_cols:
            bio_ranking_df["bio_avg_rank"] = bio_ranking_df[bio_rank_cols].mean(axis=1)
            dataset_rankings["bio"] = bio_ranking_df[
                ["Method"] + bio_rank_cols + ["bio_avg_rank"]
            ].copy()

            dataset_rankings["bio"].to_csv(
                os.path.join(save_dir, f"{dataset}_bio_ranking.csv"), index=False
            )

        all_dataset_rankings[dataset] = dataset_rankings

    return all_dataset_rankings


def plot_dataset_rankings(all_dataset_rankings, save_dir="../results/rankings"):
    os.makedirs(save_dir, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_style("whitegrid")

    for dataset, dataset_data in all_dataset_rankings.items():
        print(f"dataset {dataset} ...")

        for rank_type in ["batch", "bio"]:
            if rank_type not in dataset_data:
                continue

            df = dataset_data[rank_type]
            avg_rank_col = f"{rank_type}_avg_rank"

            if avg_rank_col not in df.columns or "Method" not in df.columns:
                print(f"pass {dataset}-{rank_type}: Required columns are missing")
                continue

            plot_data = []
            for _, row in df.iterrows():
                method = row["Method"]
                for col in df.columns:
                    if col.endswith("_rank"):
                        val = row[col]
                        if not pd.isna(val):
                            plot_data.append({"Method": method, "ranking": val})

            if not plot_data:
                print(f"pass {dataset}-{rank_type}: No valid data")
                continue

            plot_df = pd.DataFrame(plot_data)
            method_order = (
                plot_df.groupby("Method")["ranking"].mean().sort_values().index.tolist()
            )

            fig, ax = plt.subplots(figsize=(6, max(5, len(method_order) * 0.5)))

            colors = [METHOD_COLORS.get(method, "#999999") for method in method_order]

            box_plot = ax.boxplot(
                [
                    plot_df[plot_df["Method"] == method]["ranking"].values
                    for method in method_order
                ],
                vert=False,
                labels=method_order,
                patch_artist=True,
                showmeans=True,
                meanline=True,
            )

            for patch, color in zip(box_plot["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_xlabel("Average Ranking (Lower is Better)", fontsize=8)
            ax.set_ylabel("Method", fontsize=8)
            ax.set_xlim(left=0.5)
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.2)
                spine.set_color("black")

            plt.tight_layout()
            save_file = os.path.join(save_dir, f"{dataset}_{rank_type}_ranking.png")
            plt.savefig(save_file, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"save: {save_file}")


def friedman_significance_test(rank_matrix, method_names, alpha=0.05):
    n_methods, n_datasets = rank_matrix.shape

    if n_methods < 3 or n_datasets < 3:
        return {
            "friedman_stat": None,
            "friedman_p": None,
            "significant": False,
            "n_methods": n_methods,
            "n_datasets": n_datasets,
            "method_names": method_names,
        }

    try:
        friedman_stat, friedman_p = friedmanchisquare(*rank_matrix)
        significant = friedman_p < alpha

        avg_ranks = np.mean(rank_matrix, axis=1)

        result = {
            "friedman_stat": friedman_stat,
            "friedman_p": friedman_p,
            "significant": significant,
            "n_methods": n_methods,
            "n_datasets": n_datasets,
            "method_names": method_names,
            "avg_ranks": dict(zip(method_names, avg_ranks)),
            "alpha": alpha,
        }

        if significant:
            result["message"] = (
                f"The ranking is statistically significant (p = {friedman_p:.6f} < {alpha})"
            )
        else:
            result["message"] = (
                f"The ranking is not statistically significant (p = {friedman_p:.6f} >= {alpha})"
            )

        return result

    except Exception as e:
        return {
            "friedman_stat": None,
            "friedman_p": None,
            "significant": False,
            "error": str(e),
        }


def calculate_method_rankings_with_validation(
    all_dataset_rankings, save_dir="../results/rankings"
):
    os.makedirs(save_dir, exist_ok=True)

    method_rankings = {
        "batch": defaultdict(list),
        "bio": defaultdict(list),
    }

    for dataset, dataset_data in all_dataset_rankings.items():
        for rank_type in ["batch", "bio"]:
            if rank_type in dataset_data:
                ranking_df = dataset_data[rank_type]
                avg_rank_col = f"{rank_type}_avg_rank"

                if (
                    avg_rank_col in ranking_df.columns
                    and "Method" in ranking_df.columns
                ):
                    for _, row in ranking_df.iterrows():
                        method = row["Method"]
                        avg_rank = row[avg_rank_col]
                        if not pd.isna(avg_rank):
                            method_rankings[rank_type][method].append(avg_rank)

    method_coverage = {}
    for rank_type in ["batch", "bio"]:
        method_coverage[rank_type] = {}
        for method, ranks in method_rankings[rank_type].items():
            method_coverage[rank_type][method] = len(ranks)

        max_datasets = (
            max(method_coverage[rank_type].values())
            if method_coverage[rank_type]
            else 0
        )
        for method, count in method_coverage[rank_type].items():
            print(f"[{rank_type.upper()}] {method}: {count}/{max_datasets} dataset number")

    weighted_rankings = {}
    friedman_results = {}

    for rank_type in ["batch", "bio"]:
        print(
            f"\n=== {rank_type.upper()}Type method ranking and statistical verification ==="
        )

        if not method_rankings[rank_type]:
            weighted_rankings[rank_type] = {
                "ranking_df": pd.DataFrame(),
                "friedman_test": None,
            }
            continue

        performance_data = []
        max_coverage = max(method_coverage[rank_type].values())
        max_possible_rank = len(method_rankings[rank_type])

        for method, ranks in method_rankings[rank_type].items():
            if not ranks:
                continue

            mean_rank = np.mean(ranks)
            median_rank = np.median(ranks)
            std_rank = np.std(ranks)
            cv_rank = std_rank / mean_rank if mean_rank != 0 else np.nan
            coverage = len(ranks)

            coverage_weight = coverage / max_coverage
            rank_score = (max_possible_rank - mean_rank + 1) / max_possible_rank
            # Comprehensive weighted score: 70% ranking + 30% coverage
            weighted_score = rank_score * 0.7 + coverage_weight * 0.3

            performance_data.append(
                {
                    "Method": method,
                    "mean_rank": mean_rank,
                    "median_rank": median_rank,
                    "std_rank": std_rank,
                    "cv_rank": cv_rank,
                    "coverage": coverage,
                    "coverage_ratio": coverage / max_coverage,
                    "rank_score": rank_score,
                    "weighted_score": weighted_score,
                    "rank_count": len(ranks),
                    "raw_ranks": ranks,
                }
            )

        ranking_df = pd.DataFrame(performance_data).sort_values(
            "weighted_score", ascending=False
        )
        weighted_rankings[rank_type] = {"ranking_df": ranking_df}
        save_path = os.path.join(save_dir, f"{rank_type}_method_rankings.csv")
        ranking_df.to_csv(save_path, index=False)
        print(f"{rank_type.upper()} rankings saved: {save_path}")
        print(f"Total number of methods: {len(method_rankings[rank_type])}")
        print(f"Number of datasets: {max_coverage}")

        friedman_result = None
        all_methods = list(method_rankings[rank_type].keys())
        full_methods = [
            m for m, c in method_coverage[rank_type].items() if c == max_coverage
        ]
        if len(full_methods) >= 3 and max_coverage >= 3:
            try:
                rank_matrix = np.array(
                    [method_rankings[rank_type][m] for m in full_methods]
                )

                friedman_result = friedman_significance_test(rank_matrix, full_methods)
                friedman_results[f"{rank_type}_methods"] = friedman_result

                print(f"\nFriedman test results (full coverage methods):")
                print(f"  Number of methods: {len(full_methods)}")
                print(f"  Number of data sets: {max_coverage}")
                print(f"  χ² statistic: {friedman_result['friedman_stat']:.4f}")
                print(f"  p-value: {friedman_result['friedman_p']:.6f}")
                print(f"  Conclusion: {friedman_result['message']}")
            except Exception as e:
                print(f"Friedman test error: {e}")

                friedman_result = {"significant": None, "message": "Test failed"}
        else:
            print(
                f"Does not meet the Friedman test conditions (methods = {len(full_methods)}, datasets ={max_coverage})"
            )
            friedman_result = {"significant": None, "message": "Insufficient data"}

        weighted_rankings[rank_type]["friedman_test"] = friedman_result

    friedman_df = pd.DataFrame(
        [
            {
                "rank_type": rt,
                "methods": ",".join(
                    friedman_results.get(f"{rt}_methods", {}).get("methods", [])
                )
                or None,
                "friedman_stat": friedman_results.get(f"{rt}_methods", {}).get(
                    "friedman_stat", None
                ),
                "friedman_p": friedman_results.get(f"{rt}_methods", {}).get(
                    "friedman_p", None
                ),
                "significant": friedman_results.get(f"{rt}_methods", {}).get(
                    "significant", None
                ),
                "message": friedman_results.get(f"{rt}_methods", {}).get(
                    "message", None
                ),
            }
            for rt in ["batch", "bio"]
        ]
    )
    friedman_save_path = os.path.join(save_dir, "all_method_friedman_results.csv")
    friedman_df.to_csv(friedman_save_path, index=False)
    print(f"Friedman test save: {friedman_save_path}")

    return weighted_rankings, friedman_results


def calculate_per_metric_rankings_with_validation(
    data_dict,
    batch_metrics,
    bio_metrics,
    rank_metrics_func,
    save_dir="../results/rankings",
):
    os.makedirs(save_dir, exist_ok=True)
    all_metrics = batch_metrics + bio_metrics
    metric_rankings = {}
    friedman_results = {}

    for metric in all_metrics:
        print(f"\nanalysis: {metric}")
        method_scores = defaultdict(list)

        for dataset, df in data_dict.items():
            if metric not in df.columns:
                print(f"  dataset {dataset} no {metric}")
                continue

            ranked_df = rank_metrics_func(df, [metric])
            rank_col = f"{metric}_rank"

            if rank_col not in ranked_df.columns:
                continue

            for _, row in ranked_df.iterrows():
                method = row["Method"]
                rank = row[rank_col]
                if not pd.isna(rank):
                    method_scores[method].append(rank)

        all_methods = set()
        for dataset, df in data_dict.items():
            if "Method" in df.columns:
                all_methods.update(df["Method"].unique())

        max_datasets = max((len(r) for r in method_scores.values()), default=0)
        total_methods = len(all_methods)
        metric_results = []
        for method in all_methods:
            scores = method_scores.get(method, [])
            coverage = len(scores)
            mean_rank = np.mean(scores) if scores else total_methods
            coverage_ratio = coverage / max_datasets if max_datasets > 0 else 0
            rank_score = (
                (total_methods - mean_rank + 1) / total_methods
                if total_methods > 0
                else 0
            )
            weighted_score = rank_score * 0.7 + coverage_ratio * 0.3

            metric_results.append(
                {
                    "Method": method,
                    "mean_rank": mean_rank,
                    "coverage": coverage,
                    "coverage_ratio": coverage_ratio,
                    "rank_score": rank_score,
                    "weighted_score": weighted_score,
                    "raw_ranks": scores,
                }
            )

        ranking_df = pd.DataFrame(metric_results).sort_values(
            "weighted_score", ascending=False
        )
        metric_rankings[metric] = ranking_df
        ranking_path_csv = os.path.join(save_dir, f"{metric}_ranking.csv")
        ranking_df.to_csv(ranking_path_csv, index=False)

        full_methods = [m for m, r in method_scores.items() if len(r) == max_datasets]
        friedman_result = None
        if len(full_methods) >= 3 and max_datasets >= 3:
            try:
                rank_matrix = np.array([method_scores[m] for m in full_methods])
                friedman_result = friedman_significance_test(rank_matrix, full_methods)
                print(
                    f"Friedman test: method={len(full_methods)}, datasets={max_datasets}"
                )
                print(
                    f"  χ²={friedman_result['friedman_stat']:.4f}, p={friedman_result['friedman_p']:.6f}, conclusion={friedman_result['message']}"
                )
            except Exception as e:
                print(f"Friedman erro: {e}")
                friedman_result = {"significant": None, "message": "fail"}
        else:
            friedman_result = {"significant": None, "message": "data shortage"}

        friedman_results[metric] = friedman_result

    friedman_df = pd.DataFrame(
        [
            {
                "metric": metric,
                "friedman_stat": res.get("friedman_stat") if res else None,
                "friedman_p": res.get("friedman_p") if res else None,
                "significant": res.get("significant") if res else None,
                "methods": ";".join(res.get("methods", [])) if res else None,
            }
            for metric, res in friedman_results.items()
        ]
    )
    friedman_path_csv = os.path.join(save_dir, "metrics_friedman_results.csv")
    friedman_df.to_csv(friedman_path_csv, index=False)
    best_methods = []
    for metric, ranking_df in metric_rankings.items():
        if not ranking_df.empty:
            top_row = ranking_df.iloc[0]
            best_methods.append({
                "metric": metric,
                "best_method": top_row["Method"],
                "weighted_score": top_row["weighted_score"],
                "mean_rank": top_row["mean_rank"],
                "coverage": f"{int(top_row['coverage'])}of{int(ranking_df['coverage'].max())}",
            })

    best_methods_df = pd.DataFrame(best_methods)
    best_methods_path_csv = os.path.join(save_dir, "best_methods_summary.csv")
    best_methods_df.to_csv(best_methods_path_csv, index=False)

    return metric_rankings, friedman_results, best_methods_df


def plot_cross_dataset_rankings(
    weighted_rankings, friedman_results=None, save_dir="../results/rankings"
):
    os.makedirs(save_dir, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    print("\n=== Draw a weighted ranking scatter graph across datasets ===")

    for rank_type in ["batch", "bio"]:
        df = weighted_rankings[rank_type]
        if isinstance(df, dict):
            df = df.get("ranking_df", pd.DataFrame())

        if df.empty:
            print(f"pass {rank_type}: na data")
            continue

        y_min, y_max = df["mean_rank"].min(), df["mean_rank"].max()
        y_scale = 1.1
        y_vals_scaled = (df["mean_rank"] - y_min) * y_scale + y_min

        x_min, x_max = df["weighted_score"].min(), df["weighted_score"].max()
        x_scale = 1.1
        x_vals_scaled = (df["weighted_score"] - x_min) * x_scale + x_min

        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        sc = ax.scatter(
            x_vals_scaled,
            y_vals_scaled,
            s=df["coverage_ratio"] * 200,
            c=df["cv_rank"],
            cmap="RdYlBu_r",
            alpha=0.8,
            edgecolors="black",
        )

        x_margin = 0.05 * (x_vals_scaled.max() - x_vals_scaled.min())
        y_margin = 0.05 * (y_vals_scaled.max() - y_vals_scaled.min())
        ax.set_xlim(x_vals_scaled.min() - x_margin, x_vals_scaled.max() + x_margin)
        ax.set_ylim(y_vals_scaled.max() + y_margin, y_vals_scaled.min() - y_margin)

        texts = []
        for i, row in df.iterrows():
            texts.append(
                ax.text(
                    x_vals_scaled[i],
                    y_vals_scaled[i],
                    row["Method"],
                    fontsize=8,
                    ha="center",
                    va="center",
                )
            )
        adjust_text(
            texts,
            x=x_vals_scaled,
            y=y_vals_scaled,
            expand_points=(2, 2),
            expand_text=(2, 2),
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
        )

        ax.set_xlabel("Weighted Score (Higher is Better)")
        ax.set_ylabel("Mean Rank (Lower is Better)")
        ax.grid(True, linestyle="--", alpha=0.5)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Coefficient of Variation (CV)")

        if friedman_results:
            test_key = f"{rank_type}_methods"
            if test_key in friedman_results and friedman_results[test_key]:
                p_val = friedman_results[test_key]["friedman_p"]
                sig = "*" if p_val < 0.05 else "ns"
                ax.set_title(f"Friedman p={p_val:.3f}, {sig}")

        plt.tight_layout()
        save_file = os.path.join(
            save_dir, f"cross_dataset_{rank_type}_scatter_rankings.png"
        )
        plt.savefig(save_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"save: {save_file}")


def plot_metric_rankings(
    metric_rankings,
    friedman_results=None,
    save_dir="../results/rankings",
    plot_type="bar",
):
    os.makedirs(save_dir, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    for metric_type, metrics_list in [("batch", BATCH_METRICS), ("bio", BIO_METRICS)]:
        print(f"plot {metric_type} metric {plot_type}...")

        methods = set()
        data = {metric: {} for metric in metrics_list}

        for metric in metrics_list:
            if metric not in metric_rankings:
                continue
            df = metric_rankings[metric]
            for _, row in df.iterrows():
                method = row["Method"]
                score = row["weighted_score"]
                data[metric][method] = score
                methods.add(method)

        if not methods:
            print(f"pass {metric_type}: no ")
            continue

        methods = sorted(list(methods))
        metrics = metrics_list

        if plot_type == "bar":
            n_metrics = len(metrics)
            n_cols = 3
            n_rows = (n_metrics + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)

            for i, metric in enumerate(metrics):
                row, col = divmod(i, n_cols)
                ax = axes[row, col]

                if metric not in metric_rankings:
                    ax.set_visible(False)
                    continue

                df = metric_rankings[metric]
                methods_metric = df["Method"].tolist()
                scores = df["weighted_score"].tolist()
                colors = [METHOD_COLORS.get(m, "#999999") for m in methods_metric]

                bars = ax.barh(
                    range(len(methods_metric)), scores, color=colors, alpha=0.8
                )
                ax.set_yticks(range(len(methods_metric)))
                ax.set_yticklabels(methods_metric)
                ax.set_xlabel("Weighted Score")
                ax.set_xlim(0, 1)
                ax.grid(False)
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(1.2)
                    spine.set_color("black")

                title = metric
                if (
                    friedman_results
                    and metric in friedman_results
                    and friedman_results[metric]
                ):
                    p_val = friedman_results[metric]["friedman_p"]
                    title += f" (p={p_val:.3f})"

                ax.set_title(title, fontweight="bold")
                ax.invert_yaxis()

            for i in range(len(metrics), n_rows * n_cols):
                row, col = divmod(i, n_cols)
                axes[row, col].set_visible(False)

            plt.tight_layout()
            save_file = os.path.join(save_dir, f"{metric_type}_metrics_bar.png")
            plt.savefig(save_file, dpi=300, bbox_inches="tight")
            plt.close(fig)

        elif plot_type == "radar":
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]

            fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
            for method in methods:
                values = [data[m].get(method, 0) for m in metrics]
                values += values[:1]
                ax.plot(
                    angles,
                    values,
                    marker="o",
                    markersize=5,
                    label=method,
                    color=METHOD_COLORS.get(method, None),
                )
                ax.fill(
                    angles, values, alpha=0.2, color=METHOD_COLORS.get(method, None)
                )
            ax.set_xticks(angles[:-1])

            if friedman_results:
                radar_labels = []
                for metric in metrics:
                    if metric in friedman_results and friedman_results[metric]:
                        p_val = friedman_results[metric]["friedman_p"]
                        if p_val < 0.001:
                            radar_labels.append(f"{metric} ***")
                        elif p_val < 0.01:
                            radar_labels.append(f"{metric} **")
                        elif p_val < 0.05:
                            radar_labels.append(f"{metric} *")
                        else:
                            radar_labels.append(metric)
                    else:
                        radar_labels.append(metric)
            else:
                radar_labels = metrics

            ax.set_xticklabels(radar_labels, fontsize=14)
            for i, label in enumerate(ax.get_xticklabels()):
                angle = angles[i]
                x, y = label.get_position()
                if 0 <= angle < np.pi:
                    label.set_position((x, y + 0.1))
                else:
                    label.set_position((x, y - 0.1))
            ax.set_ylim(0, 1)
            ax.legend(bbox_to_anchor=(1.1, 1.05))

            plt.tight_layout()
            save_file = os.path.join(save_dir, f"{metric_type}_metrics_radar.png")
            plt.savefig(save_file, dpi=300, bbox_inches="tight")
            plt.close(fig)

        print(f"save: {save_file}")


def calculate_unified_method_rankings_with_validation(
    all_dataset_rankings, save_dir="../results/rankings"
):
    os.makedirs(save_dir, exist_ok=True)

    method_combined_ranks = defaultdict(list)
    total_datasets = len(all_dataset_rankings)

    print(f"\n=== Calculate the ranking of unified methods ===")
    print(f"total datasets: {total_datasets}")

    for dataset, dataset_data in all_dataset_rankings.items():
        batch_df = dataset_data.get("batch", pd.DataFrame())
        bio_df = dataset_data.get("bio", pd.DataFrame())

        if batch_df.empty or bio_df.empty:
            continue

        if (
            "batch_avg_rank" not in batch_df.columns
            or "bio_avg_rank" not in bio_df.columns
            or "Method" not in batch_df.columns
            or "Method" not in bio_df.columns
        ):
            continue

        batch_ranks = batch_df.set_index("Method")["batch_avg_rank"]
        bio_ranks = bio_df.set_index("Method")["bio_avg_rank"]

        common_methods = batch_ranks.index.intersection(bio_ranks.index)

        for method in common_methods:
            if pd.isna(batch_ranks[method]) or pd.isna(bio_ranks[method]):
                continue
            combined_rank = batch_ranks[method] * 0.6 + bio_ranks[method] * 0.4
            method_combined_ranks[method].append(combined_rank)

    if not method_combined_ranks:
        return pd.DataFrame(), None

    max_datasets = max(len(r) for r in method_combined_ranks.values())
    max_possible_rank = len(method_combined_ranks)

    performance_data = []
    for method, ranks in method_combined_ranks.items():
        mean_rank = np.mean(ranks)
        std_rank = np.std(ranks)
        cv_rank = std_rank / mean_rank if mean_rank != 0 else np.nan
        coverage = len(ranks)

        coverage_weight = coverage / max_datasets
        rank_score = (max_possible_rank - mean_rank + 1) / max_possible_rank
        weighted_score = rank_score * 0.7 + coverage_weight * 0.3

        performance_data.append(
            {
                "Method": method,
                "mean_combined_rank": mean_rank,
                "cv_combined_rank": cv_rank,
                "coverage": coverage,
                "coverage_ratio": coverage / max_datasets,
                "weighted_score": weighted_score,
                "raw_combined_ranks": ranks,
            }
        )

    ranking_df = pd.DataFrame(performance_data).sort_values(
        "weighted_score", ascending=False
    )

    friedman_result = None
    full_methods = [
        m for m, r in method_combined_ranks.items() if len(r) == max_datasets
    ]

    if len(full_methods) >= 3 and max_datasets >= 3:
        try:
            rank_matrix = np.array([method_combined_ranks[m] for m in full_methods])
            friedman_result = friedman_significance_test(rank_matrix, full_methods)

            print(f"\nFriedman test results (full coverage methods):")
            print(f"  Number of methods: {len(full_methods)}")
            print(f"  Number of datasets: {max_datasets}")
            print(f"  χ² statistic: {friedman_result['friedman_stat']:.4f}")
            print(f"  p-value: {friedman_result['friedman_p']:.6f}")
            print(f"  Conclusion: {friedman_result['message']}")
        except Exception as e:
            print(f"Friedman test error: {e}")
            friedman_result = {"significant": None, "message": "Test failed"}
    else:
        print(
            f"\nFriedman test not applicable (methods={len(full_methods)}, datasets={max_datasets})"
        )
        friedman_result = {"significant": None, "message": "Insufficient data"}

    ranking_df.to_csv(
        os.path.join(save_dir, "unified_method_rankings.csv"), index=False
    )
    if friedman_result:
        pd.DataFrame(
            [
                {
                    "rank_type": "unified",
                    "methods": ",".join(friedman_result.get("methods", [])),
                    "friedman_stat": friedman_result.get("friedman_stat", None),
                    "friedman_p": friedman_result.get("friedman_p", None),
                    "significant": friedman_result.get("significant", None),
                    "message": friedman_result.get("message", None),
                }
            ]
        ).to_csv(os.path.join(save_dir, "unified_friedman_results.csv"), index=False)

    return ranking_df, friedman_result


def plot_unified_rankings(
    ranking_df, friedman_result=None, save_dir="../results/rankings"
):
    os.makedirs(save_dir, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    methods = ranking_df["Method"].tolist()
    weighted_scores = ranking_df["weighted_score"].tolist()
    mean_ranks = ranking_df["mean_combined_rank"].tolist()
    coverage_ratios = ranking_df["coverage_ratio"].tolist()
    cv_ranks = ranking_df["cv_combined_rank"].tolist()

    sizes = [c * 300 for c in coverage_ratios]
    colors = cv_ranks

    fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(
        weighted_scores,
        mean_ranks,
        s=sizes,
        c=colors,
        cmap="RdYlBu_r",
        alpha=0.8,
        edgecolors="k",
    )

    ax.invert_yaxis()

    for i, method in enumerate(methods):
        ax.text(
            weighted_scores[i] + 0.005,
            mean_ranks[i] + 0.05,
            method,
            va="center",
            fontsize=9,
        )

    ax.set_xlabel("Weighted Score")
    ax.set_ylabel("Mean Combined Rank")

    xmin, xmax = min(weighted_scores), max(weighted_scores)
    x_margin = max(0.1 * (xmax - xmin), 0.02)
    ax.set_xlim(xmin - x_margin, xmax + x_margin)

    ymin, ymax = min(mean_ranks), max(mean_ranks)
    y_margin = max(0.1 * (ymax - ymin), 0.2)
    ax.set_ylim(ymax + y_margin, ymin - y_margin)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("CV of Combined Rank")

    ax.text(
        0.98,
        0.02,
        "Point size: Coverage%\nPoint color: CV of Combined Rank\nCombined Rank = Batch×0.6 + Bio×0.4\nWeight: Rank 70% + Coverage 30%",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        alpha=0.7,
        style="italic",
    )
    if friedman_result:
        p_val = friedman_result.get("friedman_p")
        if p_val is not None:
            sig = "*" if p_val < 0.05 else "ns"
            ax.set_title(f"Friedman p={p_val:.3f}, {sig}")

    plt.tight_layout()
    save_file = os.path.join(save_dir, "unified_method_bubble_rankings.png")
    plt.savefig(save_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return save_file


def main_ranking_analysis(data_dict, save_dir="../results/rankings"):
    os.makedirs(save_dir, exist_ok=True)
    log_file_path = os.path.join(save_dir, "ranking_summary.txt")
    
    original_stdout = sys.stdout
    with open(log_file_path, "w") as f:
        sys.stdout = f 
        
        print("\n=== Step 1: Calculate method rankings for each dataset ===")
        all_dataset_rankings = calculate_dataset_rankings(data_dict)

        print("\n=== Step 2: Plot rankings for each dataset ===")
        plot_dataset_rankings(all_dataset_rankings, save_dir)

        print("\n=== Step 3: Calculate cross-dataset weighted rankings ===")
        weighted_rankings, method_friedman_results = (
            calculate_method_rankings_with_validation(all_dataset_rankings)
        )
        print("\n=== Step 4: Plot cross-dataset weighted rankings ===")
        plot_cross_dataset_rankings(weighted_rankings, method_friedman_results, save_dir)

        print("\n=== Step 5: Calculate method rankings for each metric ===")
        metric_rankings, metric_friedman_results, best_methods_df = (
            calculate_per_metric_rankings_with_validation(
                data_dict, BATCH_METRICS, BIO_METRICS, rank_metrics
            )
        )

        print("\n=== Step 6: Plot metric rankings (radar charts) ===")
        plot_metric_rankings(
            metric_rankings, metric_friedman_results, save_dir, plot_type="radar"
        )
        plot_metric_rankings(
            metric_rankings, metric_friedman_results, save_dir, plot_type="bar"
        )
        print("\n=== Step 7: Calculate method rankings for unified===")
        ranking_df, friedman_result = calculate_unified_method_rankings_with_validation(
            all_dataset_rankings, save_dir=save_dir
        )

        print("\n=== Step 8: Plot unified rankings ===")
        plot_unified_rankings(ranking_df, friedman_result, save_dir)
        sys.stdout = original_stdout
    
    print("All done ")
    return (
        all_dataset_rankings,
        weighted_rankings,
        method_friedman_results,
        metric_rankings,
        metric_friedman_results,
        ranking_df,
        friedman_result,
        best_methods_df,
    )

if __name__ == "__main__":
    results_dir = "../results/individual_models"
    data_dict = {}
    for file in os.listdir(results_dir):
        if file.endswith("_all_metrics.csv"):
            dataset_name = file.replace("_all_metrics.csv", "")
            file_path = os.path.join(results_dir, file)
            df = pd.read_csv(file_path)
            data_dict[dataset_name] = df
    (
        all_dataset_rankings,
        weighted_rankings,
        metric_rankings,
        method_friedman_results,
        metric_friedman_results,
        ranking_df,
        friedman_result,
        best_methods_df,
    ) = main_ranking_analysis(data_dict, save_dir="../results/rankings")
    print("All done ")
