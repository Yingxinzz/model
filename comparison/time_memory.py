import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")


class ScalabilityAnalyzer:
    def __init__(self):
        self.data = None
        self.method_colors = {
            "PRECAST": "#3182CE",
            "DeepST": "#00B5D8",
            "STAligner": "#38A169",
            "GraphST": "#ED8936",
            "SPIRAL": "#805AD5",
            "STitch3D": "#319795",
            "Spatialign": "#FF6B6B",
        }

        self.dataset_mapping = {
            "benchmark1": "DLPFC-1",
            "benchmark2": "DLPFC-2",
            "benchmark3": "DLPFC-3",
            "benchmark_7374": "DLPFC-7374",
            "benchmark_all": "DLPFC-All",
            "benchmark_hbc": "HBC",
            "benchmark_coronal": "Coronal",
            "benchmark_mob": "MOB",
        }

    def load_benchmark_results(self, base_path="../"):
        benchmark_files = {
            "DeepST": [
                f"{base_path}DeepST/Results/deepst_benchmark1.json",
                f"{base_path}DeepST/Results/deepst_benchmark2.json",
                f"{base_path}DeepST/Results/deepst_benchmark3.json",
                f"{base_path}DeepST/Results/deepst_benchmark_7374.json",
                f"{base_path}DeepST/Results/deepst_benchmark_all.json",
                f"{base_path}DeepST/Results/deepst_benchmark_hbc.json",
                f"{base_path}DeepST/Results/deepst_benchmark_coronal.json",
            ],
            "SPIRAL": [
                f"{base_path}SPIRAL/results/spiral_benchmark1.json",
                f"{base_path}SPIRAL/results/spiral_benchmark2.json",
                f"{base_path}SPIRAL/results/spiral_benchmark3.json",
                f"{base_path}SPIRAL/results/spiral_benchmark_7374.json",
                f"{base_path}SPIRAL/results/spiral_benchmark_all.json",
                f"{base_path}SPIRAL/results/spiral_benchmark_hbc.json",
                f"{base_path}SPIRAL/results/spiral_benchmark_coronal.json",
                f"{base_path}SPIRAL/results/spiral_benchmark_mob.json",
            ],
            "PRECAST": [
                f"{base_path}PRECAST/dlpfc/results/precast_benchmark1.json",
                f"{base_path}PRECAST/dlpfc/results/precast_benchmark2.json",
                f"{base_path}PRECAST/dlpfc/results/precast_benchmark3.json",
                f"{base_path}PRECAST/dlpfc/results/precast_benchmark_7374.json",
                f"{base_path}PRECAST/dlpfc/results/precast_benchmark_all.json",
                f"{base_path}PRECAST/hbc/results/precast_benchmark_hbc.json",
                f"{base_path}PRECAST/coronal/results/precast_benchmark_coronal.json",
                f"{base_path}PRECAST/mob/results/precast_benchmark_mob.json",
            ],
            "STitch3D": [
                f"{base_path}STitch3D/results_dlpfc1/stitch3d_benchmark1.json",
                f"{base_path}STitch3D/results_dlpfc2/stitch3d_benchmark2.json",
                f"{base_path}STitch3D/results_dlpfc3/stitch3d_benchmark3.json",
                f"{base_path}STitch3D/results_dlpfc_7374/stitch3d_benchmark_7374.json",
                f"{base_path}STitch3D/results_dlpfc_all/stitch3d_benchmark_all.json",
            ],
            "GraphST": [
                f"{base_path}GraphST/results/graphst_benchmark1.json",
                f"{base_path}GraphST/results/graphst_benchmark2.json",
                f"{base_path}GraphST/results/graphst_benchmark3.json",
                f"{base_path}GraphST/results/graphst_benchmark_7374.json",
                f"{base_path}GraphST/results/graphst_benchmark_all.json",
                f"{base_path}GraphST/results/graphst_benchmark_hbc.json",
                f"{base_path}GraphST/results/graphst_benchmark_coronal.json",
                f"{base_path}GraphST/results/graphst_benchmark_mob.json",
            ],
            "spatiAlign": [
                f"{base_path}Spatialign/results_dlpfc1/spatialign_benchmark1.json",
                f"{base_path}Spatialign/results_dlpfc2/spatialign_benchmark2.json",
                f"{base_path}Spatialign/results_dlpfc3/spatialign_benchmark3.json",
                f"{base_path}Spatialign/results_dlpfc1/spatialign_benchmark_7374.json",
                f"{base_path}Spatialign/results_dlpfc_all/spatialign_benchmark_all.json",
                f"{base_path}Spatialign/results_hbc/spatialign_benchmark_hbc.json",
                f"{base_path}Spatialign/results_coronal/spatialign_benchmark_coronal.json",
                f"{base_path}Spatialign/results_mob/spatialign_benchmark_mob.json",
            ],
            "STAligner": [
                f"{base_path}STAligner/results/staligner_benchmark1.json",
                f"{base_path}STAligner/results/staligner_benchmark2.json",
                f"{base_path}STAligner/results/staligner_benchmark3.json",
                f"{base_path}STAligner/results/staligner_benchmark_all.json",
                f"{base_path}STAligner/results/staligner_benchmark_7374.json",
                f"{base_path}STAligner/results/staligner_benchmark_hbc.json",
                f"{base_path}STAligner/results/staligner_benchmark_coronal.json",
                f"{base_path}STAligner/results/staligner_benchmark_mob.json",
            ],
        }

        results = []

        for method_name, file_paths in benchmark_files.items():
            for file_path in file_paths:
                if os.path.exists(file_path):
                    try:
                        with open(file_path, "r") as f:
                            data = json.load(f)
                            if "method_name" not in data:
                                data["method_name"] = method_name

                            filename = Path(file_path).stem
                            for key in self.dataset_mapping.keys():
                                if key in filename:
                                    data["dataset"] = key
                                    data["dataset_name"] = self.dataset_mapping[key]
                                    break

                            results.append(data)
                    except Exception as e:
                        print(f" Error loading  {method_name}: {e}")
                else:
                    print(f" no: {file_path}")

        if not results:
            return pd.DataFrame()

        self.data = pd.DataFrame(results)
        return self.data

    def plot_time_comparison(
        self, time_unit="minutes", figsize=(6, 5), output_dir="../results/time_memory/"
    ):
        time_col = f"training_time_{time_unit}"
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "time_comparison.png")
        plt.figure(figsize=figsize)

        pivot_data = self.data.pivot_table(
            index="dataset_name",
            columns="method_name",
            values=time_col,
            aggfunc="mean",
        )

        ax = pivot_data.plot(
            kind="bar",
            color=[
                self.method_colors.get(col, "#999999") for col in pivot_data.columns
            ],
            figsize=figsize,
        )

        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)
            spine.set_color("black")

        ax.axhline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.7)
        plt.xlabel(" Datasets", fontsize=12)
        plt.ylabel(f" Run time ({time_unit})", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    def plot_memory_comparison(
        self, memory_unit="mb", figsize=(6, 5), output_dir="../results/time_memory/"
    ):
        memory_col = f"memory_usage_{memory_unit}"
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "memory_comparison.png")
        plt.figure(figsize=figsize)

        pivot_data = self.data.pivot_table(
            index="dataset_name",
            columns="method_name",
            values=memory_col,
            aggfunc="mean",
        )

        ax = pivot_data.plot(
            kind="bar",
            color=[
                self.method_colors.get(col, "#999999") for col in pivot_data.columns
            ],
            figsize=figsize,
        )
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2)
            spine.set_color("black")

        ax.axhline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.7)
        plt.xlabel("datasets", fontsize=12)
        plt.ylabel(f"memory usage ({memory_unit.upper()})", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    def plot_scalability_analysis(
        self, figsize=(12, 5), output_dir="../results/time_memory/"
    ):
        os.makedirs(output_dir, exist_ok=True)

        datasets = self.data["dataset_name"].unique()

        for dataset in datasets:
            dataset_data = self.data[self.data["dataset_name"] == dataset]
            if dataset_data.empty:
                continue

            fig, axes = plt.subplots(1, 2, figsize=figsize)
            methods = dataset_data["method_name"].unique()

            ax1 = axes[0]
            for method in methods:
                method_data = dataset_data[dataset_data["method_name"] == method]
                if not method_data.empty:
                    ax1.scatter(
                        method_data["training_time_minutes"],
                        method_data["memory_usage_mb"],
                        label=method,
                        color=self.method_colors.get(method, "#999999"),
                        alpha=0.7,
                        s=80,
                        edgecolors="black",
                        linewidth=0.5,
                    )

            ax1.set_xlabel("Training Time (min)", fontsize=12)
            ax1.set_ylabel("Memory Usage (MB)", fontsize=12)
            ax1.set_title("Time vs Memory Usage", fontsize=14, fontweight="bold")
            ax1.grid(False)
            for spine in ax1.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.2)
                spine.set_color("black")

            ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

            ranking = []
            for method in methods:
                method_data = dataset_data[dataset_data["method_name"] == method]
                if not method_data.empty:
                    avg_time = method_data["training_time_minutes"].mean()
                    avg_memory = method_data["memory_usage_mb"].mean()
                    ranking.append((method, avg_time, avg_memory))

            ranking_df = pd.DataFrame(
                ranking, columns=["Method", "Avg_Time", "Avg_Memory"]
            )
            ranking_df["Time_Rank"] = ranking_df["Avg_Time"].rank(method="min")
            ranking_df["Memory_Rank"] = ranking_df["Avg_Memory"].rank(method="min")
            ranking_df["Score"] = (
                0.9 * ranking_df["Time_Rank"] + 0.1 * ranking_df["Memory_Rank"]
            )
            ranking_df = ranking_df.sort_values("Score", ascending=True).reset_index(
                drop=True
            )

            ax2 = axes[1]
            y_pos = range(len(ranking_df))
            colors = [
                self.method_colors.get(method, "#999999")
                for method in ranking_df["Method"]
            ]

            bars = ax2.barh(
                y_pos,
                ranking_df["Score"],
                color=colors,
                alpha=0.7,
                edgecolor="black",
                linewidth=0.5,
            )

            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(ranking_df["Method"])
            ax2.set_xlabel("Composite Score (Lower is Better)", fontsize=12)
            ax2.set_title(
                "Composite Ranking (90% Time + 10% Memory)",
                fontsize=14,
                fontweight="bold",
            )
            ax2.grid(False)
            for spine in ax2.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.2)
                spine.set_color("black")

            ax2.invert_yaxis()

            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f"{dataset}_scalability.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    def plot_performance_ranking(self, output_dir="../results/time_memory/"):
        os.makedirs(output_dir, exist_ok=True)

        method_coverage = self.data.groupby("method_name")["dataset_name"].nunique()
        max_datasets = method_coverage.max()

        rank_records = []
        for dataset, group in self.data.groupby("dataset_name"):
            time_rank = (
                group.groupby("method_name")["training_time_minutes"]
                .mean()
                .rank(method="min")
            )
            memory_rank = (
                group.groupby("method_name")["memory_usage_mb"]
                .mean()
                .rank(method="min")
            )

            for method in group["method_name"].unique():
                rank_records.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "time_rank": time_rank.get(method, np.nan),
                        "memory_rank": memory_rank.get(method, np.nan),
                    }
                )

        rank_df = pd.DataFrame(rank_records)

        method_summary = []
        for method in rank_df["method"].unique():
            method_data = rank_df[rank_df["method"] == method]

            # ranks
            time_ranks = method_data["time_rank"].dropna().values
            memory_ranks = method_data["memory_rank"].dropna().values

            if len(time_ranks) == 0 or len(memory_ranks) == 0:
                continue

            mean_time_rank = np.mean(time_ranks)
            mean_memory_rank = np.mean(memory_ranks)

            coverage = method_coverage[method]
            coverage_ratio = coverage / max_datasets

            # stability
            stability = self.data[self.data["method_name"] == method].agg(
                {
                    "training_time_minutes": ["mean", "std"],
                    "memory_usage_mb": ["mean", "std"],
                }
            )
            time_cv = (
                stability.loc["std", "training_time_minutes"]
                / stability.loc["mean", "training_time_minutes"]
                if stability.loc["mean", "training_time_minutes"] > 0
                else np.nan
            )
            memory_cv = (
                stability.loc["std", "memory_usage_mb"]
                / stability.loc["mean", "memory_usage_mb"]
                if stability.loc["mean", "memory_usage_mb"] > 0
                else np.nan
            )
            avg_cv = np.nanmean([time_cv, memory_cv])

            max_possible_rank = len(method_coverage)
            time_score = (max_possible_rank - mean_time_rank + 1) / max_possible_rank
            memory_score = (
                max_possible_rank - mean_memory_rank + 1
            ) / max_possible_rank

            # weighted score
            weighted_score_time = time_score * 0.7 + coverage_ratio * 0.3
            weighted_score_memory = memory_score * 0.7 + coverage_ratio * 0.3
            overall_score = weighted_score_time * 0.9 + weighted_score_memory * 0.1

            method_summary.append(
                {
                    "method": method,
                    "mean_time_rank": mean_time_rank,
                    "mean_memory_rank": mean_memory_rank,
                    "coverage": coverage,
                    "coverage_ratio": coverage_ratio,
                    "time_cv": time_cv,
                    "memory_cv": memory_cv,
                    "avg_cv": avg_cv,
                    "overall_weighted_score": overall_score,
                }
            )

        df = pd.DataFrame(method_summary).sort_values(
            "overall_weighted_score", ascending=False
        )

        plt.figure(figsize=(10, 6))
        y_pos = np.arange(len(df))
        sc = plt.scatter(
            df["overall_weighted_score"],
            y_pos,
            c=df["coverage_ratio"],
            s=(1 / df["avg_cv"].replace(0, np.nan)) * 100,
            cmap="viridis",
            edgecolor="black",
            zorder=3,
        )
        plt.hlines(
            y_pos,
            xmin=0,
            xmax=df["overall_weighted_score"],
            zorder=2,
            color="lightgray",
        )
        plt.yticks(y_pos, df["method"])
        plt.xlabel("Overall Weighted Score (Higher is Better)", fontsize=14)
        plt.ylabel("Method", fontsize=16)
        cbar = plt.colorbar(sc)
        cbar.set_label("Coverage Ratio")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "performance_lollipop.png"), dpi=300)
        plt.close()

        heatmap_df = df.set_index("method")[
            ["mean_time_rank", "mean_memory_rank", "coverage_ratio", "avg_cv"]
        ]
        plt.figure(figsize=(10, max(6, 0.5 * len(df))))
        sns.heatmap(heatmap_df, annot=True, cmap="RdYlGn_r", fmt=".2f", cbar=True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "performance_heatmap.png"), dpi=300)
        plt.close()

        csv_path = os.path.join(output_dir, "performance_summary.csv")
        df.to_csv(csv_path, index=False)
        print(f"Performance summary saved to {csv_path}")
        print(f"Plots saved to {output_dir}")

        return df


def main():
    analyzer = ScalabilityAnalyzer()
    output_dir = "../results/time_memory/"
    os.makedirs(output_dir, exist_ok=True)
    analyzer.load_benchmark_results(base_path="../")
    print("1. RUN Time...")
    analyzer.plot_time_comparison(time_unit="minutes", output_dir=output_dir)

    print("2. Memory usage...")
    analyzer.plot_memory_comparison(memory_unit="mb", output_dir=output_dir)

    print("3. comprehensive scalability...")
    analyzer.plot_scalability_analysis(output_dir=output_dir)

    print("4. performance rank...")
    analyzer.plot_performance_ranking(output_dir=output_dir)


if __name__ == "__main__":
    main()
