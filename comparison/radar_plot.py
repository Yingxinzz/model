##一个雷达图（加权平均）
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kruskal
import numpy as np
file_paths = [
    '../results/1dlpfc_model_performance_results.csv', 
    '../results/2dlpfc_model_performance_results.csv', 
    '../results/3dlpfc_model_performance_results.csv',
    '../results/dlpfc_7374_model_performance_results.csv',
    '../results/all_dlpfc_model_performance_results.csv',
    '../results/coronal_model_performance_results.csv',
    '../results/mob_metrics_results.csv',
    '../results/hbc_model_performance_results.csv',
]
Dataset = ['DLPFC1', 'DLPFC2', 'DLPFC3', 'Consecutive', 'All_DLPFC', 'CMB', 'MOB', 'HBC']
dfs = []
for file, label in zip(file_paths, Dataset):
    df = pd.read_csv(file)
    df['Dataset'] = label
    dfs.append(df)

df = pd.concat(dfs, axis=0, ignore_index=True)
df.to_csv('../results/scib_all_samples.csv', index=False)
df = pd.read_csv('../results/scib_all_samples.csv')
<<<<<<< HEAD
#########计算每个 metric 维度上的加权均值
=======
>>>>>>> 4a94f0cfaeb0738880cc409ba8f469928849db88
metrics = ['graph_connectivity', 'iLISI', 'kBET', 'ASW']
weights = {
    'RAW': 8, 'DeepST': 7, 'PRECAST': 7, 'STitch3D': 5, 'STAligner': 8, 'GraphST': 8,
    'Spatialign': 8, 'SPIRAL': 8
}
<<<<<<< HEAD
results = {}
mean_values = {}
for metric in metrics:
    grouped_data = [df[df['Model'] == model][metric].values for model in df['Model'].unique()]
    stat, p_value = kruskal(*grouped_data)
    results[metric] = p_value
=======


# Kruskal-Wallis test and weighted means
results = {}
mean_values = {}
for metric in metrics:
    # Grouping by model and performing Kruskal-Wallis test
    grouped_data = [df[df['Model'] == model][metric].values for model in df['Model'].unique()]
    stat, p_value = kruskal(*grouped_data)
    results[metric] = p_value
    # Weighted mean for each model
>>>>>>> 4a94f0cfaeb0738880cc409ba8f469928849db88
    model_means = {}
    for model in df['Model'].unique():
        model_data = df[df['Model'] == model]
        weight = weights.get(model, 1)
        weighted_mean = (model_data[metric] * weight).sum() / (weight * len(model_data))
        model_means[model] = weighted_mean
    mean_values[metric] = model_means

models = df['Model'].unique()
num_models = len(models)
angles = np.linspace(0, 2 * np.pi, num_models, endpoint=False).tolist()
<<<<<<< HEAD
angles += angles[:1]
# 保存 Kruskal-Wallis 检验结果
results_df = pd.DataFrame(list(results.items()), columns=['Metric', 'p-value'])
results_df.to_csv('../results/kruskal_wallis_results.csv', index=False)
=======
angles += angles[:1]  # Close the plot
>>>>>>> 4a94f0cfaeb0738880cc409ba8f469928849db88

fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': 'polar'})
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
<<<<<<< HEAD
ax.set_xticks(angles[:-1])  
ax.set_xticklabels([])
for i, label in enumerate(models):
    angle = angles[i]
    ax.text(
        angle,
=======
ax.set_xticks(angles[:-1])  # Don't include the last angle for the label
ax.set_xticklabels([])  # Remove default xtick labels
for i, label in enumerate(models):
    angle = angles[i]
    ax.text(
        angle,  # Adjust angle to prevent overlap
>>>>>>> 4a94f0cfaeb0738880cc409ba8f469928849db88
        1.15, 
        label,
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=8,
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')  # Optional: add a box around labels
    )

<<<<<<< HEAD
colors = ["#FF0000", "#008000","#0000FF","#FFFF00" ]
=======

#colors =["#018A67", "#1868B2", "#DE582B", "#F3A332"]
#colors =["#4DBBD5","#00A087","#105573","#A7EBB2"]
#colors=["#018A67", "#F5B3A5", "#D693BE","#5DBFE9"]
colors = ["#FF0000", "#008000","#0000FF","#FFFF00" ]

>>>>>>> 4a94f0cfaeb0738880cc409ba8f469928849db88
for metric, color in zip(metrics, colors):
    values = list(mean_values[metric].values())
    values = np.append(values, values[0])  # Close the radar plot loop
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'{metric} (p={results[metric]:.3f})', color=color)
    ax.fill(angles, values, alpha=0.25, color=color) ###填充
    for i, value in enumerate(values[:-1]):
        offset_y = -0.06 if value - 0.06 > 0 else 0.05  # Check if the value is close to the boundary
        ax.text(angles[i], value + offset_y, f'{value:.2f}', horizontalalignment='center', fontsize=8, color=color)

ax.spines['polar'].set_visible(False)
ax.set_ylim(0, 1) 
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
plt.tight_layout()
#plt.savefig('../results/combined_radar_plot.png',dpi=300)
plt.savefig('../results/combined_radar_plot_full.png',dpi=300)


<<<<<<< HEAD
##########整体加权均值
weights = {'RAW': 8, 'DeepST': 7, 'PRECAST': 7, 'STitch3D': 5, 'STAligner': 8, 'GraphST': 8,'Spatialign': 8, 'SPIRAL': 8
}
df = pd.read_csv('../results/scib_all_samples.csv')
models = df['Model'].unique()
def calculate_weighted_averages(df, weights):
    weighted_averages = {}
    for model in models:
        model_data = df[df['Model'] == model].iloc[:, 1:-1] 
        if model not in weights:
            continue
        model_weight = weights.get(model, 1)
        weighted_avg = (model_data * model_weight).sum().sum() / (model_data.size * model_weight)
        weighted_averages[model] = weighted_avg
    return weighted_averages

weighted_averages = calculate_weighted_averages(df, weights)
weighted_averages_df = pd.DataFrame(list(weighted_averages.items()), columns=["Model", "Weighted Average"])
weighted_averages_df.to_csv('../results/weighted_averages.csv', index=False)


=======
# 保存 Kruskal-Wallis 检验结果
results_df = pd.DataFrame(list(results.items()), columns=['Metric', 'p-value'])
results_df.to_csv('../results/kruskal_wallis_results.csv', index=False)


import numpy as np
import pandas as pd

# 权重
weights = {'RAW': 8, 'DeepST': 7, 'PRECAST': 7, 'STitch3D': 5, 'STAligner': 8, 'GraphST': 8,'Spatialign': 8, 'SPIRAL': 8
}
# 读取数据
df = pd.read_csv('../results/scib_all_samples.csv')

# 获取唯一的模型列表
models = df['Model'].unique()

# 加权平均计算函数
def calculate_weighted_averages(df, weights):
    weighted_averages = {}
    for model in models:
        # 过滤出每个模型的数据
        model_data = df[df['Model'] == model].iloc[:, 1:-1] 
        if model not in weights:
            continue  # 如果模型不在权重字典中，则跳过
        # 获取该模型的权重
        model_weight = weights.get(model, 1)
        # 计算加权平均
        weighted_avg = (model_data * model_weight).sum().sum() / (model_data.size * model_weight)
        # 保存加权均值
        weighted_averages[model] = weighted_avg
    return weighted_averages

# 计算加权均值
weighted_averages = calculate_weighted_averages(df, weights)

# 将加权均值结果保存为 CSV 文件
weighted_averages_df = pd.DataFrame(list(weighted_averages.items()), columns=["Model", "Weighted Average"])
weighted_averages_df.to_csv('../results/weighted_averages.csv', index=False)
>>>>>>> 4a94f0cfaeb0738880cc409ba8f469928849db88
