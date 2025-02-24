import scib
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.lines as mlines
import warnings
warnings.filterwarnings("ignore")

# 用于存储每个model的结果
results = {
    'Model': ['RAW','PRECAST', 'DeepST', 'STAligner','GraphST','SPIRAL','STitch3D', 'Spatialign'],
    'graph_connectivity': [],
    'iLISI': [],
    'kBET': [],
    'ASW': []
}

##################DLPFC-Sample1
#raw
adata_raw=sc.read_h5ad('raw_adata1.h5ad')
sc.pp.neighbors(adata_raw,  n_neighbors=30,use_rep='X', random_state=666)
sc.tl.umap(adata_raw)
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_raw, label_key="celltype"))
results['iLISI'].append(scib.me.ilisi_graph(adata_raw, batch_key="batch", type_="embed", use_rep="X"))
results['kBET'].append(scib.me.kBET(adata_raw, batch_key="batch", label_key="celltype", type_="embed", embed="X"))
results['ASW'].append(scib.me.silhouette_batch(adata_raw, batch_key="batch", label_key="celltype", embed="X_umap"))
print('raw finish')

# 计算 PRECAST model的指标
adata_PRECAST = sc.read_h5ad('../PRECAST/dlpfc/results/1dlpfc_seuInt.h5ad')
sc.pp.neighbors(adata_PRECAST, n_neighbors=30,use_rep="PRECAST")
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_PRECAST, label_key="Ground_Truth"))
results['iLISI'].append(scib.me.ilisi_graph(adata_PRECAST, batch_key="new_batch", type_="embed", use_rep="PRECAST"))
results['kBET'].append(scib.me.kBET(adata_PRECAST, batch_key="new_batch", label_key="Ground_Truth", type_="embed", embed="PRECAST"))
results['ASW'].append(scib.me.silhouette_batch(adata_PRECAST, batch_key="new_batch", label_key="Ground_Truth", embed="PRECAST"))
print('PRECAST finish')

# 计算 DeepST model的指标
adata_DeepST = ad.read_h5ad("../DeepST/Results/multiple_adata.h5ad")
sc.pp.neighbors(adata_DeepST, n_neighbors=30,use_rep='DeepST_embed')
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_DeepST, label_key="ground_truth"))
results['iLISI'].append(scib.me.ilisi_graph(adata_DeepST, batch_key="batch_name",type_="embed",use_rep="DeepST_embed"))
results['kBET'].append(scib.me.kBET(adata_DeepST, batch_key="batch_name", label_key="ground_truth", type_="embed", embed="DeepST_embed"))
results['ASW'].append(scib.me.silhouette_batch(adata_DeepST, batch_key="batch_name", label_key="ground_truth", embed="DeepST_embed"))
print('DeepST finish')

# 计算 STAligner model的指标
adata_STAligner = sc.read_h5ad('../STAligner/results/staligner_Sample1_DLPFC.h5ad')
sc.pp.neighbors(adata_STAligner, n_neighbors=30,use_rep="STAligner")
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_STAligner, label_key="Ground_Truth"))
results['iLISI'].append(scib.me.ilisi_graph(adata_STAligner, batch_key="batch_name", type_="embed", use_rep="STAligner"))
results['kBET'].append(scib.me.kBET(adata_STAligner, batch_key="batch_name", label_key="Ground_Truth", type_="embed", embed="STAligner"))
results['ASW'].append(scib.me.silhouette_batch(adata_STAligner, batch_key="batch_name", label_key="Ground_Truth", embed="STAligner"))
print('STAligner finish')

# 计算 GraphST model的指标
adata_GraphST = ad.read_h5ad("../GraphST/results/DLPFC_adata.h5ad")
sc.pp.neighbors(adata_GraphST, n_neighbors=30,use_rep="emb_pca")
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_GraphST, label_key="ground_truth"))
results['iLISI'].append(scib.me.ilisi_graph(adata_GraphST, batch_key="new_batch", type_="embed", use_rep="emb_pca"))
results['kBET'].append(scib.me.kBET(adata_GraphST, batch_key="new_batch", label_key="ground_truth", type_="embed", embed="emb_pca"))
results['ASW'].append(scib.me.silhouette_batch(adata_GraphST, batch_key="new_batch", label_key="ground_truth", embed="emb_pca"))
print('GraphST finish')

# 计算 SPIRAL model的指标
adata_SPIRAL = sc.read_h5ad('../SPIRAL/results/spiral_Sample1_DLPFC.h5ad')
adata_SPIRAL.obs["batch"] = adata_SPIRAL.obs["batch"].astype("category")
sc.pp.neighbors(adata_SPIRAL,n_neighbors=30, use_rep="spiral")
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_SPIRAL, label_key="celltype"))
results['iLISI'].append(scib.me.ilisi_graph(adata_SPIRAL, batch_key="batch", type_="embed", use_rep="spiral"))
results['kBET'].append(scib.me.kBET(adata_SPIRAL, batch_key="batch", label_key="celltype", type_="embed", embed="spiral"))
results['ASW'].append(scib.me.silhouette_batch(adata_SPIRAL, batch_key="batch", label_key="celltype", embed="spiral"))
print('SPIRAL finish')

# 计算 STitch3D model的指标
adata_STitch3D = ad.read_h5ad("../STitch3D/results_dlpfc1/DLPFC_adata.h5ad")
adata_STitch3D.obs["layer"] = adata_STitch3D.obs["layer"].astype("category")
adata_STitch3D.obs["slice_id"] = adata_STitch3D.obs["slice_id"].astype("category")
sc.pp.neighbors(adata_STitch3D,n_neighbors=30, use_rep="latent")
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_STitch3D, label_key="layer"))
results['iLISI'].append(scib.me.ilisi_graph(adata_STitch3D, batch_key="slice_id", type_="embed", use_rep="latent"))
results['kBET'].append(scib.me.kBET(adata_STitch3D, batch_key="slice_id", label_key="layer", type_="embed", embed="latent"))
results['ASW'].append(scib.me.silhouette_batch(adata_STitch3D, batch_key="slice_id", label_key="layer", embed="latent"))
print('STitch3D finish')

# 计算 Spatialign model的指标
adata_Spatialign = sc.read_h5ad("../Spatialign/results_dlpfc1/multiple_adata1.h5ad")
sc.pp.neighbors(adata_Spatialign,n_neighbors=30, use_rep="correct")
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_Spatialign, label_key="celltype"))
results['iLISI'].append(scib.me.ilisi_graph(adata_Spatialign, batch_key="new_batch", type_="embed", use_rep="correct"))
results['kBET'].append(scib.me.kBET(adata_Spatialign, batch_key="new_batch", label_key="celltype", type_="embed", embed="correct"))
results['ASW'].append(scib.me.silhouette_batch(adata_Spatialign, batch_key="new_batch", label_key="celltype", embed="correct"))
print('Spatialign finish')

for key in results:
    print(f"Length of '{key}': {len(results[key])}")

df = pd.DataFrame(results)
df.to_csv('../results/1dlpfc_model_performance_results.csv', index=False)
df = pd.read_csv('../results/1dlpfc_model_performance_results.csv')
fig, ax_list = plt.subplots(1, 2, figsize=(16, 6))  
ax_list = ax_list.flatten()
# ---- 绘制分组条形图 ----
metrics = ['graph_connectivity', 'iLISI', 'kBET', 'ASW']
colors = sns.color_palette("Set1", n_colors=8)  # Remove the index column
df.set_index('Model')[metrics].T.plot(kind='bar', ax=ax_list[0], width=0.6, color=colors)
ax_list[0].set_ylabel("Score", fontsize=16, color="black")
ax_list[0].tick_params(axis='x', rotation=45)
ax_list[0].tick_params(axis='y')
# Update the legend title to "Method"
ax_list[0].legend(title="Method", bbox_to_anchor=(1.05, 1), loc='upper left')
ax_list[0].text(-0.05, 1.05, 'A', transform=ax_list[0].transAxes, fontsize=18, fontweight='bold', color='black')
# ---- 绘制箱线图 ----
metrics = ['graph_connectivity', 'iLISI', 'ASW']
df_melted = df.melt(id_vars="Model", value_vars=metrics, var_name="Metric", value_name="Value")
sns.boxplot(x="Model", y="Value", data=df_melted, palette="Set1", ax=ax_list[1])
mean_line = mlines.Line2D([], [], color='grey', linestyle='--', linewidth=1, label='Mean')    
kBET_marker = mlines.Line2D([], [], marker='o', markerfacecolor='white', markeredgewidth=1, color='grey', label='kBET') 
ax_list[1].legend(handles=[mean_line,kBET_marker], loc='upper left',  bbox_to_anchor=(1.05, 1),fontsize=16)
#ax_list[1].set_title("DLPFC1: Model Performance Comparison", fontsize=18, fontweight='bold', fontname='C059')
ax_list[1].set_ylabel('Value', fontsize=16, color="black")
ax_list[1].tick_params(axis='x', rotation=45)
ax_list[1].tick_params(axis='y')
ax_list[1].text(-0.05, 1.05, 'B', transform=ax_list[1].transAxes, fontsize=18, fontweight='bold', color='black')
for i, model in enumerate(df['Model']):
    kBET_value = df.loc[i, 'kBET']
    plt.plot(i, kBET_value, 'o', markerfacecolor='white', markeredgewidth=1, color="grey") 
    # plt.text(i, kBET_value - 0.035, f'{kBET_value:.2f}', horizontalalignment='center', color='black', fontsize=12)  # 在圆点上方标记kBET值

for i, model in enumerate(df["Model"]):
    mean_value = df[df["Model"] == model][['graph_connectivity', 'iLISI', 'ASW']].mean(axis=1).values[0]
    plt.plot([i - 0.4, i + 0.4], [mean_value, mean_value], color='grey', linestyle='--', linewidth=1)
    # plt.text(i, mean_value - 0.01, f'{mean_value:.2f}', horizontalalignment='center', color='black', fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 1])
plt.subplots_adjust(hspace=0.3) 
plt.savefig('../results/1dlpfc_models_means_results.png')


##################DLPFC-Sample2
adata_raw=sc.read_h5ad('raw_adata2.h5ad')
sc.pp.neighbors(adata_raw, n_neighbors=30,use_rep='X', random_state=666)
sc.tl.umap(adata_raw)
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_raw, label_key="celltype"))
results['iLISI'].append(scib.me.ilisi_graph(adata_raw, batch_key="batch", type_="embed", use_rep="X"))
results['kBET'].append(scib.me.kBET(adata_raw, batch_key="batch", label_key="celltype", type_="embed", embed="X"))
results['ASW'].append(scib.me.silhouette_batch(adata_raw, batch_key="batch", label_key="celltype", embed="X_umap"))
print('raw finish')

# 计算 PRECAST model的指标
adata_PRECAST = sc.read_h5ad('../PRECAST/dlpfc/results/2dlpfc_seuInt.h5ad')
sc.pp.neighbors(adata_PRECAST,n_neighbors=30,use_rep="PRECAST",random_state=666)
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_PRECAST, label_key="Ground_Truth"))
results['iLISI'].append(scib.me.ilisi_graph(adata_PRECAST, batch_key="new_batch", type_="embed", use_rep="PRECAST"))
results['kBET'].append(scib.me.kBET(adata_PRECAST, batch_key="new_batch", label_key="Ground_Truth", type_="embed", embed="PRECAST"))
results['ASW'].append(scib.me.silhouette_batch(adata_PRECAST, batch_key="new_batch", label_key="Ground_Truth", embed="PRECAST"))
print('PRECAST finish')

# 计算 DeepST model的指标
adata_DeepST = ad.read_h5ad("../DeepST/Results/multiple_adata2.h5ad")
sc.pp.neighbors(adata_DeepST, use_rep='DeepST_embed',n_neighbors=30,random_state=666)
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_DeepST, label_key="ground_truth"))
results['iLISI'].append(scib.me.ilisi_graph(adata_DeepST, batch_key="batch_name",type_="embed",use_rep="DeepST_embed"))
results['kBET'].append(scib.me.kBET(adata_DeepST, batch_key="batch_name", label_key="ground_truth", type_="embed", embed="DeepST_embed"))
results['ASW'].append(scib.me.silhouette_batch(adata_DeepST, batch_key="batch_name", label_key="ground_truth", embed="DeepST_embed"))
print('DeepST finish')

# 计算 STAligner model的指标
adata_STAligner = sc.read_h5ad('../STAligner/results/staligner_Sample2_DLPFC.h5ad')
sc.pp.neighbors(adata_STAligner, use_rep="STAligner",n_neighbors=30,random_state=666)
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_STAligner, label_key="Ground_Truth"))
results['iLISI'].append(scib.me.ilisi_graph(adata_STAligner, batch_key="batch_name", type_="embed", use_rep="STAligner"))
results['kBET'].append(scib.me.kBET(adata_STAligner, batch_key="batch_name", label_key="Ground_Truth", type_="embed", embed="STAligner"))
results['ASW'].append(scib.me.silhouette_batch(adata_STAligner, batch_key="batch_name", label_key="Ground_Truth", embed="STAligner"))
print('STAligner finish')

# 计算 GraphST model的指标
adata_GraphST = ad.read_h5ad("../GraphST/results/DLPFC_adata2.h5ad")
sc.pp.neighbors(adata_GraphST, use_rep="emb_pca",n_neighbors=30,random_state=666)
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_GraphST, label_key="ground_truth"))
results['iLISI'].append(scib.me.ilisi_graph(adata_GraphST, batch_key="new_batch", type_="embed", use_rep="emb_pca"))
results['kBET'].append(scib.me.kBET(adata_GraphST, batch_key="new_batch", label_key="ground_truth", type_="embed", embed="emb_pca"))
results['ASW'].append(scib.me.silhouette_batch(adata_GraphST, batch_key="new_batch", label_key="ground_truth", embed="emb_pca"))
print('GraphST finish')

# 计算 SPIRAL model的指标
adata_SPIRAL = sc.read_h5ad('../SPIRAL/results/spiral_Sample2_DLPFC.h5ad')
adata_SPIRAL.obs["batch"] = adata_SPIRAL.obs["batch"].astype("category")
sc.pp.neighbors(adata_SPIRAL, use_rep="spiral",n_neighbors=30,random_state=666)
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_SPIRAL, label_key="celltype"))
results['iLISI'].append(scib.me.ilisi_graph(adata_SPIRAL, batch_key="batch", type_="embed", use_rep="spiral"))
results['kBET'].append(scib.me.kBET(adata_SPIRAL, batch_key="batch", label_key="celltype", type_="embed", embed="spiral"))
results['ASW'].append(scib.me.silhouette_batch(adata_SPIRAL, batch_key="batch", label_key="celltype", embed="spiral"))
print('SPIRAL finish')

# 计算 STitch3D model的指标
adata_STitch3D = ad.read_h5ad("../STitch3D/results_dlpfc2/DLPFC_adata2.h5ad")
adata_STitch3D.obs["layer"] = adata_STitch3D.obs["layer"].astype("category")
adata_STitch3D.obs["slice_id"] = adata_STitch3D.obs["slice_id"].astype("category")
sc.pp.neighbors(adata_STitch3D, use_rep="latent",n_neighbors=30,random_state=666)
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_STitch3D, label_key="layer"))
results['iLISI'].append(scib.me.ilisi_graph(adata_STitch3D, batch_key="slice_id", type_="embed", use_rep="latent"))
results['kBET'].append(scib.me.kBET(adata_STitch3D, batch_key="slice_id", label_key="layer", type_="embed", embed="latent"))
results['ASW'].append(scib.me.silhouette_batch(adata_STitch3D, batch_key="slice_id", label_key="layer", embed="latent"))
print('STitch3D finish')

# 计算 Spatialign model的指标
adata_Spatialign = sc.read_h5ad("../Spatialign/results_dlpfc2/multiple_adata2.h5ad")
sc.pp.neighbors(adata_Spatialign, use_rep="correct",n_neighbors=30,random_state=666)
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_Spatialign, label_key="celltype"))
results['iLISI'].append(scib.me.ilisi_graph(adata_Spatialign, batch_key="new_batch", type_="embed", use_rep="correct"))
results['kBET'].append(scib.me.kBET(adata_Spatialign, batch_key="new_batch", label_key="celltype", type_="embed", embed="correct"))
results['ASW'].append(scib.me.silhouette_batch(adata_Spatialign, batch_key="new_batch", label_key="celltype", embed="correct"))
print('Spatialign finish')

# 将数据转换为DataFrame
df = pd.DataFrame(results)
df.to_csv('../results/2dlpfc_model_performance_results.csv', index=False)
df = pd.read_csv('../results/2dlpfc_model_performance_results.csv')
df = pd.read_csv('../results/2dlpfc_model_performance_results.csv')
fig, ax_list = plt.subplots(1, 2, figsize=(16, 6))  
ax_list = ax_list.flatten()
# ---- 绘制分组条形图 ----
metrics = ['graph_connectivity', 'iLISI', 'kBET', 'ASW']
colors = sns.color_palette("Set1", n_colors=8)  # Remove the index column
df.set_index('Model')[metrics].T.plot(kind='bar', ax=ax_list[0], width=0.6, color=colors)
ax_list[0].set_ylabel("Score", fontsize=16, color="black")
ax_list[0].tick_params(axis='x', rotation=45)
ax_list[0].tick_params(axis='y')
# Update the legend title to "Method"
ax_list[0].legend(title="Method", bbox_to_anchor=(1.05, 1), loc='upper left')
ax_list[0].text(-0.05, 1.05, 'A', transform=ax_list[0].transAxes, fontsize=18, fontweight='bold', color='black')
# ---- 绘制箱线图 ----
metrics = ['graph_connectivity', 'iLISI', 'ASW']
df_melted = df.melt(id_vars="Model", value_vars=metrics, var_name="Metric", value_name="Value")
sns.boxplot(x="Model", y="Value", data=df_melted, palette="Set1", ax=ax_list[1])
mean_line = mlines.Line2D([], [], color='grey', linestyle='--', linewidth=1, label='Mean')    
kBET_marker = mlines.Line2D([], [], marker='o', markerfacecolor='white', markeredgewidth=1, color='grey', label='kBET') 
ax_list[1].legend(handles=[mean_line,kBET_marker], loc='upper left',  bbox_to_anchor=(1.05, 1),fontsize=16)
# ax_list[1].set_title("DLPFC2: Model Performance Comparison", fontsize=24, fontweight='bold', fontname='C059')
ax_list[1].set_ylabel('Value', fontsize=16, color="black")
ax_list[1].tick_params(axis='x', rotation=45)
ax_list[1].tick_params(axis='y')
ax_list[1].text(-0.05, 1.05, 'B', transform=ax_list[1].transAxes, fontsize=24, fontweight='bold', color='black')
for i, model in enumerate(df['Model']):
    kBET_value = df.loc[i, 'kBET']
    plt.plot(i, kBET_value, 'o', markerfacecolor='white', markeredgewidth=1, color="grey") 
    # plt.text(i, kBET_value - 0.035, f'{kBET_value:.2f}', horizontalalignment='center', color='black', fontsize=12)  # 在圆点上方标记kBET值

for i, model in enumerate(df["Model"]):
    mean_value = df[df["Model"] == model][['graph_connectivity', 'iLISI', 'ASW']].mean(axis=1).values[0]
    plt.plot([i - 0.4, i + 0.4], [mean_value, mean_value], color='grey', linestyle='--', linewidth=1)
    # plt.text(i, mean_value - 0.045, f'{mean_value:.2f}', horizontalalignment='center', color='black', fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 1])
plt.subplots_adjust(hspace=0.3) 
plt.savefig('../results/2dlpfc_models_means_results.png')  # 保存图像
 

##################DLPFC-Sample3
#raw
adata_raw=sc.read_h5ad('raw_adata3.h5ad')
sc.pp.neighbors(adata_raw, use_rep='X', n_neighbors=15,random_state=666)
sc.tl.umap(adata_raw)
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_raw, label_key="celltype"))
results['iLISI'].append(scib.me.ilisi_graph(adata_raw, batch_key="batch", type_="embed", use_rep="X"))
results['kBET'].append(scib.me.kBET(adata_raw, batch_key="batch", label_key="celltype", type_="embed",embed="X"))
results['ASW'].append(scib.me.silhouette_batch(adata_raw, batch_key="batch", label_key="celltype", embed="X_umap"))
print('raw finish')

# 计算 PRECAST model的指标
adata_PRECAST = sc.read_h5ad('../PRECAST/dlpfc/results/3dlpfc_seuInt.h5ad')
sc.pp.neighbors(adata_PRECAST, use_rep="PRECAST",n_neighbors=15,random_state=666)
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_PRECAST, label_key="Ground_Truth"))
results['iLISI'].append(scib.me.ilisi_graph(adata_PRECAST, batch_key="new_batch", type_="embed", use_rep="PRECAST"))
results['kBET'].append(scib.me.kBET(adata_PRECAST, batch_key="new_batch", label_key="Ground_Truth", type_="embed",embed="PRECAST"))
results['ASW'].append(scib.me.silhouette_batch(adata_PRECAST, batch_key="new_batch", label_key="Ground_Truth", embed="PRECAST"))
print('PRECAST finish')

# 计算 DeepST model的指标
adata_DeepST = ad.read_h5ad("../DeepST/Results/multiple_adata3.h5ad")
sc.pp.neighbors(adata_DeepST, use_rep='DeepST_embed',n_neighbors=15,random_state=666)
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_DeepST, label_key="ground_truth"))
results['iLISI'].append(scib.me.ilisi_graph(adata_DeepST, batch_key="batch_name",type_="embed",use_rep="DeepST_embed"))
results['kBET'].append(scib.me.kBET(adata_DeepST, batch_key="batch_name", label_key="ground_truth", type_="embed",embed="DeepST_embed"))
results['ASW'].append(scib.me.silhouette_batch(adata_DeepST, batch_key="batch_name", label_key="ground_truth", embed="DeepST_embed"))
print('DeepST finish')

# 计算 STAligner model的指标
adata_STAligner = sc.read_h5ad('../STAligner/results/staligner_Sample3_DLPFC.h5ad')
sc.pp.neighbors(adata_STAligner, use_rep="STAligner",n_neighbors=15,random_state=666)
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_STAligner, label_key="Ground_Truth"))
results['iLISI'].append(scib.me.ilisi_graph(adata_STAligner, batch_key="batch_name", type_="embed", use_rep="STAligner"))
results['kBET'].append(scib.me.kBET(adata_STAligner, batch_key="batch_name", label_key="Ground_Truth", type_="embed",embed="STAligner"))
results['ASW'].append(scib.me.silhouette_batch(adata_STAligner, batch_key="batch_name", label_key="Ground_Truth", embed="STAligner"))
print('STAligner finish')

# 计算 GraphST model的指标
adata_GraphST = ad.read_h5ad("../GraphST/results/DLPFC_adata3.h5ad")
sc.pp.neighbors(adata_GraphST, use_rep="emb_pca",n_neighbors=15,random_state=666)
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_GraphST, label_key="ground_truth"))
results['iLISI'].append(scib.me.ilisi_graph(adata_GraphST, batch_key="new_batch", type_="embed", use_rep="emb_pca"))
results['kBET'].append(scib.me.kBET(adata_GraphST, batch_key="new_batch", label_key="ground_truth", type_="embed",embed="emb_pca"))
results['ASW'].append(scib.me.silhouette_batch(adata_GraphST, batch_key="new_batch", label_key="ground_truth", embed="emb_pca"))
print('GraphST finish')

# 计算 SPIRAL model的指标
adata_SPIRAL = sc.read_h5ad('../SPIRAL/results/spiral_Sample3_DLPFC.h5ad')
adata_SPIRAL.obs["batch"] = adata_SPIRAL.obs["batch"].astype("category")
sc.pp.neighbors(adata_SPIRAL, use_rep="spiral",n_neighbors=15,random_state=666)
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_SPIRAL, label_key="celltype"))
results['iLISI'].append(scib.me.ilisi_graph(adata_SPIRAL, batch_key="batch", type_="embed", use_rep="spiral"))
results['kBET'].append(scib.me.kBET(adata_SPIRAL, batch_key="batch", label_key="celltype", type_="embed",embed="spiral"))
results['ASW'].append(scib.me.silhouette_batch(adata_SPIRAL, batch_key="batch", label_key="celltype", embed="spiral"))
print('SPIRAL finish')

# 计算 STitch3D model的指标
adata_STitch3D = ad.read_h5ad("../STitch3D/results_dlpfc3/DLPFC_adata3.h5ad")
adata_STitch3D.obs["layer"] = adata_STitch3D.obs["layer"].astype("category")
adata_STitch3D.obs["slice_id"] = adata_STitch3D.obs["slice_id"].astype("category")
sc.pp.neighbors(adata_STitch3D, use_rep="latent",n_neighbors=15,random_state=666)
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_STitch3D, label_key="layer"))
results['iLISI'].append(scib.me.ilisi_graph(adata_STitch3D, batch_key="slice_id", type_="embed", use_rep="latent"))
results['kBET'].append(scib.me.kBET(adata_STitch3D, batch_key="slice_id", label_key="layer", type_="embed",embed="latent"))
results['ASW'].append(scib.me.silhouette_batch(adata_STitch3D, batch_key="slice_id", label_key="layer", embed="latent"))
print('STitch3D finish')

# 计算 Spatialign model的指标
adata_Spatialign = sc.read_h5ad("../Spatialign/results_dlpfc3/multiple_adata3.h5ad")
sc.pp.neighbors(adata_Spatialign, use_rep="correct",n_neighbors=15,random_state=666)
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_Spatialign, label_key="celltype"))
results['iLISI'].append(scib.me.ilisi_graph(adata_Spatialign, batch_key="new_batch", type_="embed", use_rep="correct"))
results['kBET'].append(scib.me.kBET(adata_Spatialign, batch_key="new_batch", label_key="celltype", type_="embed",embed="correct"))
results['ASW'].append(scib.me.silhouette_batch(adata_Spatialign, batch_key="new_batch", label_key="celltype", embed="correct"))
print('Spatialign finish')

# 将数据转换为DataFrame
df = pd.DataFrame(results)
df.to_csv('../results/3dlpfc_model_performance_results.csv', index=False)

df = pd.read_csv('../results/3dlpfc_model_performance_results.csv')
fig, ax_list = plt.subplots(1, 2, figsize=(16, 6))  
ax_list = ax_list.flatten()
# ---- 绘制分组条形图 ----
metrics = ['graph_connectivity', 'iLISI', 'kBET', 'ASW']
colors = sns.color_palette("Set1", n_colors=8)  # Remove the index column
df.set_index('Model')[metrics].T.plot(kind='bar', ax=ax_list[0], width=0.6, color=colors)
ax_list[0].set_ylabel("Score", fontsize=16, color="black")
ax_list[0].tick_params(axis='x', rotation=45)
ax_list[0].tick_params(axis='y')
# Update the legend title to "Method"
ax_list[0].legend(title="Method", bbox_to_anchor=(1.05, 1), loc='upper left')
ax_list[0].text(-0.05, 1.05, 'A', transform=ax_list[0].transAxes, fontsize=18, fontweight='bold', color='black')
# ---- 绘制箱线图 ----
metrics = ['graph_connectivity', 'iLISI', 'ASW']
df_melted = df.melt(id_vars="Model", value_vars=metrics, var_name="Metric", value_name="Value")
sns.boxplot(x="Model", y="Value", data=df_melted, palette="Set1", ax=ax_list[1])
mean_line = mlines.Line2D([], [], color='grey', linestyle='--', linewidth=1, label='Mean')    
kBET_marker = mlines.Line2D([], [], marker='o', markerfacecolor='white', markeredgewidth=1, color='grey', label='kBET') 
ax_list[1].legend(handles=[mean_line,kBET_marker], loc='upper left',  bbox_to_anchor=(1.05, 1),fontsize=16)
# ax_list[1].set_title("DLPFC3:Model Performance Comparison", fontsize=24, fontweight='bold', fontname='C059')
ax_list[1].set_ylabel('Value', fontsize=16, color="black")
ax_list[1].tick_params(axis='x', rotation=45)
ax_list[1].tick_params(axis='y')
ax_list[1].text(-0.05, 1.05, 'B', transform=ax_list[1].transAxes, fontsize=24, fontweight='bold', color='black')
for i, model in enumerate(df['Model']):
    kBET_value = df.loc[i, 'kBET']
    plt.plot(i, kBET_value, 'o', markerfacecolor='white', markeredgewidth=1, color="grey") 
    # plt.text(i, kBET_value - 0.035, f'{kBET_value:.2f}', horizontalalignment='center', color='black', fontsize=12)  # 在圆点上方标记kBET值

for i, model in enumerate(df["Model"]):
    mean_value = df[df["Model"] == model][['graph_connectivity', 'iLISI', 'ASW']].mean(axis=1).values[0]
    plt.plot([i - 0.4, i + 0.4], [mean_value, mean_value], color='grey', linestyle='--', linewidth=1)
    # plt.text(i, mean_value - 0.045, f'{mean_value:.2f}', horizontalalignment='center', color='black', fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 1])
plt.subplots_adjust(hspace=0.3) 
plt.savefig('../results/3dlpfc_models_means_results.png')


##################human breast cancer-Sample4
results = {
    'Model': ['RAW','PRECAST', 'DeepST', 'STAligner','GraphST','SPIRAL', 'Spatialign'],
    'graph_connectivity': [],
    'iLISI': [],
    'kBET': [],
    'ASW': []
}

#raw
adata_raw=sc.read_h5ad('raw_adata_bc.h5ad')
sc.pp.neighbors(adata_raw,use_rep='X', random_state=666)
sc.tl.umap(adata_raw)
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_raw, label_key="mclust"))
results['iLISI'].append(scib.me.ilisi_graph(adata_raw, batch_key="batch", type_="embed", use_rep="X"))
results['kBET'].append(scib.me.kBET(adata_raw, batch_key="batch", label_key="mclust", type_="embed", embed="X",scaled=True))
results['ASW'].append(scib.me.silhouette_batch(adata_raw, batch_key="batch", label_key="mclust", embed="X_umap"))
print('raw finish')

# 计算 PRECAST model的指标
adata_PRECAST = sc.read_h5ad('../PRECAST/breast cancer/results/bc_seuInt.h5ad')
sc.pp.neighbors(adata_PRECAST,use_rep="PRECAST")
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_PRECAST, label_key="cluster"))
results['iLISI'].append(scib.me.ilisi_graph(adata_PRECAST, batch_key="new_batch", type_="embed", use_rep="PRECAST"))
results['kBET'].append(scib.me.kBET(adata_PRECAST, batch_key="new_batch", label_key="cluster", type_="embed", embed="PRECAST",scaled=True))
results['ASW'].append(scib.me.silhouette_batch(adata_PRECAST, batch_key="new_batch", label_key="cluster", embed="PRECAST"))
print('PRECAST finish')

# 计算 DeepST model的指标
adata_DeepST = sc.read_h5ad("../DeepST/Results/adata_DeepST.h5ad")
sc.pp.neighbors(adata_DeepST,use_rep='DeepST_embed')
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_DeepST, label_key="DeepST_refine_domain"))
results['iLISI'].append(scib.me.ilisi_graph(adata_DeepST, batch_key="new_batch",type_="embed",use_rep="DeepST_embed"))
results['kBET'].append(scib.me.kBET(adata_DeepST, batch_key="new_batch", label_key="DeepST_refine_domain", type_="embed", embed="DeepST_embed",scaled=True))
results['ASW'].append(scib.me.silhouette_batch(adata_DeepST, batch_key="new_batch", label_key="DeepST_refine_domain", embed="DeepST_embed"))
print('DeepST finish')

# 计算 STAligner model的指标
adata_STAligner = sc.read_h5ad('../STAligner/results/staligner_bc.h5ad')
sc.pp.neighbors(adata_STAligner,use_rep="STAligner")
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_STAligner, label_key="mclust"))
results['iLISI'].append(scib.me.ilisi_graph(adata_STAligner, batch_key="new_batch", type_="embed", use_rep="STAligner"))
results['kBET'].append(scib.me.kBET(adata_STAligner, batch_key="new_batch", label_key="mclust", type_="embed", embed="STAligner",scaled=True))
results['ASW'].append(scib.me.silhouette_batch(adata_STAligner, batch_key="new_batch", label_key="mclust", embed="STAligner"))
print('STAligner finish')

# 计算 GraphST model的指标
adata_GraphST = sc.read_h5ad("../GraphST/results/bc_adata.h5ad")
sc.pp.neighbors(adata_GraphST,use_rep="emb_pca")
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_GraphST, label_key="domain"))
results['iLISI'].append(scib.me.ilisi_graph(adata_GraphST, batch_key="new_batch", type_="embed", use_rep="emb_pca"))
results['kBET'].append(scib.me.kBET(adata_GraphST, batch_key="new_batch", label_key="domain", type_="embed", embed="emb_pca",scaled=True))
results['ASW'].append(scib.me.silhouette_batch(adata_GraphST, batch_key="new_batch", label_key="domain", embed="emb_pca"))
print('GraphST finish')

# 计算 SPIRAL model的指标
adata_SPIRAL = sc.read_h5ad('../SPIRAL/results/spiral_bc.h5ad')
adata_SPIRAL.obs["new_batch"] = adata_SPIRAL.obs["new_batch"].astype("category")
sc.pp.neighbors(adata_SPIRAL, use_rep="spiral")
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_SPIRAL, label_key="mclust"))
results['iLISI'].append(scib.me.ilisi_graph(adata_SPIRAL, batch_key="new_batch", type_="embed", use_rep="spiral"))
results['kBET'].append(scib.me.kBET(adata_SPIRAL, batch_key="new_batch", label_key="mclust", type_="embed", embed="spiral",scaled=True))
results['ASW'].append(scib.me.silhouette_batch(adata_SPIRAL, batch_key="new_batch", label_key="mclust", embed="spiral"))
print('SPIRAL finish')

# 计算 Spatialign model的指标
adata_Spatialign = sc.read_h5ad("../Spatialign/results_bc/multiple_adata.h5ad")
sc.pp.neighbors(adata_Spatialign, use_rep="correct")
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_Spatialign, label_key="mclust"))
results['iLISI'].append(scib.me.ilisi_graph(adata_Spatialign, batch_key="new_batch", type_="embed", use_rep="correct"))
results['kBET'].append(scib.me.kBET(adata_Spatialign, batch_key="new_batch", label_key="mclust", type_="embed", embed="correct",scaled=True))
results['ASW'].append(scib.me.silhouette_batch(adata_Spatialign, batch_key="new_batch", label_key="mclust", embed="correct"))
print('Spatialign finish')

for key in results:
    print(f"Length of '{key}': {len(results[key])}")

df = pd.DataFrame(results)
df.to_csv('../results/hbc_model_performance_results.csv', index=True)
df = pd.read_csv('../results/hbc_model_performance_results.csv')
fig, ax_list = plt.subplots(1, 2, figsize=(16, 6))  
ax_list = ax_list.flatten()
# ---- 绘制分组条形图 ----
metrics = ['graph_connectivity', 'iLISI', 'kBET', 'ASW']
colors = sns.color_palette("Set1", n_colors=8)  # Remove the index column
df.set_index('Model')[metrics].T.plot(kind='bar', ax=ax_list[0], width=0.6, color=colors)
ax_list[0].set_ylabel("Score", fontsize=16, color="black")
ax_list[0].tick_params(axis='x', rotation=45)
ax_list[0].tick_params(axis='y')
# Update the legend title to "Method"
ax_list[0].legend(title="Method", bbox_to_anchor=(1.05, 1), loc='upper left')
ax_list[0].text(-0.05, 1.05, 'A', transform=ax_list[0].transAxes, fontsize=18, fontweight='bold', color='black')
# ---- 绘制箱线图 ----
metrics = ['graph_connectivity', 'iLISI', 'ASW']
df_melted = df.melt(id_vars="Model", value_vars=metrics, var_name="Metric", value_name="Value")
sns.boxplot(x="Model", y="Value", data=df_melted, palette="Set1", ax=ax_list[1])
mean_line = mlines.Line2D([], [], color='grey', linestyle='--', linewidth=1, label='Mean')    
kBET_marker = mlines.Line2D([], [], marker='o', markerfacecolor='white', markeredgewidth=1, color='grey', label='kBET') 
ax_list[1].legend(handles=[mean_line,kBET_marker], loc='upper left',  bbox_to_anchor=(1.05, 1),fontsize=16)
# ax_list[1].set_title("hbc: Model Performance Comparison", fontsize=24, fontweight='bold', fontname='C059')
ax_list[1].set_ylabel('Value', fontsize=16, color="black")
ax_list[1].tick_params(axis='x', rotation=45)
ax_list[1].tick_params(axis='y')
ax_list[1].text(-0.05, 1.05, 'B', transform=ax_list[1].transAxes, fontsize=24, fontweight='bold', color='black')
for i, model in enumerate(df['Model']):
    kBET_value = df.loc[i, 'kBET']
    plt.plot(i, kBET_value, 'o', markerfacecolor='white', markeredgewidth=1, color="grey") 
    # plt.text(i, kBET_value - 0.03, f'{kBET_value:.2f}', horizontalalignment='center', color='black', fontsize=12)

for i, model in enumerate(df["Model"]):
    mean_value = df[df["Model"] == model][['graph_connectivity', 'iLISI', 'ASW']].mean(axis=1).values[0]
    plt.plot([i - 0.4, i + 0.4], [mean_value, mean_value], color='grey', linestyle='--', linewidth=1)
    # plt.text(i, mean_value - 0.055, f'mean: {mean_value:.2f}', horizontalalignment='center', color='black', fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 1])
plt.subplots_adjust(hspace=0.3) 
plt.savefig('../results/hbc_models_means_results.png')


##################DLPFC-Sample5
#raw
adata_raw = sc.read_h5ad('raw_adata_7374.h5ad')
sc.pp.neighbors(adata_raw, use_rep='X', random_state=666)
sc.tl.umap(adata_raw)
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_raw, label_key="ground_truth"))
results['iLISI'].append(scib.me.ilisi_graph(adata_raw, batch_key="new_batch", type_="embed", use_rep="X"))
results['kBET'].append(scib.me.kBET(adata_raw, batch_key="new_batch", label_key="ground_truth", type_="embed", embed="X"))
results['ASW'].append(scib.me.silhouette_batch(adata_raw, batch_key="new_batch", label_key="ground_truth", embed="X_umap"))
print('raw finish')

# 计算 PRECAST model的指标
adata_PRECAST = sc.read_h5ad('../PRECAST/dlpfc/results/dlpfc_7374_seuInt_with_all_spatial.h5ad')
sc.pp.neighbors(adata_PRECAST, use_rep="PRECAST",random_state=666)
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_PRECAST, label_key="Ground_Truth"))
results['iLISI'].append(scib.me.ilisi_graph(adata_PRECAST, batch_key="new_batch", type_="embed", use_rep="PRECAST"))
results['kBET'].append(scib.me.kBET(adata_PRECAST, batch_key="new_batch", label_key="Ground_Truth", type_="embed", embed="PRECAST"))
results['ASW'].append(scib.me.silhouette_batch(adata_PRECAST, batch_key="new_batch", label_key="Ground_Truth", embed="PRECAST"))
print('PRECAST finish')

# 计算 DeepST model的指标
adata_DeepST = sc.read_h5ad("../DeepST/Results/multiple_adata_7374.h5ad")
sc.pp.neighbors(adata_DeepST, use_rep='DeepST_embed',random_state=666)
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_DeepST, label_key="ground_truth"))
results['iLISI'].append(scib.me.ilisi_graph(adata_DeepST, batch_key="new_batch",type_="embed",use_rep="DeepST_embed"))
results['kBET'].append(scib.me.kBET(adata_DeepST, batch_key="new_batch", label_key="ground_truth", type_="embed", embed="DeepST_embed"))
results['ASW'].append(scib.me.silhouette_batch(adata_DeepST, batch_key="new_batch", label_key="ground_truth", embed="DeepST_embed"))
print('DeepST finish')

# 计算 STAligner model的指标
adata_STAligner = sc.read_h5ad('../STAligner/results/staligner_Sample_DLPFC_7374.h5ad')
sc.pp.neighbors(adata_STAligner, use_rep="STAligner",random_state=666)
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_STAligner, label_key="ground_truth"))
results['iLISI'].append(scib.me.ilisi_graph(adata_STAligner, batch_key="new_batch", type_="embed", use_rep="STAligner"))
results['kBET'].append(scib.me.kBET(adata_STAligner, batch_key="new_batch", label_key="ground_truth", type_="embed", embed="STAligner"))
results['ASW'].append(scib.me.silhouette_batch(adata_STAligner, batch_key="new_batch", label_key="ground_truth", embed="STAligner"))
print('STAligner finish')

# 计算 GraphST model的指标
adata_GraphST = sc.read_h5ad("../GraphST/results/DLPFC_7374_adata.h5ad")
sc.pp.neighbors(adata_GraphST, use_rep="emb_pca",random_state=666)
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_GraphST, label_key="ground_truth"))
results['iLISI'].append(scib.me.ilisi_graph(adata_GraphST , batch_key="new_batch", type_="embed", use_rep="emb_pca"))
results['kBET'].append(scib.me.kBET(adata_GraphST, batch_key="new_batch", label_key="ground_truth", type_="embed", embed="emb_pca"))
results['ASW'].append(scib.me.silhouette_batch(adata_GraphST, batch_key="new_batch", label_key="ground_truth", embed="emb_pca"))
print('GraphST finish')

# 计算 SPIRAL model的指标
adata_SPIRAL = sc.read_h5ad('../SPIRAL/results/spiral_Sample_DLPFC_7374.h5ad')
sc.pp.neighbors(adata_SPIRAL, use_rep="spiral",random_state=666)
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_SPIRAL, label_key="ground_truth"))
results['iLISI'].append(scib.me.ilisi_graph(adata_SPIRAL, batch_key="new_batch", type_="embed", use_rep="spiral"))
results['kBET'].append(scib.me.kBET(adata_SPIRAL, batch_key="new_batch", label_key="ground_truth", type_="embed", embed="spiral"))
results['ASW'].append(scib.me.silhouette_batch(adata_SPIRAL, batch_key="new_batch", label_key="ground_truth", embed="spiral"))
print('SPIRAL finish')

# 计算 STitch3D model的指标
adata_STitch3D = sc.read_h5ad("../STitch3D/results_dlpfc1/DLPFC_7374_adata.h5ad")
adata_STitch3D.obs["ground_truth"] = adata_STitch3D.obs["ground_truth"].astype("category")
adata_STitch3D.obs["batch"] = adata_STitch3D.obs["batch"].astype("category")
sc.pp.neighbors(adata_STitch3D, use_rep="latent",random_state=666)
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_STitch3D, label_key="ground_truth"))
results['iLISI'].append(scib.me.ilisi_graph(adata_STitch3D, batch_key="batch", type_="embed", use_rep="latent"))
results['kBET'].append(scib.me.kBET(adata_STitch3D, batch_key="batch", label_key="ground_truth", type_="embed", embed="latent"))
results['ASW'].append(scib.me.silhouette_batch(adata_STitch3D, batch_key="batch", label_key="ground_truth", embed="latent"))
print('STitch3D finish')

# 计算 Spatialign model的指标
adata_Spatialign = sc.read_h5ad("../Spatialign/results_dlpfc1/multiple_adata_7374.h5ad")
sc.pp.neighbors(adata_Spatialign, use_rep="correct",random_state=666)
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_Spatialign, label_key="celltype"))
results['iLISI'].append(scib.me.ilisi_graph(adata_Spatialign, batch_key="batch", type_="embed", use_rep="correct"))
results['kBET'].append(scib.me.kBET(adata_Spatialign, batch_key="batch", label_key="celltype", type_="embed", embed="correct"))
results['ASW'].append(scib.me.silhouette_batch(adata_Spatialign, batch_key="batch", label_key="celltype", embed="correct"))
print('Spatialign finish')

# 将数据转换为DataFrame
df = pd.DataFrame(results)
df.to_csv('../results/dlpfc_7374_model_performance_results.csv', index=False)
df = pd.read_csv('../results/dlpfc_7374_model_performance_results.csv')
fig, ax_list = plt.subplots(1, 2, figsize=(16, 6))  
ax_list = ax_list.flatten()
metrics = ['graph_connectivity', 'iLISI', 'kBET', 'ASW']
colors = sns.color_palette("Set1", n_colors=8)  # Remove the index column
df.set_index('Model')[metrics].T.plot(kind='bar', ax=ax_list[0], width=0.6, color=colors)
ax_list[0].set_ylabel("Score", fontsize=16, color="black")
ax_list[0].tick_params(axis='x', rotation=45)
ax_list[0].tick_params(axis='y')
# Update the legend title to "Method"
ax_list[0].legend(title="Method", bbox_to_anchor=(1.05, 1), loc='upper left')
ax_list[0].text(-0.05, 1.05, 'A', transform=ax_list[0].transAxes, fontsize=18, fontweight='bold', color='black')
# ---- 绘制箱线图 ----
metrics = ['graph_connectivity', 'iLISI', 'ASW']
df_melted = df.melt(id_vars="Model", value_vars=metrics, var_name="Metric", value_name="Value")
sns.boxplot(x="Model", y="Value", data=df_melted, palette="Set1", ax=ax_list[1])
mean_line = mlines.Line2D([], [], color='grey', linestyle='--', linewidth=1, label='Mean')    
kBET_marker = mlines.Line2D([], [], marker='o', markerfacecolor='white', markeredgewidth=1, color='grey', label='kBET') 
ax_list[1].legend(handles=[mean_line,kBET_marker], loc='upper left',  bbox_to_anchor=(1.05, 1),fontsize=16)
# ax_list[1].set_title("DLPFC: Model Performance Comparison", fontsize=24, fontweight='bold', fontname='C059')
ax_list[1].set_ylabel('Value', fontsize=16, color="black")
ax_list[1].tick_params(axis='x', rotation=45)
ax_list[1].tick_params(axis='y')
ax_list[1].text(-0.05, 1.05, 'B', transform=ax_list[1].transAxes, fontsize=24, fontweight='bold', color='black')
for i, model in enumerate(df['Model']):
    kBET_value = df.loc[i, 'kBET']
    plt.plot(i, kBET_value, 'o', markerfacecolor='white', markeredgewidth=1, color="grey") 
    # plt.text(i, kBET_value - 0.035, f'{kBET_value:.2f}', horizontalalignment='center', color='black', fontsize=12)  # 在圆点上方标记kBET值

for i, model in enumerate(df["Model"]):
    mean_value = df[df["Model"] == model][['graph_connectivity', 'iLISI', 'ASW']].mean(axis=1).values[0]
    plt.plot([i - 0.4, i + 0.4], [mean_value, mean_value], color='grey', linestyle='--', linewidth=1)
    # plt.text(i, mean_value - 0.045, f'{mean_value:.2f}', horizontalalignment='center', color='black', fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 1])
plt.subplots_adjust(hspace=0.3) 
plt.savefig('../results/7374_dlpfc_models_means_results.png')


##################DLPFC-Sample1+2+3 Sample6
#raw
adata_raw=sc.read_h5ad('raw_adata_all.h5ad')
sc.pp.neighbors(adata_raw, use_rep='X', random_state=666)
sc.tl.umap(adata_raw)
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_raw, label_key="celltype"))
results['iLISI'].append(scib.me.ilisi_graph(adata_raw, batch_key="batch", type_="embed", use_rep="X"))
results['kBET'].append(scib.me.kBET(adata_raw, batch_key="batch", label_key="celltype", type_="embed", embed="X"))
results['ASW'].append(scib.me.silhouette_batch(adata_raw, batch_key="batch", label_key="celltype", embed="X_umap"))
print('raw finish')

# 计算 PRECAST model的指标
adata_PRECAST = sc.read_h5ad('../PRECAST/dlpfc/results/dlpfc_all_seuInt.h5ad')
sc.pp.neighbors(adata_PRECAST, use_rep="PRECAST",random_state=666)
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_PRECAST, label_key="Ground_Truth"))
results['iLISI'].append(scib.me.ilisi_graph(adata_PRECAST, batch_key="sample_group", type_="embed", use_rep="PRECAST"))
results['kBET'].append(scib.me.kBET(adata_PRECAST, batch_key="sample_group", label_key="Ground_Truth", type_="embed", embed="PRECAST"))
results['ASW'].append(scib.me.silhouette_batch(adata_PRECAST, batch_key="sample_group", label_key="Ground_Truth", embed="PRECAST"))
print('PRECAST finish')

# 计算 DeepST model的指标
adata_DeepST = ad.read_h5ad("../DeepST/Results/all_multiple_adata.h5ad")
sc.pp.neighbors(adata_DeepST, use_rep='DeepST_embed',random_state=666)
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_DeepST, label_key="ground_truth"))
results['iLISI'].append(scib.me.ilisi_graph(adata_DeepST, batch_key="sample_name",type_="embed",use_rep="DeepST_embed"))
results['kBET'].append(scib.me.kBET(adata_DeepST, batch_key="sample_name", label_key="ground_truth", type_="embed", embed="DeepST_embed"))
results['ASW'].append(scib.me.silhouette_batch(adata_DeepST, batch_key="sample_name", label_key="ground_truth", embed="DeepST_embed"))
print('DeepST finish')

# 计算 STAligner model的指标
adata_STAligner = sc.read_h5ad('../STAligner/results/staligner_all_DLPFC.h5ad')
sc.pp.neighbors(adata_STAligner, use_rep="STAligner",random_state=666)
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_STAligner, label_key="Ground_Truth"))
results['iLISI'].append(scib.me.ilisi_graph(adata_STAligner, batch_key="batch_name", type_="embed", use_rep="STAligner"))
results['kBET'].append(scib.me.kBET(adata_STAligner, batch_key="batch_name", label_key="Ground_Truth", type_="embed", embed="STAligner"))
results['ASW'].append(scib.me.silhouette_batch(adata_STAligner, batch_key="batch_name", label_key="Ground_Truth", embed="STAligner"))
print('STAligner finish')

# 计算 GraphST model的指标
adata_GraphST = ad.read_h5ad("../GraphST/results/all_DLPFC_adata.h5ad")
sc.pp.neighbors(adata_GraphST, use_rep="emb_pca",random_state=666)
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_GraphST, label_key="ground_truth"))
results['iLISI'].append(scib.me.ilisi_graph(adata_GraphST , batch_key="batch_name", type_="embed", use_rep="emb_pca"))
results['kBET'].append(scib.me.kBET(adata_GraphST, batch_key="batch_name", label_key="ground_truth", type_="embed", embed="emb_pca"))
results['ASW'].append(scib.me.silhouette_batch(adata_GraphST, batch_key="batch_name", label_key="ground_truth", embed="emb_pca"))
print('GraphST finish')

# 计算 SPIRAL model的指标
adata_SPIRAL = sc.read_h5ad('../SPIRAL/results/spiral_Sample_DLPFC_all.h5ad')
sc.pp.neighbors(adata_SPIRAL, use_rep="spiral",random_state=666)
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_SPIRAL, label_key="celltype"))
results['iLISI'].append(scib.me.ilisi_graph(adata_SPIRAL, batch_key="batch_name", type_="embed", use_rep="spiral"))
results['kBET'].append(scib.me.kBET(adata_SPIRAL, batch_key="batch_name", label_key="celltype", type_="embed", embed="spiral"))
results['ASW'].append(scib.me.silhouette_batch(adata_SPIRAL, batch_key="batch_name", label_key="celltype", embed="spiral"))
print('SPIRAL finish')

# 计算 STitch3D model的指标
adata_STitch3D = ad.read_h5ad("../STitch3D/results_dlpfc_all/all_DLPFC_adata.h5ad")
adata_STitch3D.obs["layer"] = adata_STitch3D.obs["layer"].astype("category")
adata_STitch3D.obs["batch_name"] = adata_STitch3D.obs["batch_name"].astype("category")
sc.pp.neighbors(adata_STitch3D, use_rep="latent",random_state=666)
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_STitch3D, label_key="layer"))
results['iLISI'].append(scib.me.ilisi_graph(adata_STitch3D, batch_key="batch_name", type_="embed", use_rep="latent"))
results['kBET'].append(scib.me.kBET(adata_STitch3D, batch_key="batch_name", label_key="layer", type_="embed", embed="latent"))
results['ASW'].append(scib.me.silhouette_batch(adata_STitch3D, batch_key="batch_name", label_key="layer", embed="latent"))
print('STitch3D finish')

# 计算 Spatialign model的指标
adata_Spatialign = sc.read_h5ad("../Spatialign/results_dlpfc_all/multiple_adata.h5ad")
sc.pp.neighbors(adata_Spatialign, use_rep="correct",random_state=666)
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_Spatialign, label_key="celltype"))
results['iLISI'].append(scib.me.ilisi_graph(adata_Spatialign, batch_key="new_batch", type_="embed", use_rep="correct"))
results['kBET'].append(scib.me.kBET(adata_Spatialign, batch_key="new_batch", label_key="celltype", type_="embed", embed="correct"))
results['ASW'].append(scib.me.silhouette_batch(adata_Spatialign, batch_key="new_batch", label_key="celltype", embed="correct"))
print('Spatialign finish')

# 将数据转换为DataFrame
df = pd.DataFrame(results)
df.to_csv('../results/all_dlpfc_model_performance_results.csv', index=False)
df = pd.read_csv('../results/mob_metrics_results.csv')
fig, ax_list = plt.subplots(1, 2, figsize=(16, 6))  
ax_list = ax_list.flatten()
# ---- 绘制分组条形图 ----
metrics = ['graph_connectivity', 'iLISI', 'kBET', 'ASW']
colors = sns.color_palette("Set1", n_colors=8)  # Remove the index column
df.set_index('Model')[metrics].T.plot(kind='bar', ax=ax_list[0], width=0.6, color=colors)
ax_list[0].set_ylabel("Score", fontsize=16, color="black")
ax_list[0].tick_params(axis='x', rotation=45)
ax_list[0].tick_params(axis='y')
# Update the legend title to "Method"
ax_list[0].legend(title="Method", bbox_to_anchor=(1.05, 1), loc='upper left')
ax_list[0].text(-0.05, 1.05, 'A', transform=ax_list[0].transAxes, fontsize=18, fontweight='bold', color='black')
# ---- 绘制箱线图 ----
metrics = ['graph_connectivity', 'iLISI', 'ASW']
df_melted = df.melt(id_vars="Model", value_vars=metrics, var_name="Metric", value_name="Value")
sns.boxplot(x="Model", y="Value", data=df_melted, palette="Set1", ax=ax_list[1])
mean_line = mlines.Line2D([], [], color='grey', linestyle='--', linewidth=1, label='Mean')    
kBET_marker = mlines.Line2D([], [], marker='o', markerfacecolor='white', markeredgewidth=1, color='grey', label='kBET') 
ax_list[1].legend(handles=[mean_line, kBET_marker], loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=16)
# ax_list[1].set_title("mob: Model Performance Comparison", fontsize=24, fontweight='bold', fontname='C059')
ax_list[1].set_ylabel('Value', fontsize=16, color="black")
ax_list[1].tick_params(axis='x', rotation=45)
ax_list[1].tick_params(axis='y')
ax_list[1].text(-0.05, 1.05, 'B', transform=ax_list[1].transAxes, fontsize=24, fontweight='bold', color='black')
for i, model in enumerate(df['Model']):
    kBET_value = df.loc[i, 'kBET']
    plt.plot(i, kBET_value, 'o', markerfacecolor='white', markeredgewidth=1, color="grey") 
    # plt.text(i, kBET_value - 0.03, f'{kBET_value:.2f}', horizontalalignment='center', color='black', fontsize=12)  # 在圆点上方标记kBET值

for i, model in enumerate(df["Model"]):
    mean_value = df[df["Model"] == model][['graph_connectivity', 'iLISI', 'ASW']].mean(axis=1).values[0]
    plt.plot([i - 0.4, i + 0.4], [mean_value, mean_value], color='grey', linestyle='--', linewidth=1)
    # plt.text(i, mean_value + 0.01, f'mean: {mean_value:.2f}', horizontalalignment='center', color='black', fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 1])
plt.subplots_adjust(hspace=0.3) 
plt.savefig('../results/mob_models_means_results.png')



##################mouse coronal brain-Sample7
results = {
    'Model': ['RAW','PRECAST', 'DeepST', 'STAligner','GraphST','SPIRAL', 'Spatialign'],
    'graph_connectivity': [],
    'iLISI': [],
    'kBET': [],
    'ASW': []
}

#raw
adata_raw=sc.read_h5ad('raw_adata_coronal.h5ad')
sc.pp.neighbors(adata_raw,  n_neighbors=30,use_rep='X', random_state=666)
sc.tl.umap(adata_raw)
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_raw, label_key="ground_truth"))
results['iLISI'].append(scib.me.ilisi_graph(adata_raw, batch_key="batch", type_="embed", use_rep="X"))
results['kBET'].append(scib.me.kBET(adata_raw, batch_key="batch", label_key="ground_truth", type_="embed", embed="X"))
results['ASW'].append(scib.me.silhouette_batch(adata_raw, batch_key="batch", label_key="ground_truth", embed="X_umap"))
print('raw finish')

# 计算 PRECAST model的指标
adata_PRECAST = sc.read_h5ad('../PRECAST/coronal/results/coronal_seuInt_truth.h5ad')
sc.pp.neighbors(adata_PRECAST, n_neighbors=30,use_rep="PRECAST")
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_PRECAST, label_key="ground_truth"))
results['iLISI'].append(scib.me.ilisi_graph(adata_PRECAST, batch_key="new_batch", type_="embed", use_rep="PRECAST"))
results['kBET'].append(scib.me.kBET(adata_PRECAST, batch_key="new_batch", label_key="ground_truth", type_="embed", embed="PRECAST"))
results['ASW'].append(scib.me.silhouette_batch(adata_PRECAST, batch_key="new_batch", label_key="ground_truth", embed="PRECAST"))
print('PRECAST finish')

# 计算 DeepST model的指标
adata_DeepST = sc.read_h5ad("../DeepST/results_coronal/adata_DeepST.h5ad")
sc.pp.neighbors(adata_DeepST, n_neighbors=30,use_rep='DeepST_embed')
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_DeepST, label_key="ground_truth"))
results['iLISI'].append(scib.me.ilisi_graph(adata_DeepST, batch_key="new_batch",type_="embed",use_rep="DeepST_embed"))
results['kBET'].append(scib.me.kBET(adata_DeepST, batch_key="new_batch", label_key="ground_truth", type_="embed", embed="DeepST_embed"))
results['ASW'].append(scib.me.silhouette_batch(adata_DeepST, batch_key="new_batch", label_key="ground_truth", embed="DeepST_embed"))
print('DeepST finish')

# 计算 STAligner model的指标
adata_STAligner = sc.read_h5ad('../STAligner/results/staligner_coronal.h5ad')
sc.pp.neighbors(adata_STAligner, n_neighbors=30,use_rep="STAligner")
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_STAligner, label_key="ground_truth"))
results['iLISI'].append(scib.me.ilisi_graph(adata_STAligner, batch_key="new_batch", type_="embed", use_rep="STAligner"))
results['kBET'].append(scib.me.kBET(adata_STAligner, batch_key="new_batch", label_key="ground_truth", type_="embed", embed="STAligner"))
results['ASW'].append(scib.me.silhouette_batch(adata_STAligner, batch_key="new_batch", label_key="ground_truth", embed="STAligner"))
print('STAligner finish')

# 计算 GraphST model的指标
adata_GraphST = sc.read_h5ad("../GraphST/results/coronal_adata.h5ad")
sc.pp.neighbors(adata_GraphST, n_neighbors=30,use_rep="emb_pca")
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_GraphST, label_key="ground_truth"))
results['iLISI'].append(scib.me.ilisi_graph(adata_GraphST, batch_key="new_batch", type_="embed", use_rep="emb_pca"))
results['kBET'].append(scib.me.kBET(adata_GraphST, batch_key="new_batch", label_key="ground_truth", type_="embed", embed="emb_pca"))
results['ASW'].append(scib.me.silhouette_batch(adata_GraphST, batch_key="new_batch", label_key="ground_truth", embed="emb_pca"))
print('GraphST finish')

# 计算 SPIRAL model的指标
adata_SPIRAL = sc.read_h5ad('../SPIRAL/results/spiral_coronal.h5ad')
adata_SPIRAL.obs["batch"] = adata_SPIRAL.obs["batch"].astype("category")
sc.pp.neighbors(adata_SPIRAL,n_neighbors=30, use_rep="spiral")
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_SPIRAL, label_key="ground_truth"))
results['iLISI'].append(scib.me.ilisi_graph(adata_SPIRAL, batch_key="batch", type_="embed", use_rep="spiral"))
results['kBET'].append(scib.me.kBET(adata_SPIRAL, batch_key="batch", label_key="ground_truth", type_="embed", embed="spiral"))
results['ASW'].append(scib.me.silhouette_batch(adata_SPIRAL, batch_key="batch", label_key="ground_truth", embed="spiral"))
print('SPIRAL finish')

# 计算 Spatialign model的指标
adata_Spatialign = sc.read_h5ad("../Spatialign/results_coronal/multiple_adata.h5ad")
sc.pp.neighbors(adata_Spatialign,n_neighbors=30, use_rep="correct")
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_Spatialign, label_key="ground_truth"))
results['iLISI'].append(scib.me.ilisi_graph(adata_Spatialign, batch_key="new_batch", type_="embed", use_rep="correct"))
results['kBET'].append(scib.me.kBET(adata_Spatialign, batch_key="new_batch", label_key="ground_truth", type_="embed", embed="correct"))
results['ASW'].append(scib.me.silhouette_batch(adata_Spatialign, batch_key="new_batch", label_key="ground_truth", embed="correct"))
print('Spatialign finish')

for key in results:
    print(f"Length of '{key}': {len(results[key])}")

df = pd.DataFrame(results)
df.to_csv('../results/coronal_model_performance_results.csv', index=False)

df=pd.read_csv("../results/coronal_model_performance_results.csv")
df = pd.read_csv('../results/coronal_model_performance_results.csv')
fig, ax_list = plt.subplots(1, 2, figsize=(16, 6))  
ax_list = ax_list.flatten()
# ---- 绘制分组条形图 ----
metrics = ['graph_connectivity', 'iLISI', 'kBET', 'ASW']
colors = sns.color_palette("Set1", n_colors=8)  # Remove the index column
df.set_index('Model')[metrics].T.plot(kind='bar', ax=ax_list[0], width=0.6, color=colors)
ax_list[0].set_ylabel("Score", fontsize=16, color="black")
ax_list[0].tick_params(axis='x', rotation=45)
ax_list[0].tick_params(axis='y')
# Update the legend title to "Method"
ax_list[0].legend(title="Method", bbox_to_anchor=(1.05, 1), loc='upper left')
ax_list[0].text(-0.05, 1.05, 'A', transform=ax_list[0].transAxes, fontsize=18, fontweight='bold', color='black')
# ---- 绘制箱线图 ----
metrics = ['graph_connectivity', 'iLISI', 'ASW']
df_melted = df.melt(id_vars="Model", value_vars=metrics, var_name="Metric", value_name="Value")
sns.boxplot(x="Model", y="Value", data=df_melted, palette="Set1", ax=ax_list[1])
mean_line = mlines.Line2D([], [], color='grey', linestyle='--', linewidth=1, label='Mean')    
kBET_marker = mlines.Line2D([], [], marker='o', markerfacecolor='white', markeredgewidth=1, color='grey', label='kBET') 
ax_list[1].legend(handles=[mean_line,kBET_marker], loc='upper left',  bbox_to_anchor=(1.05, 1),fontsize=16)
#ax_list[1].set_title("Coronal: Model Performance Comparison", fontsize=18, fontweight='bold', fontname='C059')
ax_list[1].set_ylabel('Value', fontsize=16, color="black")
ax_list[1].tick_params(axis='x', rotation=45)
ax_list[1].tick_params(axis='y')
ax_list[1].text(-0.05, 1.05, 'B', transform=ax_list[1].transAxes, fontsize=18, fontweight='bold', color='black')
for i, model in enumerate(df['Model']):
    kBET_value = df.loc[i, 'kBET']
    plt.plot(i, kBET_value, 'o', markerfacecolor='white', markeredgewidth=1, color="grey") 
    # plt.text(i, kBET_value - 0.035, f'{kBET_value:.2f}', horizontalalignment='center', color='black', fontsize=12)  # 在圆点上方标记kBET值

for i, model in enumerate(df["Model"]):
    mean_value = df[df["Model"] == model][['graph_connectivity', 'iLISI', 'ASW']].mean(axis=1).values[0]
    plt.plot([i - 0.4, i + 0.4], [mean_value, mean_value], color='grey', linestyle='--', linewidth=1)
    # plt.text(i, mean_value - 0.01, f'{mean_value:.2f}', horizontalalignment='center', color='black', fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 1])
plt.subplots_adjust(hspace=0.3) 
plt.savefig('../results/coronal_models_means_results.png')



##################mouse olfactory bulb-Sample8
results = {
    'Model': ['RAW','STAligner','SPIRAL', 'Spatialign'],
    'graph_connectivity': [],
    'iLISI': [],
    'kBET': [],
    'ASW': []
}
######raw
adata_raw=sc.read_h5ad('raw_adata_mob.h5ad')
sc.pp.pca(adata_raw)
sc.pp.neighbors(adata_raw, use_rep='X_pca',n_neighbors=20,random_state=666)
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_raw, label_key="mclust"))
results['iLISI'].append(scib.me.ilisi_graph(adata_raw, batch_key="slice_name", type_="embed", use_rep="X_pca"))
results['kBET'].append(scib.me.kBET(adata_raw, batch_key="slice_name", label_key="mclust", type_="embed", embed="X_pca"))
results['ASW'].append(scib.me.silhouette_batch(adata_raw, batch_key="slice_name", label_key="mclust", embed="X_pca"))
print('raw finish')

######STAligner
adata_STAligner = sc.read_h5ad('../STAligner/results/staligner_mob3.h5ad')
sc.pp.neighbors(adata_STAligner, n_neighbors=20,use_rep="STAligner",random_state=666)
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_STAligner, label_key="louvain"))
results['iLISI'].append(scib.me.ilisi_graph(adata_STAligner, batch_key="batch_name", type_="embed", use_rep="STAligner"))
results['kBET'].append(scib.me.kBET(adata_STAligner, batch_key="batch_name", label_key="louvain", type_="embed", embed="STAligner"))
results['ASW'].append(scib.me.silhouette_batch(adata_STAligner, batch_key="batch_name", label_key="louvain", embed="STAligner"))
print('STAligner finish')

######SPIRAL
adata_SPIRAL = sc.read_h5ad('../SPIRAL/results/spiral_mob3.h5ad')
sc.pp.neighbors(adata_SPIRAL, use_rep="spiral",n_neighbors=20,random_state=666)
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_SPIRAL, label_key="SPIRAL"))
results['iLISI'].append(scib.me.ilisi_graph(adata_SPIRAL, batch_key="batch", type_="embed", use_rep="spiral"))
results['kBET'].append(scib.me.kBET(adata_SPIRAL, batch_key="batch", label_key="SPIRAL", type_="embed", embed="spiral"))
results['ASW'].append(scib.me.silhouette_batch(adata_SPIRAL, batch_key="batch", label_key="SPIRAL", embed="spiral"))
print('SPIRAL finish')

########spatiAlign
adata_spatiAlign = sc.read_h5ad("../Spatialign/results_mob/multiple_adata.h5ad")
sc.pp.neighbors(adata_spatiAlign, use_rep="correct",n_neighbors=20,random_state=666)
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_spatiAlign, label_key="louvain"))
results['iLISI'].append(scib.me.ilisi_graph(adata_spatiAlign, batch_key="new_batch", type_="embed", use_rep="correct"))
results['kBET'].append(scib.me.kBET(adata_spatiAlign, batch_key="new_batch", label_key="louvain", type_="embed", embed="correct"))
results['ASW'].append(scib.me.silhouette_batch(adata_spatiAlign, batch_key="new_batch", label_key="louvain", embed="correct"))

print('spatiAlign finish')

df1 = pd.DataFrame(results)
df1.to_csv('../results/mob_metrics_results1.csv', index=False)


import scanpy as sc
import scib
import numpy as np
import pandas as pd
results = {
    'Model': ['GraphST'],
    'graph_connectivity': [],
    'iLISI': [],
    'kBET': [],
    'ASW': []
}
adata_GraphST = sc.read_h5ad("../GraphST/results/mob_adata.h5ad")
sc.pp.neighbors(adata_GraphST, use_rep='emb_pca', n_neighbors=20,random_state=666)
batch_size = 1000
n_batches = adata_GraphST.n_obs // batch_size + (1 if adata_GraphST.n_obs % batch_size > 0 else 0)
iLISI_scores = []
graph_conn_scores = []
kBET_scores = []
ASW_scores = []
for i in range(n_batches):
    batch_adata = adata_GraphST[i * batch_size: (i + 1) * batch_size]
    try:
        ilisi_result = scib.me.ilisi_graph(batch_adata, batch_key="new_batch", type_="embed", use_rep="emb_pca")
        iLISI_scores.append(ilisi_result)
        # kBET_result = scib.me.kBET(batch_adata, batch_key="new_batch", label_key="domain", type_="embed", embed="emb_pca")
        # kBET_scores.append(kBET_result)
        # print(f"Batch {i + 1}/{n_batches} processed successfully.")
    except Exception as e:
        print(f"Error in batch {i + 1}: {e}")
        continue

final_iLISI_score = np.nanmean(iLISI_scores)
final_kBET_score = np.nanmean(kBET_scores)
results['graph_connectivity'].append(scib.me.graph_connectivity(adata_GraphST, label_key="domain"))
results['kBET'].append(final_kBET_score)
results['iLISI'].append(final_iLISI_score)
results['ASW'].append(scib.me.silhouette_batch(adata_GraphST, batch_key="new_batch", label_key="domain", embed="emb_pca"))
df2 = pd.DataFrame(results)
df2.to_csv('../results/mob_metrics_results2.csv', index=False)
####合并df1和df2
df=pd.read_csv("../results/mob_metrics_results.csv")
df = pd.read_csv('../results/mob_metrics_results.csv')
fig, ax_list = plt.subplots(1, 2, figsize=(16, 6))  
ax_list = ax_list.flatten()
# ---- 绘制分组条形图 ----
metrics = ['graph_connectivity', 'iLISI', 'kBET', 'ASW']
colors = sns.color_palette("Set1", n_colors=8)  # Remove the index column
df.set_index('Model')[metrics].T.plot(kind='bar', ax=ax_list[0], width=0.6, color=colors)
ax_list[0].set_ylabel("Score", fontsize=16, color="black")
ax_list[0].tick_params(axis='x', rotation=45)
ax_list[0].tick_params(axis='y')
# Update the legend title to "Method"
ax_list[0].legend(title="Method", bbox_to_anchor=(1.05, 1), loc='upper left')
ax_list[0].text(-0.05, 1.05, 'A', transform=ax_list[0].transAxes, fontsize=18, fontweight='bold', color='black')
# ---- 绘制箱线图 ----
metrics = ['graph_connectivity', 'iLISI', 'ASW']
df_melted = df.melt(id_vars="Model", value_vars=metrics, var_name="Metric", value_name="Value")
sns.boxplot(x="Model", y="Value", data=df_melted, palette="Set1", ax=ax_list[1])
mean_line = mlines.Line2D([], [], color='grey', linestyle='--', linewidth=1, label='Mean')    
kBET_marker = mlines.Line2D([], [], marker='o', markerfacecolor='white', markeredgewidth=1, color='grey', label='kBET') 
ax_list[1].legend(handles=[mean_line, kBET_marker], loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=16)
# ax_list[1].set_title("mob: Model Performance Comparison", fontsize=24, fontweight='bold', fontname='C059')
ax_list[1].set_ylabel('Value', fontsize=16, color="black")
ax_list[1].tick_params(axis='x', rotation=45)
ax_list[1].tick_params(axis='y')
ax_list[1].text(-0.05, 1.05, 'B', transform=ax_list[1].transAxes, fontsize=24, fontweight='bold', color='black')
for i, model in enumerate(df['Model']):
    kBET_value = df.loc[i, 'kBET']
    plt.plot(i, kBET_value, 'o', markerfacecolor='white', markeredgewidth=1, color="grey") 
    # plt.text(i, kBET_value - 0.03, f'{kBET_value:.2f}', horizontalalignment='center', color='black', fontsize=12)  # 在圆点上方标记kBET值

for i, model in enumerate(df["Model"]):
    mean_value = df[df["Model"] == model][['graph_connectivity', 'iLISI', 'ASW']].mean(axis=1).values[0]
    plt.plot([i - 0.4, i + 0.4], [mean_value, mean_value], color='grey', linestyle='--', linewidth=1)
    # plt.text(i, mean_value + 0.01, f'mean: {mean_value:.2f}', horizontalalignment='center', color='black', fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 1])
plt.subplots_adjust(hspace=0.3) 
plt.savefig('../results/mob_models_means_results.png')

