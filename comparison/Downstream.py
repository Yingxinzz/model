###########RAWdata
import scanpy as sc
import pandas as pd
import os
datasets = ['151673', '151674', '151675', '151676',
               '151669', '151670','151671', '151672',
               '151507', '151508', '151509', '151510'
               ]
print(datasets)
file_fold = '../RAW_SLICE/DLPFC/'
adatas=[]
for dataset in datasets:   
    adata = sc.read_visium(file_fold+dataset, count_file=dataset+'_filtered_feature_bc_matrix.h5', load_images=True)
    adata.var_names_make_unique()
    Ann_df = pd.read_csv(os.path.join(file_fold+dataset, dataset + '_truth.txt'), sep='\t', header=None, index_col=0)
    Ann_df.columns = ['Ground Truth']
    Ann_df[Ann_df.isna()] = "unknown"
    adata.obs['celltype'] = Ann_df.loc[adata.obs_names, 'Ground Truth'].astype('category')
    adata = adata[adata.obs['celltype']!='unknown']
    adata.obs_names = [x+'_'+dataset for x in adata.obs_names]
    adata.obs['batch'] = dataset
    # Normalization
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var['highly_variable']]
    adatas.append(adata)

raw_adata = adatas[0].concatenate(*adatas[1:], batch_key='batch', batch_categories=datasets)
raw_adata.obs["new_batch"] = raw_adata.obs["batch"]  
new_batch_1 = raw_adata.obs["new_batch"].isin(['151673', '151674', '151675', '151676'])
new_batch_2 = raw_adata.obs["new_batch"].isin(['151669', '151670', '151671', '151672'])
new_batch_3 = raw_adata.obs["new_batch"].isin(['151507', '151508', '151509', '151510'])
raw_adata.obs["batch_name"] = list(sum(new_batch_1)*['Sample 1'])+list(sum(new_batch_2)*['Sample 2'])+list(sum(new_batch_3)*['Sample 3'])
import scanpy as sc
from sklearn.mixture import GaussianMixture
sc.pp.scale(raw_adata)
X = raw_adata.X
n_components = 7 
gmm = GaussianMixture(n_components=n_components, random_state=42)
raw_adata.obs['mclust'] = gmm.fit_predict(X)
raw_adata.obs["mclust"] = raw_adata.obs["mclust"].astype("category")
raw_adata.write('raw_adata_12.h5ad')

########STAligner
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('../')
import STAligner
os.environ['R_HOME'] = '/root/anaconda3/envs/SS/lib/R'
os.environ['R_LIBS'] = '/root/anaconda3/envs/SS/lib/R/library'
os.environ['R_USER'] = "/root/anaconda3/envs/SG/lib/python3.8/site-packages/rpy2"
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
import scipy.linalg
from scipy.sparse import csr_matrix
import torch
used_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

datasets = ['151673', '151674', '151675', '151676',
               '151669', '151670','151671', '151672',
               '151507', '151508', '151509', '151510'
               ]
file_fold = '../../RAW_SLICE/DLPFC/'
Batch_list = []
adj_list = []
for dataset in datasets:
    print(dataset)
    adata = sc.read_visium(file_fold+dataset, count_file=dataset+'_filtered_feature_bc_matrix.h5', load_images=True)
    adata.var_names_make_unique()
    Ann_df = pd.read_csv(os.path.join(file_fold+dataset, dataset + '_truth.txt'), sep='\t', header=None, index_col=0)
    Ann_df.columns = ['Ground Truth']
    Ann_df[Ann_df.isna()] = "unknown"
    adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'Ground Truth'].astype('category')
    adata.obs_names = [x+'_'+dataset for x in adata.obs_names]
    adata.obs['batch'] = dataset  
    STAligner.Cal_Spatial_Net(adata, rad_cutoff=150) 
    STAligner.Stats_Spatial_Net(adata)
    ## Normalization
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=10000) #ensure enough common HVGs in the combined matrix
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var['highly_variable']]
    adj_list.append(adata.uns['adj'])
    Batch_list.append(adata)

### Concat the scanpy objects for multiple slices
adata_concat = ad.concat(Batch_list, label="slice_name", keys=datasets)
adata_concat.obs['Ground_Truth'] = adata_concat.obs['Ground Truth'].astype('category')
adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')
print('adata_concat.shape: ', adata_concat.shape)
# combine silce names into sample name
new_batch_1 = adata_concat.obs["slice_name"].isin(['151673', '151674', '151675', '151676'])
new_batch_2 = adata_concat.obs["slice_name"].isin(['151669', '151670', '151671', '151672'])
new_batch_3 = adata_concat.obs["slice_name"].isin(['151507', '151508', '151509', '151510'])
adata_concat.obs["sample_name"] = list(sum(new_batch_1)*['Sample 1'])+list(sum(new_batch_2)*['Sample 2'])+list(sum(new_batch_3)*['Sample 3'])
adata_concat.obs["sample_name"] = adata_concat.obs["sample_name"].astype('category')
adata_concat.obs["batch_name"] = adata_concat.obs["sample_name"].copy()

## Concat the spatial network for multiple slices
adj_concat = np.asarray(adj_list[0].todense())
for batch_id in range(1,len(datasets)):
    adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))

adata_concat.uns['edgeList'] = np.nonzero(adj_concat)
import time
start_time = time.time()
adata_concat = STAligner.train_STAligner(adata_concat, verbose=True, knn_neigh=100, device=used_device)  # epochs = 1500,
end_time = time.time()
print(f"well time: {end_time - start_time} seconds")
edge_list = [[left, right] for left, right in zip(adata_concat.uns['edgeList'][0], adata_concat.uns['edgeList'][1])]
adata_concat.uns['edgeList'] = edge_list

STAligner.mclust_R(adata_concat, num_cluster=7, used_obsm='STAligner')
adata_concat = adata_concat[adata_concat.obs['Ground Truth']!='unknown']

from sklearn.metrics import adjusted_rand_score as ari_score
print('mclust, ARI = %01.3f' % ari_score(adata_concat.obs['Ground Truth'], adata_concat.obs['mclust']))
adata_concat.write('../results/12_staligner_DLPFC.h5ad')




import matplotlib.lines as mlines
import seaborn as sns
from sklearn import preprocessing
from matplotlib.colors import ListedColormap
import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import adjusted_rand_score
from scipy.optimize import linear_sum_assignment

import scanpy as sc
import pandas as pd
adata_raw = sc.read_h5ad('raw_adata_12.h5ad')
label_mapping = {
    '0': '1',
    '1': '2',
    '2': '3',
    '3': '4',
    '4': '5',
    '5': '6',
    '6': '7'
}
adata_raw.obs['mclust'] = adata_raw.obs['mclust'].astype(str).map(label_mapping)
df=pd.read_csv('../STAligner/results/12_dlpfc_model_performance_results.csv')
adata = sc.read_h5ad('../STAligner/results/12_staligner_DLPFC.h5ad')

fig, ax_list = plt.subplots(5, 7, figsize=(42, 30))  
ax_list = ax_list.flatten()
# ---- Part 1: UMAP / PAGA plots (2x4) ----
# Uncorrected UMAP and PAGA plots
ax_list[0].text(0.05, 1.05, 'A', ha='center', va='bottom', fontsize=32, fontweight='bold', transform=ax_list[0].transAxes)
sc.pp.neighbors(adata_raw, use_rep='X', random_state=22)
sc.tl.umap(adata_raw, random_state=22)
umap_titles = ['uncorrected', 'RAW-Ground Truth', 'RAW-clusters']
umap_colors = ['batch_name', 'celltype', 'mclust']
for i, title in enumerate(umap_titles):
    sc.pl.umap(adata_raw, color=umap_colors[i], title=title, ax=ax_list[i], show=False)

# PAGA plot for uncorrected data
sc.tl.paga(adata_raw, groups='celltype')  
sc.pl.paga(adata_raw, color='celltype', title='PAGA plot with Raw X', ax=ax_list[3], show=False)
spot_size=100
sc.pl.spatial (adata_raw,color='celltype',title='RAW spatial truth', ax=ax_list[4], spot_size=spot_size, cmap='viridis', show=False)
sc.pl.spatial (adata_raw,color='mclust',title='RAW spatial cluster', ax=ax_list[5], spot_size=spot_size, cmap='viridis', show=False)

ax_list[7].text(0.05, 1.05, 'B', ha='center', va='bottom', fontsize=32, fontweight='bold', transform=ax_list[7].transAxes)
sc.pp.neighbors(adata, use_rep='STAligner', random_state=22)
sc.tl.umap(adata, random_state=22)
umap_titles_corrected = ['corrected', 'Ground Truth', 'clusters']
umap_colors_corrected = ['batch_name', 'Ground Truth', 'mclust']
for i, title in enumerate(umap_titles_corrected):
    sc.pl.umap(adata, color=umap_colors_corrected[i], title=title, ax=ax_list[7 + i], show=False)

# PAGA plot for corrected data
sc.tl.paga(adata, groups='Ground Truth')  
sc.pl.paga(adata, color='Ground Truth', title='PAGA plot with STAligner', ax=ax_list[10], show=False)
spot_size=100
sc.pl.spatial (adata,color='Ground Truth',title='spatial truth', ax=ax_list[11], spot_size=spot_size, cmap='viridis', show=False)
sc.pl.spatial (adata,color='mclust',title='spatial cluster', ax=ax_list[12], spot_size=spot_size, cmap='viridis', show=False)

# ---- Part 2: Performance Metrics (1x4) ----
# Bar plot for model performance
colors = sns.color_palette("Set1", n_colors=df.shape[1] - 1)
df.set_index('Model').plot(kind='bar', ax=ax_list[6], width=0.6, color=colors)
ax_list[6].set_ylabel("Score", fontsize=16, color="black")
ax_list[6].tick_params(axis='x', rotation=45)
ax_list[6].tick_params(axis='y')
ax_list[6].legend(title="Metrics", bbox_to_anchor=(1.05, 1), loc='upper left')
ax_list[6].text(0.05, 1.05, 'C', ha='center', va='bottom', fontsize=32, fontweight='bold', transform=ax_list[6].transAxes)
# Box plot for metrics
metrics = ['graph_connectivity', 'iLISI','ASW']
df_melted = df.melt(id_vars="Model", value_vars=metrics, var_name="Metric", value_name="Value")
sns.boxplot(x="Model", y="Value", data=df_melted, palette="Set2", ax=ax_list[13])
mean_line = mlines.Line2D([], [], color='grey', linestyle='--', linewidth=1, label='Mean')    
kBET_marker = mlines.Line2D([], [], marker='o', markerfacecolor='white', markeredgewidth=1, color='grey', label='kBET') 
ax_list[13].legend(handles=[mean_line, kBET_marker], loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=16)
ax_list[13].set_ylabel('Value', fontsize=12, color="black")
ax_list[13].tick_params(axis='x', rotation=45)
ax_list[13].tick_params(axis='y')
ax_list[13].text(0.05, 1.05, 'D', ha='center', va='bottom', fontsize=32, fontweight='bold', transform=ax_list[13].transAxes)
for i, model in enumerate(df['Model']):
    kBET_value = df.loc[i, 'kBET']
    ax_list[13].plot(i, kBET_value, 'o', markerfacecolor='white', markeredgewidth=1, color="grey") 
    ax_list[13].text(i, kBET_value + 0.02, f'kBET: {kBET_value:.2f}', horizontalalignment='center', color='black', fontsize=12)  # 在圆点上方标记kBET值

for i, model in enumerate(df["Model"]):
    mean_value = df[df["Model"] == model][metrics].mean(axis=1).values[0]
    ax_list[13].plot([i - 0.4, i + 0.4], [mean_value, mean_value], color='grey', linestyle='--', linewidth=1)
    ax_list[13].text(i, mean_value - 0.05, f'{mean_value:.2f}', horizontalalignment='center', color='black', fontsize=12)

# ---- Part 3: Violin plots for gene expression (3x7) ----
from scipy import stats
genes = ['CXCL14', 'HPCAL1', 'CARTPT', 'PVALB', 'PCP4', 'KRT17', 'MBP']
for i, gene in enumerate(genes):
    sc.pl.violin(adata, keys=[gene], groupby='Ground_Truth', jitter=True, rotation=45, size=4, scale='width', ax=ax_list[14 + i])
    ax_list[14 + i].set_title(f'{gene}')

ax_list[14].text(0.05, 1.05, 'E', ha='center', va='bottom', fontsize=32, fontweight='bold', transform=ax_list[14].transAxes)
unique_mclust = adata.obs['mclust'].unique()
palette = sns.color_palette("Set1", n_colors=len(unique_mclust))
palette_dict = {str(mclust): color for mclust, color in zip(unique_mclust, palette)}

for i, gene in enumerate(genes):
    sc.pl.violin(adata, keys=[gene], groupby='mclust', jitter=True, rotation=45, size=4, scale='width', ax=ax_list[21 + i], palette=palette_dict)
    ax_list[21 + i].set_title(f'{gene}')

ax_list[21].text(0.05, 1.05, 'F', ha='center', va='bottom', fontsize=32, fontweight='bold', transform=ax_list[21].transAxes)
for i, gene in enumerate(genes):
    sc.pl.violin(adata_raw, keys=[gene], groupby='mclust', jitter=True, rotation=45, size=4, scale='width', ax=ax_list[28 + i])
    ax_list[28 + i].set_title(f'{gene}')

ax_list[28].text(0.05, 1.05, 'G', ha='center', va='bottom', fontsize=32, fontweight='bold', transform=ax_list[28].transAxes)
plt.tight_layout()

plt.savefig('../results/combined_figure.png')
print('Saved combined figure as combined_figure.png')
plt.savefig('../results/combined_figure.png')

######计算marker基因的显著性
genes = ['HPCAL1', 'PCP4']
layers = {'HPCAL1': 3, 'PCP4': 2}
p_values = {}
for gene in genes:
    target_layer = layers[gene]
    target_cells = adata.obs['mclust'] == target_layer
    other_cells = adata.obs['mclust'] != target_layer
    target_cells = target_cells.values if hasattr(target_cells, "values") else target_cells
    other_cells = other_cells.values if hasattr(other_cells, "values") else other_cells
    group1 = adata[:, gene].X[target_cells]
    group2 = adata[:, gene].X[other_cells]
    if hasattr(group1, "toarray"):
        group1 = group1.toarray().flatten()
        group2 = group2.toarray().flatten()
    else:
        group1 = np.array(group1).flatten()
        group2 = np.array(group2).flatten()
    if len(group1) == 0 or len(group2) == 0:
        print(f"Skipping {gene} due to empty groups.")
        continue
    _, p_value = stats.mannwhitneyu(group1, group2, alternative='greater')
    p_values[gene] = p_value

for gene, p in p_values.items():
    print(f"{gene}: Mann-Whitney U test p-value = {p:.4e}")
