import scanpy as sc
import pandas as pd
import os
file_fold = '../RAW_SLICE/DLPFC/'
datasets = ['151673', '151674', '151675', '151676']
adatas=[]
for dataset in datasets:   
    adata = sc.read_visium(file_fold+dataset, count_file=dataset+'_filtered_feature_bc_matrix.h5', load_images=True)
    adata.var_names_make_unique()
    # read the annotation
    Ann_df = pd.read_csv(os.path.join(file_fold+dataset, dataset + '_truth.txt'), sep='\t', header=None, index_col=0)
    Ann_df.columns = ['Ground Truth']
    Ann_df[Ann_df.isna()] = "unknown"
    adata.obs['celltype'] = Ann_df.loc[adata.obs_names, 'Ground Truth'].astype('category')
    adata = adata[adata.obs['celltype']!='unknown']
    adata.obs_names = [x+'_'+dataset for x in adata.obs_names]
    adata.obs['batch'] = dataset  # Add batch information
    # Normalization
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var['highly_variable']]
    adatas.append(adata)

raw_adata = adatas[0].concatenate(*adatas[1:], batch_key='batch', batch_categories=datasets)
import scanpy as sc
from sklearn.mixture import GaussianMixture
sc.pp.scale(raw_adata)
X = raw_adata.X
n_components = 7 
gmm = GaussianMixture(n_components=n_components, random_state=42)
raw_adata.obs['mclust'] = gmm.fit_predict(X)
raw_adata.obs["mclust"] = raw_adata.obs["mclust"].astype("category")
raw_adata.write('raw_adata1.h5ad')

adatas=[]
datasets = ['151669', '151670','151671', '151672']
for dataset in datasets:   
    adata = sc.read_visium(file_fold+dataset, count_file=dataset+'_filtered_feature_bc_matrix.h5', load_images=True)
    adata.var_names_make_unique()
    # read the annotation
    Ann_df = pd.read_csv(os.path.join(file_fold+dataset, dataset + '_truth.txt'), sep='\t', header=None, index_col=0)
    Ann_df.columns = ['Ground Truth']
    Ann_df[Ann_df.isna()] = "unknown"
    adata.obs['celltype'] = Ann_df.loc[adata.obs_names, 'Ground Truth'].astype('category')
    adata = adata[adata.obs['celltype']!='unknown']
    adata.obs_names = [x+'_'+dataset for x in adata.obs_names]
    adata.obs['batch'] = dataset  # Add batch information
    # Normalization
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var['highly_variable']]
    adatas.append(adata)

raw_adata = adatas[0].concatenate(*adatas[1:], batch_key='batch', batch_categories=datasets)
import scanpy as sc
from sklearn.mixture import GaussianMixture
sc.pp.scale(raw_adata)
X = raw_adata.X
n_components = 5
gmm = GaussianMixture(n_components=n_components, random_state=42)
raw_adata.obs['mclust'] = gmm.fit_predict(X)
raw_adata.obs["mclust"] = raw_adata.obs["mclust"].astype("category")
raw_adata.write('raw_adata2.h5ad')

adatas=[]
datasets = ['151507', '151508', '151509', '151510']
for dataset in datasets:   
    adata = sc.read_visium(file_fold+dataset, count_file=dataset+'_filtered_feature_bc_matrix.h5', load_images=True)
    adata.var_names_make_unique()
    # read the annotation
    Ann_df = pd.read_csv(os.path.join(file_fold+dataset, dataset + '_truth.txt'), sep='\t', header=None, index_col=0)
    Ann_df.columns = ['Ground Truth']
    Ann_df[Ann_df.isna()] = "unknown"
    adata.obs['celltype'] = Ann_df.loc[adata.obs_names, 'Ground Truth'].astype('category')
    adata = adata[adata.obs['celltype']!='unknown']
    adata.obs_names = [x+'_'+dataset for x in adata.obs_names]
    adata.obs['batch'] = dataset  # Add batch information
    # Normalization
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var['highly_variable']]
    adatas.append(adata)

raw_adata = adatas[0].concatenate(*adatas[1:], batch_key='batch', batch_categories=datasets)
import scanpy as sc
from sklearn.mixture import GaussianMixture
sc.pp.scale(raw_adata)
X = raw_adata.X
n_components = 7 
gmm = GaussianMixture(n_components=n_components, random_state=42)
raw_adata.obs['mclust'] = gmm.fit_predict(X)
raw_adata.obs["mclust"] = raw_adata.obs["mclust"].astype("category")
raw_adata.write('raw_adata3.h5ad')


datasets=['151673','151669','151507']
adatas=[]
for dataset in datasets:   
    adata = sc.read_visium(file_fold+dataset, count_file=dataset+'_filtered_feature_bc_matrix.h5', load_images=True)
    adata.var_names_make_unique()
    # read the annotation
    Ann_df = pd.read_csv(os.path.join(file_fold+dataset, dataset + '_truth.txt'), sep='\t', header=None, index_col=0)
    Ann_df.columns = ['Ground Truth']
    Ann_df[Ann_df.isna()] = "unknown"
    adata.obs['celltype'] = Ann_df.loc[adata.obs_names, 'Ground Truth'].astype('category')
    adata = adata[adata.obs['celltype']!='unknown']
    adata.obs_names = [x+'_'+dataset for x in adata.obs_names]
    adata.obs['batch'] = dataset  # Add batch information
    # Normalization
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var['highly_variable']]
    adatas.append(adata)

raw_adata = adatas[0].concatenate(*adatas[1:], batch_key='batch', batch_categories=datasets)
raw_adata.obs["new_batch"] = raw_adata.obs["batch"]  
new_batch_1 = raw_adata.obs["new_batch"].isin(['151673'])
new_batch_2 = raw_adata.obs["new_batch"].isin(['151669'])
new_batch_3 = raw_adata.obs["new_batch"].isin(['151507'])
raw_adata.obs["batch_name"] = list(sum(new_batch_1)*['Sample 1'])+list(sum(new_batch_2)*['Sample 2'])+list(sum(new_batch_3)*['Sample 3'])
import scanpy as sc
from sklearn.mixture import GaussianMixture
sc.pp.scale(raw_adata)
X = raw_adata.X
n_components = 7 
gmm = GaussianMixture(n_components=n_components, random_state=42)
raw_adata.obs['mclust'] = gmm.fit_predict(X)
raw_adata.obs["mclust"] = raw_adata.obs["mclust"].astype("category")
raw_adata.write('raw_adata_all.h5ad')

import scanpy as sc
import pandas as pd
import os
file_fold = '../RAW_SLICE/DLPFC/'
datasets = ['151673', '151674']
adatas=[]
for dataset in datasets:   
    adata = sc.read_visium(file_fold+dataset, count_file=dataset+'_filtered_feature_bc_matrix.h5', load_images=True)
    adata.var_names_make_unique()
    # read the annotation
    Ann_df = pd.read_csv(os.path.join(file_fold+dataset, dataset + '_truth.txt'), sep='\t', header=None, index_col=0)
    Ann_df.columns = ['Ground Truth']
    Ann_df[Ann_df.isna()] = "unknown"
    adata.obs['ground_truth'] = Ann_df.loc[adata.obs_names, 'Ground Truth'].astype('category')
    adata = adata[adata.obs['ground_truth']!='unknown']
    adata.obs_names = [x+'_'+dataset for x in adata.obs_names]
    adata.obs['new_batch'] = dataset  # Add batch information
    # Normalization
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var['highly_variable']]
    adatas.append(adata)

raw_adata = adatas[0].concatenate(*adatas[1:], batch_key='new_batch', batch_categories=datasets)
import scanpy as sc
from sklearn.mixture import GaussianMixture
sc.pp.scale(raw_adata)
X = raw_adata.X
n_components = 7 
gmm = GaussianMixture(n_components=n_components, random_state=42)
raw_adata.obs['mclust'] = gmm.fit_predict(X)
raw_adata.obs["mclust"] = raw_adata.obs["mclust"].astype("category")
raw_adata.write('raw_adata_7374.h5ad')




import anndata as ad
from scipy.sparse import csr_matrix
datasets = ['10X', 'BGI','SlideV2' ]
file_fold = "../RAW_SLICE/"
adatas = []
for dataset in datasets:
    adata = sc.read_h5ad(os.path.join('../RAW_SLICE', dataset + '.h5ad'))
    adata.X = csr_matrix(adata.X)
    adata.var_names_make_unique()
    print('Before flitering: ', adata.shape)
    print('After flitering: ', adata.shape)
    adata.obs_names = [x+'_'+dataset for x in adata.obs_names]
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adatas.append(adata) 

raw_adata = ad.concat(adatas, label="slice_name", keys=datasets)
import scanpy as sc
from sklearn.mixture import GaussianMixture
sc.pp.scale(raw_adata)
sc.tl.pca(raw_adata, n_comps=100, random_state=666)
X = raw_adata.obsm["X_pca"]
n_components = 9
gmm = GaussianMixture(n_components=n_components, random_state=42)
raw_adata.obs['mclust'] = gmm.fit_predict(X)
raw_adata.obs["mclust"] = raw_adata.obs["mclust"].astype("category")
raw_adata.write('raw_adata_mob.h5ad')

import scanpy as sc
import pandas as pd
import os
from sklearn.mixture import GaussianMixture
datasets = ['section1', 'section2']
file_fold = '../RAW_SLICE/hbc/'
adatas = []
for dataset in datasets:   
    adata = sc.read_visium(file_fold + dataset, load_images=True)
    adata.var_names_make_unique()
    adata.obs_names = [x + '_' + dataset for x in adata.obs_names]
    adata.obs['batch'] = dataset  
    # 数据标准化
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var['highly_variable']]
    adatas.append(adata)

raw_adata = adatas[0].concatenate(*adatas[1:], batch_key='batch', batch_categories=datasets)
from sklearn.decomposition import PCA
sc.pp.scale(raw_adata)
sc.tl.pca(raw_adata, n_comps=150, random_state=666)
X = raw_adata.obsm["X_pca"]
gmm = GaussianMixture(n_components=14, random_state=666)
raw_adata.obs['mclust'] = gmm.fit_predict(X)
print(raw_adata.obs.head())
raw_adata.obs["mclust"] = raw_adata.obs["mclust"].astype("category")
raw_adata.write('raw_adata_bc.h5ad')


import scanpy as sc
import pandas as pd
import os
file_fold = '../RAW_SLICE/coronal/'
datasets=['FFPE', 'DAPI', 'Normal']
adatas=[]
for dataset in datasets:   
    adata = sc.read_visium(file_fold+dataset, count_file='filtered_feature_bc_matrix.h5', load_images=True)
    adata.var_names_make_unique()
    # read the annotation
    Ann_df = pd.read_csv(os.path.join(file_fold, dataset, dataset + '_truth.csv'), sep=',', header=0, index_col=0)
    Ann_df.index = Ann_df.index.str.replace(r'^[A-Za-z]+-', '', regex=True)
    Ann_df.index = Ann_df.index.str.strip()  # 去掉空格
    missing_barcodes = set(adata.obs_names) - set(Ann_df.index)
    if missing_barcodes:
        print(f"Missing barcodes: {missing_barcodes}") 
    adata.obs_names = adata.obs_names.str.strip()  
    Ann_df.index = Ann_df.index.str.strip()
    common_barcodes = adata.obs_names[adata.obs_names.isin(Ann_df.index)]
    cell_info_new = Ann_df.loc[common_barcodes, 'celltype_new']
    adata.obs['ground_truth'] = cell_info_new
    adata = adata[adata.obs['ground_truth'] != 'unknown']
    adata.obs_names = [x+'_'+dataset for x in adata.obs_names]
    adata.obs['batch'] = dataset  # Add batch information
    # Normalization
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var['highly_variable']]
    adatas.append(adata)

raw_adata = adatas[0].concatenate(*adatas[1:], batch_key='batch', batch_categories=datasets)
import scanpy as sc
from sklearn.mixture import GaussianMixture
sc.pp.scale(raw_adata)
X = raw_adata.X
n_components = 12 
gmm = GaussianMixture(n_components=n_components, random_state=42)
raw_adata.obs['mclust'] = gmm.fit_predict(X)
raw_adata.obs["mclust"] = raw_adata.obs["mclust"].astype("category")
raw_adata.write('raw_adata_coronal.h5ad')
