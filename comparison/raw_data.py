import scanpy as sc
import pandas as pd
import os
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import anndata as ad
file_fold = '../RAW_SLICE/DLPFC/'
datasets = ['151673', '151674', '151675', '151676']
adatas = [] 
for dataset in datasets:   
    adata = sc.read_visium(
        os.path.join(file_fold, dataset),
        count_file=dataset + '_filtered_feature_bc_matrix.h5',
        load_images=True
    )
    adata.var_names_make_unique()
    Ann_df = pd.read_csv(
        os.path.join(file_fold, dataset, dataset + '_truth.txt'),
        sep='\t', header=None, index_col=0
    )
    Ann_df.columns = ['Ground Truth']
    Ann_df[Ann_df.isna()] = "unknown"
    Ann_df.index = [x + '_' + dataset for x in Ann_df.index]
    adata.obs_names = [x + '_' + dataset for x in adata.obs_names]
    adata.obs['celltype'] = Ann_df.loc[adata.obs_names, 'Ground Truth'].astype('category')
    adata = adata[adata.obs['celltype'] != 'unknown']
    coords = pd.DataFrame(adata.obsm["spatial"], index=adata.obs_names)
    adata.obsm["spatial"] = coords.values
    adata.obs['new_batch'] = dataset  
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var['highly_variable']]
    adatas.append(adata)

raw_adata = ad.concat(
    adatas,
    join='outer',         
    label='new_batch',     
    keys=datasets         
)
raw_adata = raw_adata[:, raw_adata.var['highly_variable']]

sc.tl.pca(raw_adata, n_comps=50)
X_pca = raw_adata.obsm["X_pca"]
gmm = GaussianMixture(n_components=7, random_state=42)
raw_adata.obs['mclust'] = gmm.fit_predict(X_pca)
raw_adata.obs["mclust"] = raw_adata.obs["mclust"].astype("category")
raw_adata.obs["new_batch"] = raw_adata.obs["new_batch"].astype("category")
print(raw_adata.obs["new_batch"].cat.categories)
raw_adata.write('../DATA_RAW/raw_adata1.h5ad')


datasets = ['151669', '151670','151671', '151672']
adatas=[]
for dataset in datasets:   
    adata = sc.read_visium(
        os.path.join(file_fold, dataset),
        count_file=dataset + '_filtered_feature_bc_matrix.h5',
        load_images=True
    )
    adata.var_names_make_unique()
    Ann_df = pd.read_csv(
        os.path.join(file_fold, dataset, dataset + '_truth.txt'),
        sep='\t', header=None, index_col=0
    )
    Ann_df.columns = ['Ground Truth']
    Ann_df[Ann_df.isna()] = "unknown"
    Ann_df.index = [x + '_' + dataset for x in Ann_df.index]
    adata.obs_names = [x + '_' + dataset for x in adata.obs_names]
    adata.obs['celltype'] = Ann_df.loc[adata.obs_names, 'Ground Truth'].astype('category')
    adata = adata[adata.obs['celltype'] != 'unknown']
    coords = pd.DataFrame(adata.obsm["spatial"], index=adata.obs_names)
    adata.obsm["spatial"] = coords.values
    adata.obs['new_batch'] = dataset  
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var['highly_variable']]
    adatas.append(adata)

raw_adata = ad.concat(
    adatas,
    join='outer',         
    label='new_batch',     
    keys=datasets         
)
sc.tl.pca(raw_adata, n_comps=50)
X_pca = raw_adata.obsm["X_pca"]
gmm = GaussianMixture(n_components=7, random_state=42)
raw_adata.obs['mclust'] = gmm.fit_predict(X_pca)
raw_adata.obs["mclust"] = raw_adata.obs["mclust"].astype("category")
raw_adata.obs["new_batch"] = raw_adata.obs["new_batch"].astype("category")
raw_adata.write('../DATA_RAW/raw_adata2.h5ad')


datasets = ['151507', '151508', '151509', '151510']
adatas=[]
for dataset in datasets:   
    adata = sc.read_visium(
        os.path.join(file_fold, dataset),
        count_file=dataset + '_filtered_feature_bc_matrix.h5',
        load_images=True
    )
    adata.var_names_make_unique()
    Ann_df = pd.read_csv(
        os.path.join(file_fold, dataset, dataset + '_truth.txt'),
        sep='\t', header=None, index_col=0
    )
    Ann_df.columns = ['Ground Truth']
    Ann_df[Ann_df.isna()] = "unknown"
    Ann_df.index = [x + '_' + dataset for x in Ann_df.index]
    adata.obs_names = [x + '_' + dataset for x in adata.obs_names]
    adata.obs['celltype'] = Ann_df.loc[adata.obs_names, 'Ground Truth'].astype('category')
    adata = adata[adata.obs['celltype'] != 'unknown']
    coords = pd.DataFrame(adata.obsm["spatial"], index=adata.obs_names)
    adata.obsm["spatial"] = coords.values
    adata.obs['new_batch'] = dataset  
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var['highly_variable']]
    adatas.append(adata)

raw_adata = ad.concat(
    adatas,
    join='outer',         
    label='new_batch',     
    keys=datasets         
)
sc.tl.pca(raw_adata, n_comps=50)
X_pca = raw_adata.obsm["X_pca"]
gmm = GaussianMixture(n_components=7, random_state=42)
raw_adata.obs['mclust'] = gmm.fit_predict(X_pca)
raw_adata.obs["mclust"] = raw_adata.obs["mclust"].astype("category")
raw_adata.obs["new_batch"] = raw_adata.obs["new_batch"].astype("category")
raw_adata.write('../DATA_RAW/raw_adata3.h5ad')


datasets=['151673','151669','151507']
adatas=[]
for dataset in datasets:   
    adata = sc.read_visium(
        os.path.join(file_fold, dataset),
        count_file=dataset + '_filtered_feature_bc_matrix.h5',
        load_images=True
    )
    adata.var_names_make_unique()
    Ann_df = pd.read_csv(
        os.path.join(file_fold, dataset, dataset + '_truth.txt'),
        sep='\t', header=None, index_col=0
    )
    Ann_df.columns = ['Ground Truth']
    Ann_df[Ann_df.isna()] = "unknown"
    Ann_df.index = [x + '_' + dataset for x in Ann_df.index]
    adata.obs_names = [x + '_' + dataset for x in adata.obs_names]
    adata.obs['celltype'] = Ann_df.loc[adata.obs_names, 'Ground Truth'].astype('category')
    adata = adata[adata.obs['celltype'] != 'unknown']
    coords = pd.DataFrame(adata.obsm["spatial"], index=adata.obs_names)
    adata.obsm["spatial"] = coords.values
    adata.obs['batch'] = dataset  
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var['highly_variable']]
    adatas.append(adata)

raw_adata = ad.concat(
    adatas,
    join='outer',         
    label='new_batch',     
    keys=datasets         
)
raw_adata.obs["new_batch"] = raw_adata.obs["batch"].astype("category")
new_batch_1 = raw_adata.obs["new_batch"].isin(['151673'])
new_batch_2 = raw_adata.obs["new_batch"].isin(['151669'])
new_batch_3 = raw_adata.obs["new_batch"].isin(['151507'])
raw_adata.obs["new_batch"] = list(sum(new_batch_1)*['Sample 1'])+list(sum(new_batch_2)*['Sample 2'])+list(sum(new_batch_3)*['Sample 3'])
sc.tl.pca(raw_adata, n_comps=50)
X_pca = raw_adata.obsm["X_pca"]
gmm = GaussianMixture(n_components=7, random_state=42)
raw_adata.obs['mclust'] = gmm.fit_predict(X_pca)
raw_adata.obs["mclust"] = raw_adata.obs["mclust"].astype("category")
raw_adata.write('../DATA_RAW/raw_adata_all.h5ad')


datasets = ['151673', '151674']
adatas=[]
for dataset in datasets:   
    adata = sc.read_visium(
        os.path.join(file_fold, dataset),
        count_file=dataset + '_filtered_feature_bc_matrix.h5',
        load_images=True
    )
    adata.var_names_make_unique()
    Ann_df = pd.read_csv(
        os.path.join(file_fold, dataset, dataset + '_truth.txt'),
        sep='\t', header=None, index_col=0
    )
    Ann_df.columns = ['Ground Truth']
    Ann_df[Ann_df.isna()] = "unknown"
    Ann_df.index = [x + '_' + dataset for x in Ann_df.index]
    adata.obs_names = [x + '_' + dataset for x in adata.obs_names]
    adata.obs['celltype'] = Ann_df.loc[adata.obs_names, 'Ground Truth'].astype('category')
    adata = adata[adata.obs['celltype'] != 'unknown']
    coords = pd.DataFrame(adata.obsm["spatial"], index=adata.obs_names)
    adata.obsm["spatial"] = coords.values
    adata.obs['new_batch'] = dataset  
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var['highly_variable']]
    adatas.append(adata)

raw_adata = ad.concat(
    adatas,
    join='outer',         
    label='new_batch',     
    keys=datasets         
)
sc.tl.pca(raw_adata, n_comps=50)
X_pca = raw_adata.obsm["X_pca"]
gmm = GaussianMixture(n_components=7, random_state=42)
raw_adata.obs['mclust'] = gmm.fit_predict(X_pca)
raw_adata.obs["mclust"] = raw_adata.obs["mclust"].astype("category")
raw_adata.obs["new_batch"] = raw_adata.obs["new_batch"].astype("category")
raw_adata.write('../DATA_RAW/raw_adata_7374.h5ad')




import scanpy as sc
import pandas as pd
import anndata as ad
import os
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix
datasets = ['10X', 'BGI','SlideV2' ]
file_fold = "../RAW_SLICE/mob/"
adatas = []
for dataset in datasets:
    adata = sc.read_h5ad(os.path.join(file_fold, dataset + '.h5ad'))
    adata.X = csr_matrix(adata.X)
    adata.var_names_make_unique()
    print('Before flitering: ', adata.shape)
    print('After flitering: ', adata.shape)
    adata.obs_names = [x+'_'+dataset for x in adata.obs_names]
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adatas.append(adata) 

raw_adata = ad.concat(adatas, label="new_batch", keys=datasets)
sc.tl.pca(raw_adata, n_comps=50)
X_pca = raw_adata.obsm["X_pca"]
gmm = GaussianMixture(n_components=10, random_state=42)
raw_adata.obs['mclust'] = gmm.fit_predict(X_pca)
raw_adata.obs["mclust"] = raw_adata.obs["mclust"].astype("category")
raw_adata.obs["new_batch"] = raw_adata.obs["new_batch"].astype("category")
raw_adata.write('../DATA_RAW/raw_adata_mob.h5ad')



datasets = ['section1', 'section2']
file_fold = '../RAW_SLICE/hbc/'
adatas = []

for dataset in datasets:   
    adata = sc.read_visium(file_fold + dataset, load_images=True)
    adata.var_names_make_unique()
    new_obs_names = [x.rsplit('-', 1)[0] + f"_{dataset}" for x in adata.obs_names]
    adata.obs_names = new_obs_names
    adata.obs['new_batch'] = dataset  
    adata.obs['celltype'] = 'Unknown'
    meta = pd.read_csv(os.path.join(file_fold, dataset, "metadata.csv"), index_col=0)
    meta.index = [idx.rsplit('-', 1)[0] + f"_{dataset}" for idx in meta.index]
    common_barcodes = adata.obs_names.intersection(meta.index)
    adata.obs.loc[common_barcodes, 'celltype'] = meta.loc[common_barcodes, 'celltype']
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var['highly_variable']]
    adatas.append(adata)

raw_adata = adatas[0].concatenate(*adatas[1:], batch_key='new_batch', batch_categories=datasets, index_unique=None)
spatial_coords = []
for ad in adatas:
    coords = pd.DataFrame(ad.obsm["spatial"], index=ad.obs_names)
    spatial_coords.append(coords)

spatial_coords = pd.concat(spatial_coords, axis=0)
spatial_coords = spatial_coords.reindex(raw_adata.obs_names)
raw_adata.obsm["spatial"] = spatial_coords.values
sc.tl.pca(raw_adata, n_comps=50, random_state=666)
X_pca = raw_adata.obsm["X_pca"]
gmm = GaussianMixture(n_components=8, random_state=42)
raw_adata.obs['mclust'] = gmm.fit_predict(X_pca)
raw_adata.obs["mclust"] = raw_adata.obs["mclust"].astype("category")
raw_adata.obs["new_batch"] = raw_adata.obs["new_batch"].astype("category")
print(raw_adata.obs.head())
raw_adata.write('../DATA_RAW/raw_adata_hbc.h5ad')


import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import anndata as ad
import os
import scanpy as sc
import pandas as pd
batch_name_map = {
    "FFPE": "FFPE",
    "DAPI": "DAPI",
    "Normal": "Normal"
}
file_fold = '../RAW_SLICE/coronal/'
datasets = ["FFPE", "DAPI", "Normal"]
adatas = []

for dataset in datasets:   
    adata = sc.read_visium(
        file_fold + dataset, 
        count_file="filtered_feature_bc_matrix.h5", 
        load_images=True
    )
    adata.var_names_make_unique()
    batch_name = batch_name_map[dataset]
    adata.obs["new_batch"] = batch_name
    adata.obs.index = batch_name + "-" + adata.obs.index
    adata.obs.index = adata.obs.index.str.replace(r"(-\d+)+$", "-1", regex=True)
    Ann_df = pd.read_csv(
        os.path.join(file_fold, dataset, dataset + "_truth.csv"),
        sep=",", header=0, index_col=0
    )
    Ann_df.index = Ann_df.index.astype(str)
    common_barcodes = adata.obs_names[adata.obs_names.isin(Ann_df.index)]
    cell_info_new = Ann_df.loc[common_barcodes, "celltype_new"]
    adata.obs["celltype"] = cell_info_new
    #adata = adata[adata.obs["celltype"] != "unknown"]
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var["highly_variable"]]
    adatas.append(adata)

raw_adata = adatas[0].concatenate(
    *adatas[1:], 
    batch_key="new_batch", 
    batch_categories=[batch_name_map[d] for d in datasets],
    index_unique=None)

import numpy as np
spatial_list = []
for i, adata in enumerate(adatas):
    spatial = adata.obsm["spatial"]
    spatial_list.append(spatial)

raw_adata.obsm["spatial"] = np.vstack(spatial_list)
raw_adata = raw_adata[~raw_adata.obs["celltype"].isna()].copy()
sc.tl.pca(raw_adata, n_comps=50)
X_pca = raw_adata.obsm["X_pca"]
gmm = GaussianMixture(n_components=12, random_state=42)
raw_adata.obs['mclust'] = gmm.fit_predict(X_pca)
raw_adata.obs["mclust"] = raw_adata.obs["mclust"].astype("category")
raw_adata.write('../DATA_RAW/raw_adata_coronal.h5ad')
