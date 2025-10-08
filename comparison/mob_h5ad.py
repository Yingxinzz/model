##cd BE/DATA_RAW
import pandas as pd
import scanpy as sc

input_dir = "../SPIRAL/data/mouse_olfactory_bulb/processed/35um/"
results_dir = "../RAW_SLICE/mob/"
datasets = ['BGI', 'SlideV2', '10X']
celltype_mapping = {
    'BGI': {
        'ONL': 'ONL',
        'GL': 'GL',           
        'GCL': 'GCL',         
        'MCL': 'MCL',
        'OPL': 'OPL',         
        'SEZ': 'SEZ',
        'Meninges': 'Meninges', 
        'Low_Quality': None   
    },
    
    'SlideV2': {
        'ONL': 'ONL',
        'GL': 'GL',            
        'GCL': 'GCL',         
        'MCL': 'MCL',
        'EPL': 'EPL',         
        'SEZ': 'SEZ'
    },
    
    '10X': {
        'ONL': 'ONL',
        'GL_1': 'GL',         
        'GL_2': 'GL',         
        'GCL_1': 'GCL',       
        'GCL_2': 'GCL',       
        'MCL': 'MCL',
        'EPL': 'EPL'         
    }
}
import numpy as np
for dataset in datasets:
    print(f"Processing {dataset}...")
    feat_file = f"{input_dir}{dataset}_mat.csv"
    meta_file = f"{input_dir}{dataset}_meta.csv"
    coord_file = f"{input_dir}{dataset}_coord.csv"
    data_matrix = pd.read_csv(feat_file, header=0, index_col=0)
    meta_df = pd.read_csv(meta_file, header=0, index_col=0)
    coord_df = pd.read_csv(coord_file, header=0, index_col=0)
    adata = sc.AnnData(
        X=data_matrix.values.astype(np.float32),
        obs=meta_df,
        var=pd.DataFrame(index=data_matrix.columns)
    )
    adata.var_names_make_unique()
    adata.obsm["spatial"] = coord_df.values
    adata.layers["counts"] = data_matrix.values.astype(np.float32)
    adata.raw = adata
    if dataset in celltype_mapping:
        mapping = celltype_mapping[dataset]
        if mapping:
            adata.obs['celltype'] = adata.obs['celltype'].replace(mapping)
            valid_cells = ~adata.obs['celltype'].isna()
            adata = adata[valid_cells].copy()
    adata.obs['dataset'] = dataset
    adata.obs['batch'] = adata.obs['batch'].replace({0: 'BGI', 1: 'SlideV2', 2: '10X'})
    if dataset == '10X':
        edge_file = f"{input_dir}10X_edge_KNN_6.csv"
    elif dataset == 'SlideV2':
        edge_file = f"{input_dir}SlideV2_edge_KNN_8.csv"
    else:  # BGI
        edge_file = f"{input_dir}BGI_edge_KNN_8.csv"
    edges = pd.read_csv(edge_file, header=None, sep=r'\s+')
    if edges.shape[1] == 2:
        edges.columns = ["source", "target"]
    else:
        raise ValueError
    adata.uns['edges'] = edges
    import scipy.sparse as sp
    n_cells = adata.n_obs
    mat = sp.lil_matrix((n_cells, n_cells))
    id_to_idx = {cell_id: idx for idx, cell_id in enumerate(adata.obs_names)}
    for s, t in zip(edges["source"], edges["target"]):
        if s in id_to_idx and t in id_to_idx:
            i, j = id_to_idx[s], id_to_idx[t]
            mat[i, j] = 1
            mat[j, i] = 1
    adata.obsp["connectivities"] = mat.tocsr()
    adata.obsp["distances"] = mat.tocsr()  
    adata.uns["spatial"] = {dataset: {"metadata": {}}}
    adata.write(f"{results_dir}{dataset}.h5ad")

print("All datasets processed!")



# for dataset in datasets:
#     print(f"Processing {dataset}...")
    
#     feat_file = f"{input_dir}{dataset}_mat.csv"
#     meta_file = f"{input_dir}{dataset}_meta.csv"
#     coord_file = f"{input_dir}{dataset}_coord.csv"
#     data_matrix = pd.read_csv(feat_file, header=0, index_col=0)
#     data_matrix = pd.read_csv(feat_file, header=0, index_col=0)
#     meta_df = pd.read_csv(meta_file, header=0, index_col=0)
#     coord_df = pd.read_csv(coord_file, header=0, index_col=0)
#     adata = sc.AnnData(
#         X=data_matrix.values,
#         obs=pd.read_csv(meta_file, header=0, index_col=0),
#         var=pd.DataFrame(index=data_matrix.columns),
#         dtype="float32"
#     )
#     adata.var_names_make_unique()
#     adata.obsm["spatial"] = coord_df.values
    
#     if dataset == '10X':
#         edge_file = f"{input_dir}10X_edge_KNN_6.csv"
#     elif dataset == 'SlideV2':
#         edge_file = f"{input_dir}SlideV2_edge_KNN_8.csv"
#     else:  # BGI
#         edge_file = f"{input_dir}BGI_edge_KNN_8.csv"
    
#     adata.uns['edges'] = pd.read_csv(edge_file, header=0)
    
#     if dataset in celltype_mapping:
#         mapping = celltype_mapping[dataset]
#         if mapping:  
#             adata.obs['celltype'] = adata.obs['celltype'].replace(mapping)
#             valid_cells = ~adata.obs['celltype'].isna()
#             adata = adata[valid_cells].copy()
    
#     adata.obs['dataset'] = dataset
#     adata.obs['batch'] = adata.obs['batch'].replace({0: 'BGI', 1: 'SlideV2', 2: '10X'})
    
#     adata.write(f"{results_dir}{dataset}.h5ad")
#     print(f"  Saved {adata.n_obs} cells with {len(adata.obs['celltype'].unique())} cell types")


