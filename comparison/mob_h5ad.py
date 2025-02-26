import pandas as pd
import scanpy as sc

# 输入目录
input_dir="../data/mouse_olfactory_bulb/processed/35um/"
results_dir="../results/mob/"
datasets = ['BGI', 'SlideV2', '10X']

for dataset in datasets:
    feat_file = f"{input_dir}{dataset}_mat.csv"
    meta_file = f"{input_dir}{dataset}_meta.csv"
    coord_file = f"{input_dir}{dataset}_coord.csv"
    if dataset == '10X':
        edge_file = f"{input_dir}10X_edge_KNN_6.csv"
    elif dataset == 'SlideV2':
        edge_file = f"{input_dir}SlideV2_edge_KNN_8.csv"
    elif dataset == 'BGI':
        edge_file = f"{input_dir}BGI_edge_KNN_8.csv"
    data_matrix = pd.read_csv(feat_file, header=0, index_col=0)
    X = data_matrix.values
    print(f"Shape of data matrix: {data_matrix.shape}")  # (n_cells, n_genes)
    print(f"Number of genes: {data_matrix.columns.shape[0]}")  # 应该与 n_genes 一致
    adata = sc.AnnData(X, dtype='float32')
    # 将基因名称添加到 var 属性
    adata.var_names = data_matrix.columns.tolist()  # 设置基因名称
    adata.var = pd.DataFrame(index=data_matrix.columns)  # 创建 var DataFrame，并设置基因名称作为索引
    adata.var['gene_name'] = adata.var_names
    # 读取其他信息
    adata.obs = pd.read_csv(meta_file, header=0, index_col=0)
    adata.obs['batch'] = adata.obs['batch'].replace({0: 'BGI', 1: 'SlideV2', 2: '10X'})
    adata.obsm['spatial'] = pd.read_csv(coord_file, header=0, index_col=0).values
    edges = pd.read_csv(edge_file, header=0)
    adata.uns['edges'] = edges
    adata.write(f"{results_dir}{dataset}.h5ad")

import pandas as pd
import scanpy as sc

input_dir = "../data/mouse_olfactory_bulb/processed/35um/"
results_dir = "../results/mob/"
datasets = ['BGI', 'SlideV2', '10X']

# 使用字典映射数据集到边文件
edge_files = {
    '10X': f"{input_dir}10X_edge_KNN_6.csv",
    'SlideV2': f"{input_dir}SlideV2_edge_KNN_8.csv",
    'BGI': f"{input_dir}BGI_edge_KNN_8.csv"
}

for dataset in datasets:
    feat_file = f"{input_dir}{dataset}_mat.csv"
    meta_file = f"{input_dir}{dataset}_meta.csv"
    coord_file = f"{input_dir}{dataset}_coord.csv"
    
    data_matrix = pd.read_csv(feat_file, header=0, index_col=0)
    X = data_matrix.values
    print(f"Shape of data matrix: {data_matrix.shape}")  # (n_cells, n_genes)
    print(f"Number of genes: {data_matrix.columns.shape[0]}")  # 应该与 n_genes 一致
    
    adata = sc.AnnData(X, dtype='float32')
    adata.var_names = data_matrix.columns.tolist()  # 设置基因名称
    adata.var = pd.DataFrame(index=adata.var_names)  # 创建 var DataFrame，并设置基因名称作为索引
    adata.var['gene_name'] = adata.var_names
    
    adata.obs = pd.read_csv(meta_file, header=0, index_col=0)
    adata.obs['batch'] = adata.obs['batch'].replace({0: 'BGI', 1: 'SlideV2', 2: '10X'})
    adata.obsm['spatial'] = pd.read_csv(coord_file, header=0, index_col=0).values
    
    edges = pd.read_csv(edge_files[dataset], header=0)
    adata.uns['edges'] = edges
    
    adata.write(f"{results_dir}{dataset}.h5ad")