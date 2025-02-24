import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append('../DeepST')
import os 
from DeepST import run
import matplotlib.pyplot as plt
from pathlib import Path
import scanpy as sc
import anndata
import torch
from torch.utils.data import DataLoader
from scipy.sparse import issparse,csr_matrix
from sklearn.preprocessing import maxabs_scale, MaxAbsScaler
from torch.utils.data import TensorDataset
from pathlib import Path, PurePath
from typing import Optional, Union
from anndata import AnnData
import numpy as np
from PIL import Image
import pandas as pd
from _compat import Literal
import scanpy
import scipy

def read_h5ad_visium(path, 
                     load_images=True,
                     image_path='../../Visium_Mouse_Olfactory_Bulb/spatial/', 
                     scalefactors_path=None,  # 默认值为 None
                     quality='hires'):
    adata = sc.read_h5ad(path)
    adata.var_names_make_unique()
    library_id = "default_library"  # 设置默认库 ID
    adata.uns["spatial"] = {
        library_id: {
            "images": {},
            "scalefactors": {}
        }
    }
    print(adata.uns)
    if load_images and image_path is not None:
        img = plt.imread(image_path)  # 读取图像
        adata.uns["spatial"][library_id]["images"]["hires"] = img  # 存储 hires 图像
    if scalefactors_path is not None:
        with open(scalefactors_path) as f:
            scalefactors = json.load(f)
        library_id = list(adata.uns["spatial"].keys())[0] if "spatial" in adata.uns and adata.uns["spatial"] else 'default'
        if quality == "fulres":
            image_coor = adata.obsm["spatial"]
            img = plt.imread(image_path, 0)  # 替换为实际文件名
            adata.uns["spatial"].setdefault(library_id, {"images": {}})  # 确保库存在
            adata.uns["spatial"][library_id]["images"]["fulres"] = img
        else:
            scale = adata.uns["spatial"][library_id]["scalefactors"].get("tissue_" + quality + "_scalef", 1)
            image_coor = adata.obsm["spatial"] * scale
        adata.obs["imagecol"] = image_coor[:, 0]
        adata.obs["imagerow"] = image_coor[:, 1]
        adata.uns["spatial"][library_id]["use_quality"] = quality
    return adata

def read_stereoSeq(path,
                   bin_size=100,
                   is_sparse=True,
                   library_id=None,
                   scale=None,
                   quality="hires",
                   spot_diameter_fullres=1,
                   background_color="white"):
    from scipy import sparse
    adata = sc.read_h5ad(path)
    if "spatial" not in adata.obsm:
        raise ValueError(f"Spatial coordinates are missing from {path}. Please check the data format.")
    if scale is None:
        max_coor = np.max(adata.obsm["spatial"])
        scale = 20 / max_coor
    adata.obs["imagecol"] = adata.obsm["spatial"][:, 0] * scale
    adata.obs["imagerow"] = adata.obsm["spatial"][:, 1] * scale
    max_size = np.max([adata.obs["imagecol"].max(), adata.obs["imagerow"].max()])
    max_size = int(max_size + 0.1 * max_size)
    if background_color == "black":
        image = Image.new("RGB", (max_size, max_size), (0, 0, 0))
    else:
        image = Image.new("RGB", (max_size, max_size), (255, 255, 255))
    imgarr = np.array(image)
    if library_id is None:
        library_id = "StereoSeq"
    adata.uns["spatial"] = {}
    adata.uns["spatial"][library_id] = {}
    adata.uns["spatial"][library_id]["images"] = {}
    adata.uns["spatial"][library_id]["images"][quality] = imgarr
    adata.uns["spatial"][library_id]["use_quality"] = quality
    adata.uns["spatial"][library_id]["scalefactors"] = {}
    adata.uns["spatial"][library_id]["scalefactors"]["tissue_" + quality + "_scalef"] = scale
    adata.uns["spatial"][library_id]["scalefactors"]["spot_diameter_fullres"] = spot_diameter_fullres
    return adata

def read_SlideSeq(path, 
                 library_id=None,
                 scale=None,
                 quality="hires",
                 spot_diameter_fullres=50,
                 background_color="white"):
    adata = sc.read_h5ad(path)
    if scale is None:
        max_coor = np.max(adata.obsm["spatial"])
        scale = 20 / max_coor
    adata.obs["imagecol"] = adata.obsm["spatial"][:, 0] * scale
    adata.obs["imagerow"] = adata.obsm["spatial"][:, 1] * scale
    max_size = np.max([adata.obs["imagecol"].max(), adata.obs["imagerow"].max()])
    max_size = int(max_size + 0.1 * max_size)
    if background_color == "black":
        image = Image.new("RGB", (max_size, max_size), (0, 0, 0))
    else:
        image = Image.new("RGB", (max_size, max_size), (255, 255, 255))
    imgarr = np.array(image)
    if library_id is None:
        library_id = "Slide-seq"
    adata.uns["spatial"] = {}
    adata.uns["spatial"][library_id] = {}
    adata.uns["spatial"][library_id]["images"] = {}
    adata.uns["spatial"][library_id]["images"][quality] = imgarr
    adata.uns["spatial"][library_id]["use_quality"] = quality
    adata.uns["spatial"][library_id]["scalefactors"] = {}
    adata.uns["spatial"][library_id]["scalefactors"][
        "tissue_" + quality + "_scalef"] = scale
    adata.uns["spatial"][library_id]["scalefactors"][
        "spot_diameter_fullres"
    ] = spot_diameter_fullres
    return adata


save_path = "../Results" 
n_domains = 9
deepen = run(save_path = save_path,
	task = "Integration", 
	pre_epochs = 800, 
	epochs = 1000, 
	use_gpu = True)
#####10X
import json
visium_path = "../../RAW_SLICE/10X_float32.h5ad"
image_path = '../../Visium_Mouse_Olfactory_Bulb/spatial/tissue_hires_image.png' 
scalefactors_path='../../Visium_Mouse_Olfactory_Bulb/spatial/scalefactors_json.json'
visium_data = read_h5ad_visium(
    path=visium_path,        
    load_images=True,
    image_path=image_path,    
    scalefactors_path=scalefactors_path,  
    quality='hires'
)
visium_data.X = visium_data.X.toarray()  
visium_data = deepen._get_image_crop(visium_data, data_name=os.path.basename(visium_path))
visium_data = deepen._get_augment(visium_data, spatial_type="BallTree")
visium_graph = deepen._get_graph(visium_data.obsm["spatial"], distType="BallTree")

# BGI
bgi_path = "../../RAW_SLICE/BGI_float32.h5ad"
bgi_data = read_stereoSeq(bgi_path)
bgi_data = deepen._get_image_crop(bgi_data, data_name=os.path.basename(bgi_path))
bgi_data = deepen._get_augment(bgi_data, spatial_type="BallTree")
bgi_graph = deepen._get_graph(bgi_data.obsm["spatial"], distType="BallTree")

# SlideSeq
slide_seq_path = "../../RAW_SLICE/SlideV2_float32.h5ad"
slide_seq_data = read_SlideSeq(slide_seq_path)
slide_seq_data = deepen._get_image_crop(slide_seq_data, data_name=os.path.basename(slide_seq_path))
slide_seq_data = deepen._get_augment(slide_seq_data, spatial_type="BallTree",use_morphological=False)
slide_seq_graph = deepen._get_graph(slide_seq_data.obsm["spatial"], distType="BallTree")


# 添加基因
visium_data.var_names = visium_data.var.index
bgi_data.var_names = bgi_data.var.index
slide_seq_data.var_names = slide_seq_data.var.index

# Convert var_names to strings explicitly for all datasets
visium_data.var_names = visium_data.var_names.astype(str)
bgi_data.var_names = bgi_data.var_names.astype(str)
slide_seq_data.var_names = slide_seq_data.var_names.astype(str)

# Now, try concatenating the datasets again
multiple_adata, multiple_graph = deepen._get_multiple_adata(
    adata_list=[visium_data, bgi_data, slide_seq_data],
    data_name_list=['10X', 'BGI', 'SlideV2'],
    graph_list=[visium_graph, slide_seq_graph, bgi_graph]
)
data = deepen._data_process(multiple_adata, pca_n_comps=200)
deepst_embed = deepen._fit(
    data=data,
    graph_dict=multiple_graph,
    domains=multiple_adata.obs["batch"].values,  # Input to Domain Adversarial Model
    n_domains=3
)

multiple_adata.obsm["DeepST_embed"] = deepst_embed
multiple_adata = deepen._get_cluster_data(multiple_adata, n_domains=n_domains, priori=True)
###### Define the number of space domains, and the model can also be customized. If it is a model custom priori = False.
import anndata as ad 
multiple_adata.write("../Results/multiple_adata_mob.h5ad")
multiple_adata = ad.read_h5ad("../Results/multiple_adata_mob.h5ad")

#绘制批次校正后的umap图
fig, ax_list = plt.subplots(1, 3, figsize=(12, 4))
sc.pp.neighbors(multiple_adata, use_rep='DeepST_embed',random_state=666) 
sc.tl.umap(multiple_adata,random_state=666)
sc.pl.umap(multiple_adata, color='batch_name', ax=ax_list[0], title='Batch corrected', show=False)
sc.pl.umap(multiple_adata, color='DeepST_refine_domain', ax=ax_list[1], title='Colored by clusters', show=False)
# sc.pp.neighbors(multiple_adata, use_rep='X', n_neighbors=10, n_pcs=40,random_state=666)
# sc.tl.umap(multiple_adata,random_state=666)
# sc.pl.umap(multiple_adata, color='batch_name', title='Uncorrected', ax=ax_list[0], show=False)
plt.tight_layout(w_pad=0.05)
plt.savefig(os.path.join(save_path, '3mob_umap.pdf'), bbox_inches='tight', dpi=600)

#用scib进行基准测试
import scib
scib.me.graph_connectivity(multiple_adata, label_key="celltype")
scib.me.ilisi_graph(multiple_adata, batch_key="batch_name",type_="embed",use_rep="DeepST_embed")
scib.me.ilisi_graph(multiple_adata, batch_key="batch_name", type_="embed", use_rep="X_umap")
scib.me.ilisi_graph(multiple_adata, batch_key="batch_name",type_="KNN")
scib.me.kBET(multiple_adata, batch_key="batch_name", label_key="celltype", type_="embed", embed="DeepST_embed")
scib.me.silhouette_batch(multiple_adata, batch_key="batch_name", label_key="DeepST_refine_domain", embed="DeepST_embed")#绘制每个数据集的空间图
