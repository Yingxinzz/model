import os
import numpy as np
import argparse

import pandas as pd
from sklearn.decomposition import PCA
from operator import itemgetter
import random
import matplotlib.pyplot as plt
import umap.umap_ as umap
import time
import torch
import sys
sys.path.append('../')
from spiral.main import SPIRAL_integration
from spiral.layers import *
from spiral.utils import *
from spiral.CoordAlignment import CoordAlignment
from warnings import filterwarnings
filterwarnings("ignore")


sample=['section1', 'section2']
Batch_list = []
file_fold = '../../RAW_SLICE/hbc/'
meta_df = pd.read_csv("../../RAW_SLICE/hbc/section1/meta.csv") 
Batch_list = []
for i in np.arange(len(sample)):
    adata = sc.read_visium(file_fold + sample[i], load_images=True)
    adata.obs_names = [x + '_' + sample[i] for x in adata.obs_names]
    meta_file = f"{file_fold}{sample[i]}/meta.csv" 
    metadata = pd.read_csv(meta_file, index_col=0)
    adata.obs = adata.obs.join(metadata)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
    adata = adata[:, adata.var['highly_variable']]
    adata.obs_names_make_unique()  
    adata.var_names_make_unique() 
    Batch_list.append(adata)

adata_concat = sc.concat(Batch_list, label="new_batch", keys=sample)
vf=np.array(adata_concat.var.index)
for i in np.arange(len(sample)):
    adata1=adata_concat[adata_concat.obs['new_batch']==sample[i],:]
    cells=adata1.obs_names
    mat=pd.DataFrame(adata1.X.todense(),index=cells,columns=vf)
    coord=pd.DataFrame(adata1.obsm['spatial'],index=cells,columns=['x','y'])
    meta = adata1.obs.copy()
    meta['batch'] = meta['new_batch']
    meta.to_csv("../results/"+sample[i]+"_meta.csv")
    mat.to_csv("../results/"+sample[i]+"_mat.csv")
    coord.to_csv("../results/"+sample[i]+"_coord.csv")

import os
import scanpy as sc
import pandas as pd
import numpy as np
import sklearn.neighbors
import scipy.sparse as sp
from sklearn.decomposition import PCA

def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True):
    """\
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff.
        When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.

    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """
    assert(model in ['Radius', 'KNN'])
    
    if verbose:
        print('------Calculating spatial graph...')
    
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    
    if model == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it] * indices[it].shape[0], indices[it], distances[it])))
    
    if model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff + 1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))
    
    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']
    
    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance'] > 0, :]
    
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index)))
    
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    
    if verbose:
        print('The graph contains %d edges, %d cells.' % (Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' % (Spatial_Net.shape[0] / adata.n_obs))
    adata.uns['Spatial_Net'] = Spatial_Net

knn=6
rad=10
dirs="../results/"
for i in np.arange(len(sample)):
    feat=pd.read_csv(dirs + sample[i]+"_mat.csv",header=0,index_col=0,sep=',')
    coord=pd.read_csv(dirs+ sample[i]+"_coord.csv",header=0,index_col=0,sep=',')
    coord.columns=['x','y']
    adata = sc.AnnData(feat)
    adata.var_names_make_unique()
    adata.X=sp.csr_matrix(adata.X)
    adata.obsm["spatial"] = coord.loc[:,['x','y']].to_numpy()
    Cal_Spatial_Net(adata, rad_cutoff=rad, k_cutoff=knn, model='KNN', verbose=True)
    if 'highly_variable' in adata.var.columns:
        adata_Vars =  adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata
    features = pd.DataFrame(adata_Vars.X.toarray()[:, ], index=adata_Vars.obs.index, columns=adata_Vars.var.index)
    cells = np.array(features.index)
    cells_id_tran  = dict(zip(cells, range(cells.shape[0])))
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")
    Spatial_Net = adata.uns['Spatial_Net']
    G_df = Spatial_Net.copy()
    np.savetxt(dirs+sample[i]+"_edge_KNN_"+str(knn)+".csv",G_df.values[:,:2],fmt='%s')





R_dirs="/root/anaconda3/envs/SS/lib/R"
os.environ['R_HOME']=R_dirs
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
input_dir = "../results"
results_dir = "../results/bc/"
datasets = np.array(['section1', 'section2'])
SEP = ','
net_cate = '_KNN_'
rad = 150
knn = 6
N_WALKS = knn
WALK_LEN = 1
N_WALK_LEN = knn
NUM_NEG = knn
feat_file = []
meta_file = []
edge_file = []
coord_file = []
for i in range(len(datasets)):
    feat_file.append(input_dir + '/' + datasets[i] + "_mat.csv")
    meta_file.append(input_dir + '/' + datasets[i] + "_meta.csv")
    edge_file.append(input_dir + '/' + datasets[i] + f"_edge_KNN_{knn}.csv")  # Using f-string for formatting
    coord_file.append(input_dir + '/' + datasets[i] + "_coord.csv")

N = pd.read_csv(feat_file[0], header=0, index_col=0).shape[1]
M = 1 if len(datasets) == 2 else len(datasets)

# Debug print to verify the paths
print("Feature files:", feat_file)
print("Edge files:", edge_file)
print("Coordinate files:", coord_file)

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help='The seed of initialization.')
parser.add_argument('--AEdims', type=list, default=[N,[512],32], help='Dim of encoder.')
parser.add_argument('--AEdimsR', type=list, default=[32,[512],N], help='Dim of decoder.')
parser.add_argument('--GSdims', type=list, default=[512,32], help='Dim of GraphSAGE.')
parser.add_argument('--zdim', type=int, default=32, help='Dim of embedding.')
parser.add_argument('--znoise_dim', type=int, default=4, help='Dim of noise embedding.')
parser.add_argument('--CLdims', type=list, default=[4,[],M], help='Dim of classifier.')
parser.add_argument('--DIdims', type=list, default=[28,[32,16],M], help='Dim of discriminator.')
parser.add_argument('--beta', type=float, default=1.0, help='weight of GraphSAGE.')
parser.add_argument('--agg_class', type=str, default=MeanAggregator, help='Function of aggregator.')
parser.add_argument('--num_samples', type=str, default=knn, help='number of neighbors to sample.')

parser.add_argument('--N_WALKS', type=int, default=N_WALKS, help='number of walks of random work for postive pairs.')
parser.add_argument('--WALK_LEN', type=int, default=WALK_LEN, help='walk length of random work for postive pairs.')
parser.add_argument('--N_WALK_LEN', type=int, default=N_WALK_LEN, help='number of walks of random work for negative pairs.')
parser.add_argument('--NUM_NEG', type=int, default=NUM_NEG, help='number of negative pairs.')

parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=1024, help='Size of batches to train.') ####512 for withon donor;1024 for across donor###
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay.')
parser.add_argument('--alpha1', type=float, default=N, help='Weight of decoder loss.')
parser.add_argument('--alpha2', type=float, default=1, help='Weight of GraphSAGE loss.')
parser.add_argument('--alpha3', type=float, default=1, help='Weight of classifier loss.')
parser.add_argument('--alpha4', type=float, default=1, help='Weight of discriminator loss.')
parser.add_argument('--lamda', type=float, default=1, help='Weight of GRL.')
parser.add_argument('--Q', type=float, default=10, help='Weight negative loss for sage losss.')

params,unknown=parser.parse_known_args()
SPII=SPIRAL_integration(params,feat_file,edge_file,meta_file)
SPII.train()
if not os.path.exists(results_dir+"model/"):
     os.makedirs(results_dir+"model/")

model_file = os.path.join(
    results_dir,
    "model",
    f"SPIRAL_embed_{SPII.params.batch_size}_{'_'.join(map(str, datasets))}.pt"
)
torch.save(SPII.model.state_dict(),model_file)

# #加载模型（如果后面出现错误）
# #Create a new instance of the model
# model = SPII.model
# #Load the state dictionary into the model
# model_file=(results_dir+"model/SPIRAL_embed_1024_section1_section2.pt")
# model.load_state_dict(torch.load(model_file))
# #Set the model to evaluation mode if you are not continuing training
# model.eval()

SPII.model.eval()
all_idx=np.arange(SPII.feat.shape[0])
all_layer,all_mapping=layer_map(all_idx.tolist(),SPII.adj,len(SPII.params.GSdims))
all_rows=SPII.adj.tolil().rows[all_layer[0]]
all_feature=torch.Tensor(SPII.feat.iloc[all_layer[0],:].values).float().cuda()
all_embed,ae_out,clas_out,disc_out=SPII.model(all_feature,all_layer,all_mapping,all_rows,SPII.params.lamda,SPII.de_act,SPII.cl_act)
[ae_embed,gs_embed,embed]=all_embed
[x_bar,x]=ae_out
embed=embed.cpu().detach()
names=['GTT_'+str(i) for i in range(embed.shape[1])]
embed1=pd.DataFrame(np.array(embed),index=SPII.feat.index,columns=names)
if not os.path.exists(results_dir+"gtt_output/"):
    os.makedirs(results_dir+"gtt_output/")

embed_file = os.path.join(results_dir,"gtt_output",f"SPIRAL_embed_{SPII.params.batch_size}_{'_'.join(map(str, datasets))}.csv")
embed1.to_csv(embed_file)
meta=SPII.meta.values
embed_df = pd.DataFrame(embed.cpu().detach().numpy(), index=SPII.feat.index)
znoise_dim = SPII.params.znoise_dim
embed_new_df = pd.concat([
    pd.DataFrame(np.zeros((embed_df.shape[0], znoise_dim)), index=embed_df.index),
    embed_df.iloc[:, znoise_dim:]
], axis=1)
embed_new = torch.tensor(embed_new_df.values).float().cuda()
xbar_new=np.array(SPII.model.agc.ae.de(embed_new.cuda(),nn.Sigmoid())[1].cpu().detach())
xbar_new1=pd.DataFrame(xbar_new,index=SPII.feat.index,columns=SPII.feat.columns)
xbar_new1.to_csv(os.path.join(results_dir,"gtt_output",f"SPIRAL_correct_{SPII.params.batch_size}_{'_'.join(map(str, datasets))}.csv"))
meta=SPII.meta.values

import umap.umap_ as umap
import matplotlib.pyplot as plt
# x=PCA(n_components=30).fit_transform(SPII.feat)
# embed=pd.read_csv(dirs+"gtt_output/AGC_embed"+flags+"_xent.csv",index_col=0,header=0).values
#############cluster
import anndata
import scanpy as sc
ann=anndata.AnnData(SPII.feat)
ann.obsm['spiral']=embed1.iloc[:,SPII.params.znoise_dim:].values
sc.pp.neighbors(ann,use_rep='spiral')
n_clust=14
ann = mclust_R(ann, used_obsm='spiral', num_cluster=n_clust)
ann.X=SPII.feat
ann.obs['batch']=SPII.meta.loc[:,'batch'].values
ub=np.unique(ann.obs['batch'])
sc.tl.umap(ann)
coord=pd.read_csv(coord_file[0],header=0,index_col=0)
for i in np.arange(1,len(datasets)):
    coord=pd.concat((coord,pd.read_csv(coord_file[i],header=0,index_col=0)))

coord.columns=['y','x']
ann.obsm['spatial']=coord.loc[ann.obs_names,:].values

ann.obs["new_batch"] = ann.obs["batch"].astype(str)
ann.write('../results/spiral_bc.h5ad')
ann = sc.read_h5ad('../results/spiral_bc.h5ad')

f, axs = plt.subplots(1, 3, figsize=(15, 5))
sc.tl.pca(adata_SPIRAL, n_comps=100, random_state=666)
sc.pp.neighbors(adata_SPIRAL,use_rep='X',random_state=666)
sc.tl.umap(adata_SPIRAL,random_state=666)
sc.pl.umap(adata_SPIRAL, color='new_batch', ax=axs[0], title='uncorrect', show=False)
sc.pp.neighbors(adata_SPIRAL,use_rep='spiral',random_state=666)
sc.tl.umap(adata_SPIRAL,random_state=666)
sc.pl.umap(adata_SPIRAL, color='new_batch', ax=axs[1], title='Batch', show=False)
sc.pl.umap(adata_SPIRAL, color='mclust', ax=axs[2],title='Clustered UMAP', show=False)

plt.tight_layout()
plt.savefig('../results/umap_comparison_bc.png')




# import scib
# sc.pp.neighbors(adata_SPIRAL, use_rep="spiral")
# scib.me.graph_connectivity(adata_SPIRAL, label_key="celltype")
# scib.me.ilisi_graph(adata_SPIRAL, batch_key="batch", type_="embed", use_rep="spiral")
# scib.me.kBET(adata_SPIRAL, batch_key="batch", label_key="celltype", type_="embed", embed="spiral")
# #scib.me.kBET(adata_SPIRAL, batch_key="batch", label_key="ceelltype", type_="knn")
# scib.me.silhouette_batch(adata_SPIRAL, batch_key="batch", label_key="celltype", embed="spiral")
