import os
import gc
import time
import json
import psutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import anndata
import scanpy as sc
import argparse
from sklearn.decomposition import PCA
from operator import itemgetter
import random
import matplotlib.pyplot as plt
import umap.umap_ as umap
import sys
sys.path.append('../')
from spiral.main import SPIRAL_integration
from spiral.layers import *
from spiral.utils import *
from spiral.CoordAlignment import CoordAlignment
from warnings import filterwarnings
filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

# =============== Configuration and Data Setup ===============
input_dir="../data/mouse_olfactory_bulb/processed/35um/"
results_dir="../results/"
datasets=["BGI","SlideV2","10X"]
SEP=','
net_cate='_KNN_'
knn=8
N_WALKS=knn
WALK_LEN=1
N_WALK_LEN=knn
NUM_NEG=knn

feat_file=[]
edge_file=[]
meta_file=[]
coord_file=[]
for dataset in datasets:
    if dataset in ["BGI", "SlideV2"]:
        knn = 8
    else:
        knn = 6 
    feat_file.append(input_dir + str(dataset) + "_mat.csv")
    edge_file.append(input_dir + str(dataset) + "_edge" + net_cate + str(knn) + ".csv")
    meta_file.append(input_dir + str(dataset) + "_meta.csv")
    coord_file.append(input_dir + str(dataset) + "_coord.csv")

N=pd.read_csv(feat_file[0],header=0,index_col=0).shape[1]
if (len(datasets)==2):
    M=1
else:
    M=len(datasets)

total_cells = 0
for feat in feat_file:
    df = pd.read_csv(feat, header=0, index_col=0)
    total_cells += df.shape[0]

print(f"Total cells to process: {total_cells:,}")
print(f"Total genes: {N:,}")
print(f"Number of datasets: {len(datasets)}")

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
parser.add_argument('--num_samples', type=int, default=20, help='number of neighbors to sample.')

parser.add_argument('--N_WALKS', type=int, default=N_WALKS, help='number of walks of random work for postive pairs.')
parser.add_argument('--WALK_LEN', type=int, default=WALK_LEN, help='walk length of random work for postive pairs.')
parser.add_argument('--N_WALK_LEN', type=int, default=N_WALK_LEN, help='number of walks of random work for negative pairs.')
parser.add_argument('--NUM_NEG', type=int, default=NUM_NEG, help='number of negative pairs.')

parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=512, help='Size of batches to train.')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay.')
parser.add_argument('--alpha1', type=float, default=N, help='Weight of decoder loss.')
parser.add_argument('--alpha2', type=float, default=1, help='Weight of GraphSAGE loss.')
parser.add_argument('--alpha3', type=float, default=1, help='Weight of classifier loss.')
parser.add_argument('--alpha4', type=float, default=1, help='Weight of discriminator loss.')
parser.add_argument('--lamda', type=float, default=1, help='Weight of GRL.') #####Stereo-seq:35um resolution 
parser.add_argument('--Q', type=float, default=10, help='Weight negative loss for sage losss.')

params,unknown=parser.parse_known_args()

# =============== SPIRAL Training with Benchmarking ===============
print("Starting SPIRAL training benchmarking...")

# Clear memory before training for fair comparison
gc.collect()
torch.cuda.empty_cache()

memory_before = get_memory_usage()
training_start_time = time.time()

print("Training SPIRAL model...")
SPII=SPIRAL_integration(params,feat_file,edge_file,meta_file)
SPII.train()

training_end_time = time.time()
training_time = training_end_time - training_start_time
memory_after = get_memory_usage()
memory_used = memory_after - memory_before

print("Training completed!")

# =============== Model Saving ===============
if not os.path.exists(results_dir+"model/"):
    os.makedirs(results_dir+"model/")
model_file = os.path.join(
    results_dir,
    "model",
    f"SPIRAL_embed_{SPII.params.batch_size}_{'_'.join(map(str, datasets))}.pt"
)
torch.save(SPII.model.state_dict(),model_file)

SPII.model.eval()
all_idx=np.arange(SPII.feat.shape[0])
all_layer,all_mapping=layer_map(all_idx.tolist(),SPII.adj,len(SPII.params.GSdims))
all_rows=SPII.adj.tolil().rows[all_layer[0]]
all_feature=torch.Tensor(SPII.feat.iloc[all_layer[0],:].values).float().cuda()
all_embed,ae_out,clas_out,disc_out=SPII.model(all_feature,all_layer,all_mapping,all_rows,SPII.params.lamda,SPII.de_act,SPII.cl_act)
[ae_embed,gs_embed,embed]=all_embed
[x_bar,x]=ae_out

embed=embed.cpu().detach()
embed=embed[:,SPII.params.znoise_dim:]
names=['GTT_'+str(i) for i in range(embed.shape[1])]
embed1=pd.DataFrame(np.array(embed),index=SPII.feat.index,columns=names)
if not os.path.exists(results_dir+"gtt_output/"):
    os.makedirs(results_dir+"gtt_output/")

embed_file = os.path.join(results_dir,"gtt_output",f"SPIRAL_embed_{SPII.params.batch_size}_{'_'.join(map(str, datasets))}.csv")
embed1.to_csv(embed_file)

# =============== Clustering and Analysis ===============
print("Performing clustering analysis...")

adata=anndata.AnnData(SPII.feat)
adata.obsm['spiral']=embed1.values
sc.pp.neighbors(adata,use_rep='spiral')
n_clust=8
adata = mclust_R(adata, used_obsm='spiral', num_cluster=n_clust)
adata.obs['new_batch']=SPII.meta.loc[:,'batch'].values
adata.obs['celltype']=SPII.meta.loc[:,'celltype'].values
celltype_mapping = {
    'ONL': 'ONL',
    'GL': 'GL',
    'GCL': 'GCL',
    'MCL': 'MCL',
    'OPL': 'OPL',
    'EPL': 'EPL',
    'SEZ': 'SEZ',
    'Meninges': 'Meninges',
    'GL_1': 'GL',
    'GL_2': 'GL',
    'GCL_1': 'GCL',
    'GCL_2': 'GCL',
    'Low_Quality': None
}
adata.obs['celltype'] = adata.obs['celltype'].map(celltype_mapping).fillna(adata.obs['celltype'])
if None in adata.obs['celltype'].values:
    adata = adata[adata.obs['celltype'] != None].copy()

coord=pd.read_csv(coord_file[0],header=0,index_col=0)
for i in np.arange(1,len(datasets)):
    coord=pd.concat((coord,pd.read_csv(coord_file[i],header=0,index_col=0)))

adata.obsm['spatial']=coord.loc[adata.obs_names,:].values
adata.obs['SPIRAL']=adata.obs['mclust']
adata.write('../results/spiral_mob3.h5ad')

# =============== Save Benchmark Results ===============
benchmark_results = {
    'method_name': 'SPIRAL',
    'training_time_seconds': training_time,
    'training_time_minutes': training_time / 60,
    'training_time_hours': training_time / 3600,
    'memory_usage_mb': memory_used,
    'memory_usage_gb': memory_used / 1024,
    'total_cells': total_cells,
    'final_cells': SPII.feat.shape[0],
    'total_genes': N,
    'embedding_dim': embed.shape[1],
    'n_datasets': len(datasets),
    'epochs': params.epochs,
    'timestamp': pd.Timestamp.now().isoformat()
}

# Save benchmark results
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

with open(os.path.join(results_dir, "spiral_benchmark_mob.json"), "w") as f:
    json.dump(benchmark_results, f, indent=2)

# =============== Print Benchmark Results ===============
gc.collect()
torch.cuda.empty_cache()

final_memory = get_memory_usage()
print(f"Final memory usage: {final_memory:.1f} MB")
