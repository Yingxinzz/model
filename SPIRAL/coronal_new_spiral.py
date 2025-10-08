import os
import gc
import time
import json
import psutil
import argparse
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
input_dir = "../data/mouse_brain_coronal/processed"
results_dir = "../results/"
datasets = ['10X_FFPE', '10X_DAPI', '10X_Normal']
SEP=','
net_cate='_KNN_'
rad=150
knn=6

N_WALKS=knn
WALK_LEN=1
N_WALK_LEN=knn
NUM_NEG=knn

feat_file = []
meta_file = []
edge_file = []
coord_file = []
for i in range(len(datasets)):
    feat_file.append(input_dir + '/' + datasets[i] + "_mat.csv")
    meta_file.append(input_dir + '/' + datasets[i] + "_meta.csv")
    edge_file.append(input_dir + '/' + datasets[i] + f"_edge_KNN_{knn}.csv")  
    coord_file.append(input_dir + '/' + datasets[i] + "_coord.csv")

N = pd.read_csv(feat_file[0], header=0, index_col=0).shape[1]
M = 1 if len(datasets) == 2 else len(datasets)

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

# =============== Clustering and Analysis ===============
print("Performing clustering analysis...")

ann=anndata.AnnData(SPII.feat)
ann.obsm['spiral']=embed1.iloc[:,SPII.params.znoise_dim:].values
sc.pp.neighbors(ann,use_rep='spiral')
n_clust=12
ann = mclust_R(ann, used_obsm='spiral', num_cluster=n_clust)
ann.X=SPII.feat
ann.obs['batch']=SPII.meta.loc[:,'batch'].values
ann.obs['celltype']=SPII.meta.loc[:,'celltype_new'].values
ub=np.unique(ann.obs['batch'])

coord=pd.read_csv(coord_file[0],header=0,index_col=0)
for i in np.arange(1,len(datasets)):
    coord=pd.concat((coord,pd.read_csv(coord_file[i],header=0,index_col=0)))

coord.columns=['y','x']
ann.obsm['spatial']=coord.loc[ann.obs_names,:].values
ann.obs["new_batch"] = ann.obs["batch"].str.replace("10X_", "", regex=False)
ann.write('../results/spiral_coronal.h5ad')
adata.write('../SPIRAL/results/spiral_coronal.h5ad')

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

with open(os.path.join(results_dir, "spiral_benchmark_coronal.json"), "w") as f:
    json.dump(benchmark_results, f, indent=2)

# =============== Print Benchmark Results ===============
# Clean up memory after processing
gc.collect()
torch.cuda.empty_cache()

final_memory = get_memory_usage()
print(f"Final memory usage: {final_memory:.1f} MB")
