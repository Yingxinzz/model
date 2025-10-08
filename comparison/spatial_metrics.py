import scanpy as sc
import pandas as pd
import squidpy as sq
import numpy as np
from scipy.spatial import *
from sklearn.preprocessing import *

from sklearn.metrics import *
from scipy.spatial.distance import *

from sklearn.neighbors import NearestNeighbors
from tqdm import trange
import os


class spatialbench:

    @staticmethod
    def _get_spatial_entropy(C, C_sum):
        H = 0
        for i in range(len(C)):
            for j in range(len(C)):
                z = C[i, j]
                if z != 0:
                    H += -(z / C_sum) * np.log(z / C_sum)
        return H

    @staticmethod
    def spatial_entropy(k_neighbors, labels, degree=4):
        S = np.broadcast_to(labels[:, None], (len(labels), degree))
        N = labels[k_neighbors]
        cluster_names = np.unique(labels)
        cluster_nums = len(cluster_names)
        C = np.zeros((cluster_nums, cluster_nums))
        for i in range(cluster_nums):
            for j in range(cluster_nums):
                C[i, j] = np.sum(
                    np.logical_and(S == cluster_names[i], N == cluster_names[j])
                )
        C_sum = C.sum()
        return spatialbench._get_spatial_entropy(C, C_sum)

    @staticmethod
    def spatial_coherence_score(adata, annotation_key, degree=4, rep_time=1000, seed=0):
        spatial_coords = adata.obsm["spatial"]
        origin_labels = adata.obs[annotation_key].values
        neigh = NearestNeighbors(n_neighbors=degree, metric="euclidean").fit(
            spatial_coords
        )
        k_neighbors = neigh.kneighbors(n_neighbors=degree, return_distance=False)
        true_entropy = spatialbench.spatial_entropy(
            k_neighbors, origin_labels, degree=degree
        )
        entropies = []
        rng = np.random.default_rng(seed)
        shuffled_labels = origin_labels.copy()
        for _ in trange(rep_time):
            rng.shuffle(shuffled_labels)
            entropies.append(
                spatialbench.spatial_entropy(
                    k_neighbors, shuffled_labels, degree=degree
                )
            )
        scs = (true_entropy - np.mean(entropies)) / np.std(entropies)
        return scs

    @staticmethod
    def moran_geary_preservation(adata, celltype_key, top_n=3):
        adata = adata.copy()

        count_dict = adata.obs[celltype_key].value_counts()
        valid_celltypes = count_dict[count_dict > 3].index
        adata = adata[adata.obs[celltype_key].isin(valid_celltypes)]

        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.tl.rank_genes_groups(adata, groupby=celltype_key, use_raw=False)

        selected_genes = []
        for i in range(top_n):
            selected_genes.extend(list(adata.uns["rank_genes_groups"]["names"][i]))
        selected_genes = np.unique(selected_genes)

        sq.gr.spatial_neighbors(adata)
        sq.gr.spatial_autocorr(
            adata, mode="moran", genes=selected_genes, n_perms=100, n_jobs=1
        )
        sq.gr.spatial_autocorr(
            adata, mode="geary", genes=selected_genes, n_perms=100, n_jobs=1
        )

        moran_vals = adata.uns["moranI"]["I"]
        geary_vals = adata.uns["gearyC"]["C"]

        moran_vals = moran_vals[~np.isnan(moran_vals)]
        geary_vals = geary_vals[~np.isnan(geary_vals)]

        print("[DEBUG] MoranI values:", moran_vals.head(10))
        print("[DEBUG] GearyC values:", geary_vals.head(10))

        moranI = np.median(moran_vals) if len(moran_vals) > 0 else np.nan
        gearyC = np.median(geary_vals) if len(geary_vals) > 0 else np.nan

        return moranI, gearyC
