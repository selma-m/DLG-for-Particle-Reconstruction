import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import torch_geometric.typing
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear

import torch_cluster
from torch_cluster import knn
from torch_geometric.nn import GravNetConv
from torch_geometric.testing import is_full_test, withPackage
from torch_kmeans import SoftKMeans

import numpy as np
import scipy as sp
import sklearn
import sklearn.cluster
from sklearn.metrics.cluster._supervised import check_clusterings
from sklearn.metrics import mutual_info_score


# Helper functions
def map_to_integers(arr):
    value_to_int = {}
    current_int = 0
    result = []

    for value in arr:
        # If the value hasn't been mapped yet, map it to the current integer
        if value not in value_to_int:
            value_to_int[value] = current_int
            current_int += 1
        # Append the mapped integer to the result list
        result.append(value_to_int[value])

    return result


# GravNet clustering architecture
class GravNetClustering(nn.Module):
    def __init__(self, k = 30, S_dim = 10, f_lr_dim = 21, n_neighbors = 40, f_out_dim = 42):

        # S_dim: dimensions of the embedding space S (this is used for finding the k-NN)
        # n_neighbors: number of neighbors to consider when aggregating information
        # f_out_dim: dimension of the output of the final dense layer of GravNet
        # f_lr_dim: dimension of the updated features (this is what is aggregated)

        super(GravNetClustering, self).__init__()

        self.k = k

        # First block
        self.first_block = nn.Sequential(
            # 14 is two times the number of input node features
            nn.Linear(in_features = 14, out_features = 64), # dense layer
            nn.Tanh(),
            nn.Linear(in_features = 64, out_features = 128), # dense layer
            nn.Tanh(),
            nn.Linear(in_features = 128, out_features = 64), # dense layer
            nn.Tanh(),
            GravNetConv(64,f_out_dim,space_dimensions = S_dim, propagate_dimensions = f_lr_dim, k = n_neighbors)
            # output is batch_size x n_nodes x f_out_dim
        )

        # Second block
        self.second_block = nn.Sequential(
            nn.Linear(in_features = f_out_dim*2, out_features = 64),
            nn.Tanh(),
            nn.Linear(in_features = 64, out_features = 128),
            nn.Tanh(),
            nn.Linear(in_features = 128, out_features = 64),
            nn.Tanh(),
            GravNetConv(64,f_out_dim,space_dimensions = S_dim, propagate_dimensions = f_lr_dim, k = n_neighbors*2)
        )

        # Third block
        self.third_block = nn.Sequential(
            nn.Linear(in_features = f_out_dim*2, out_features = 64),
            nn.Tanh(),
            nn.Linear(in_features = 64, out_features = 128),
            nn.Tanh(),
            nn.Linear(in_features = 128, out_features = 64),
            nn.Tanh(),
            GravNetConv(64,f_out_dim,space_dimensions = S_dim, propagate_dimensions = f_lr_dim, k = n_neighbors*2)
        )

        # Fourth block
        self.fourth_block = nn.Sequential(
            nn.Linear(in_features = f_out_dim*2, out_features = 64),
            nn.Tanh(),
            nn.Linear(in_features = 64, out_features = 128),
            nn.Tanh(),
            nn.Linear(in_features = 128, out_features = 64),
            nn.Tanh(),
            GravNetConv(64,f_out_dim,space_dimensions = S_dim, propagate_dimensions = f_lr_dim, k = n_neighbors*2)
        )

        # 2 dense layers
        self.lin_1 = Linear(in_channels = f_out_dim*4, out_channels = 64)
        self.act = nn.LeakyReLU() # arguably more expressive than ReLU
        self.lin_2 = Linear(in_channels = 64, out_channels = 100)
        # self.softmax = nn.Softmax(dim = -1)


    def forward(self, x):
        feats = []

        ## First block
        global_mean = x.mean(dim = 1, keepdim = True)
        x = torch.cat((x,global_mean.expand_as(x)), dim = -1) # batch_size x n_nodes x 2*feature_dim
        x = self.first_block(x)
        feats.append(x)

        ## Second block
        global_mean = x.mean(dim = 1, keepdim = True)
        x = torch.cat((x,global_mean.expand_as(x)), dim = -1) # batch_size x n_nodes x 2*feature_dim
        x = self.second_block(x)
        feats.append(x)

        ## Third block
        global_mean = x.mean(dim = 1, keepdim = True)
        x = torch.cat((x,global_mean.expand_as(x)), dim = -1) # batch_size x n_nodes x 2*feature_dim
        x = self.third_block(x)
        feats.append(x)

        ## Fourth block
        global_mean = x.mean(dim = 1, keepdim = True)
        x = torch.cat((x,global_mean.expand_as(x)), dim = -1) # batch_size x n_nodes x 2*feature_dim
        x = self.fourth_block(x)
        feats.append(x)

        # Concatenate outputs of the four blocks
        x = torch.cat(feats, dim = -1) # batch x n_nodes x f_out_dim * 4
        embedding = x

        ## Final dense layers
        x = self.lin_1(x)
        x = self.act(x)
        x = self.lin_2(x)
        x = self.act(x) # B x n_nodes x k
        # x = self.softmax(x)

        # We return the embedding before the dense layers for visualization
        return x, embedding


## Performance metrics

# Weighted V-Score
def weighted_v_score(labels_true, labels_pred, beta=1.0, labels_weight=None):

    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)

    if len(labels_true) == 0:
        return 1.0, 1.0, 1.0

    entropy_C_hat = weighted_entropy(labels_true, weights=labels_weight)
    entropy_K_hat = weighted_entropy(labels_pred, weights=labels_weight)

    contingency_hat = weighted_contingency_matrix(labels_true, labels_pred, sparse=True, weights=labels_weight)
    MI_hat = mutual_info_score(None, None, contingency=contingency_hat)

    homogeneity_hat = MI_hat / (entropy_C_hat) if entropy_C_hat else 1.0
    completeness_hat = MI_hat / (entropy_K_hat) if entropy_K_hat else 1.0

    if homogeneity_hat + completeness_hat == 0.0:
        v_measure_score_hat = 0.0
    else:
        v_measure_score_hat = (
            (1 + beta)
            * homogeneity_hat
            * completeness_hat
            / (beta * homogeneity_hat + completeness_hat)
        )

    return homogeneity_hat, completeness_hat, v_measure_score_hat

def weighted_entropy(labels, weights=None):
    """Calculates the entropy for a labeling."""
    if weights is None:
        weights = np.ones(len(labels))
    
    _, labels = np.unique(labels, return_inverse=True)

    pi_hat = np.bincount(labels, weights=weights)
    pi_hat = pi_hat[pi_hat > 0]
    pi_hat_sum = np.sum(pi_hat)

    return -np.sum((pi_hat / pi_hat_sum) * (np.log(pi_hat) - np.log(pi_hat_sum)))

def weighted_contingency_matrix(labels_true, labels_pred, sparse=False, weights=None):
    """Build a contengency matrix describing the relationship between labels.
    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        Ground truth class labels to be used as a reference
    labels_pred : array, shape = [n_samples]
        Cluster labels to evaluate
    sparse : boolean, default False
        If True, return a sparse CSR continency matrix. If 'auto', the sparse
        matrix is returned for a dense input and vice-versa.
    weights : array, shape = [n_samples], optional
        Sample weights.
    """

    classes, class_idx = np.unique(labels_true, return_inverse=True)
    clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
    n_classes = classes.shape[0]
    n_clusters = clusters.shape[0]

    if weights is None:
        weights = np.ones(len(labels_true))

    # Make a float sparse array
    contingency = sp.sparse.coo_matrix(
        (weights, (class_idx, cluster_idx)),
        shape=(n_classes, n_clusters),
        dtype=np.float64
    )

    if sparse:
        contingency = contingency.tocsr()
        contingency.sum_duplicates()
        return contingency

    return contingency.toarray() 
