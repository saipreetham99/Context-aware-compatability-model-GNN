# file: utils.py
import json
import numpy as np
import time
import scipy.sparse as sp
import tensorflow as tf


def to_tf_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a TensorFlow SparseTensor."""
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return tf.SparseTensor(indices=coords, values=values, dense_shape=shape)


def support_dropout(sup, do, edge_drop=False):
    sup = sp.tril(sup).copy()
    assert 0.0 < do < 1.0
    n_nodes = sup.shape[0]
    # nodes that I want to isolate
    isolate = np.random.choice(range(n_nodes), int(n_nodes * do), replace=False)
    nnz_rows, nnz_cols = sup.nonzero()

    # mask the nodes that have been selected
    mask = np.in1d(nnz_rows, isolate) | np.in1d(nnz_cols, isolate)
    sup.data[mask] = 0
    sup.eliminate_zeros()

    if edge_drop:
        prob = np.random.uniform(0, 1, size=sup.data.shape)
        remove = prob < do
        sup.data[remove] = 0
        sup.eliminate_zeros()

    sup = sup + sup.transpose()
    return sup


def write_log(data, logfile):
    with open(logfile, "w") as outfile:
        json.dump(data, outfile)


def get_degree_supports(adj, k, adj_self_con=False, verbose=True):
    if verbose:
        print("Computing adj matrices up to {}th degree".format(k))

    if k == 0:
        return [sp.identity(adj.shape[0], dtype=np.float32)]

    supports = [sp.identity(adj.shape[0], dtype=np.float32)]

    # adj with or without self-connections
    adj_with_self = adj.astype(np.float32)
    if adj_self_con:
        adj_with_self += sp.identity(adj.shape[0], dtype=np.float32)
    supports.append(adj_with_self)

    prev_power = adj.astype(np.float32)
    for i in range(k - 1):
        pow_adj = prev_power.dot(adj)
        new_adj = (pow_adj > 0).astype(np.float32)  # Binarize
        new_adj.setdiag(0)
        new_adj.eliminate_zeros()
        supports.append(new_adj)
        prev_power = pow_adj
    return supports


def normalize_nonsym_adj(adj):
    degree = np.asarray(adj.sum(1)).flatten()
    degree[degree == 0.0] = np.inf
    degree_inv = 1.0 / degree
    degree_inv_mat = sp.diags(degree_inv)
    adj_norm = degree_inv_mat.dot(adj)
    return adj_norm


class Graph(object):
    """docstring for Graph."""

    def __init__(self, adj):
        super(Graph, self).__init__()
        self.adj = adj
        self.n_nodes = adj.shape[0]
        self.level = 0

    def run_K_BFS(self, n, K):
        """
        Returns a list of K edges, sampled using BFS starting from n
        """
        visited = {n}
        edges = []
        queue = [n]

        while queue and len(edges) < K:
            node = queue.pop(0)
            neighbors = list(self.adj[node].nonzero()[1])

            for neigh in neighbors:
                if neigh not in visited:
                    visited.add(neigh)
                    edges.append((node, neigh))
                    queue.append(neigh)
                    if len(edges) == K:
                        return edges
        return edges
