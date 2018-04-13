from kfsims.node import make_simple_nodes
import networkx as nx
import numpy as np


def create_network(n, k=5, iterations=10):
    nodes = make_simple_nodes(n, iterations)
    G_ = nx.random_regular_graph(k, len(nodes))
    #G_ = nx.caveman_graph(int(n/k), k)
    G = nx.relabel_nodes(G_, {ix: nodes[ix] for ix in range(len(nodes))})
    return G


def _get_neighbors_att(cluster, prior_pref):
    res = []
    for ngh in cluster:
        res.append(getattr(ngh, prior_pref + '_prior').hp)
    return res


def fuse_parameters(params_):
    params = np.array(params_)
    return np.mean(params, axis=0)


def get_cluster_params(cluster):
    Ps = _get_neighbors_att(cluster, 'P')# + [node.P_prior.hp]
    Rs = _get_neighbors_att(cluster, 'R')# + [node.R_prior.hp]
    new_P = fuse_parameters(Ps)
    new_R = fuse_parameters(Rs)
    return new_P, new_R


def update_nodes_neighbors_cluster(G, node):
    cluster = set(G.neighbors(node)) | {node}
    hyp_P, hyp_R = get_cluster_params(cluster)
    for n in cluster:
        n.R_prior.hp = hyp_R
        n.P_prior.hp = hyp_P
    return cluster
