from kfsims.node import make_simple_nodes
import networkx as nx
import numpy as np


def create_network(n, k=5, iterations=10):
    nodes = make_simple_nodes(n, iterations)
    G_ = nx.random_regular_graph(k, len(nodes))
    G = nx.relabel_nodes(G_, {ix: nodes[ix] for ix in range(len(nodes))})
    return G


def _get_neighbors_att(G, node, prior_pref):
    res = []
    for ngh in G.neighbors(node):
        res.append(getattr(ngh, prior_pref + '_prior').hp)
    return res


def fuse_parameters(params_):
    params = np.array(params_)
    return np.mean(params, axis=0)


def node_neighbors_fusion_update(G, node):
    Ps = _get_neighbors_att(G, node, 'P') + [node.P_prior.hp]
    Rs = _get_neighbors_att(G, node, 'R') + [node.R_prior.hp]
    new_P = fuse_parameters(Ps)
    new_R = fuse_parameters(Rs)
    return new_P, new_R


def update_node_by_neighbors(G, node):
    hyp_P, hyp_R = node_neighbors_fusion_update(G, node)
    node.R_prior.hp = hyp_R
    node.P_prior.hp = hyp_P
