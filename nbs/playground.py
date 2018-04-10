from kfsims.node import make_simple_nodes
from kfsims.common import init_trajectory
import networkx as nx
import numpy as np

def create_network(n, k=5):
    nodes = make_simple_nodes(n)
    G_ = nx.random_regular_graph(k, len(nodes))
    G = nx.relabel_nodes(G_, {ix: nodes[ix] for ix in range(len(nodes))})
    G.get_by_mid = lambda x: G[[node for node in nodes if node.label == x][0]]  #G.get_by_mid('86088')
    G.local_kf_on_all_nodes = lambda: [next(node._kf_iterator) for node in G]
    return G

def _get_neighbors_att(G, node, prior_pref):
    """
    Examples:
        _get_neighbors_att(G, node, 'P')
        _get_neighbors_att(G, node, 'R')
    """
    res = []
    for ngh in G.neighbors(node):
        res.append(getattr(ngh, prior_pref + '_prior').hp)
    return res

def fuse_parameters(params_):
    params = np.array(params_)
    r = sum(params) / len(params)
    return r

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

traj = init_trajectory()

# tohle je dle mého ono
net = create_network(6, 3)
msrms = {node: (i for i in node.observe()) for node in net}
rmses = []
for i in range(traj.X.shape[1]):
    for node, ms in msrms.items():
        m = next(ms)
        x, *_ = node.single_kf(m)
        x


    for node in net:
        # print(f'hyperparams {node.label}')
        # print_node_P_nu_R_nu(node)
        update_node_by_neighbors(net, node)
        # print_node_P_nu_R_nu(node)

for node in net:
    rmses.append(node.post_rmse(traj.X))
np.mean(rmses)