import networkx as nx
import numpy as np


def compute_clustering(G: nx.Graph) -> dict:
    """
    Local clustering coefficient per node.
    Returns dict mapping node -> coefficient.
    """
    return nx.clustering(G)


def compute_local_efficiency(G: nx.Graph) -> dict:
    """
    Local efficiency per node: global efficiency of the subgraph induced by each node's neighbors.
    Returns dict mapping node -> local efficiency.
    """
    eff = {}
    for n in G.nodes():
        neighbors = list(G.neighbors(n))
        if len(neighbors) < 2:
            eff[n] = 0.0
        else:
            sub = G.subgraph(neighbors)
            eff[n] = nx.global_efficiency(sub)
    return eff


def compute_connectivity_diversity(G: nx.Graph) -> dict:
    """
    Connectivity diversity per node: variance of edge weights (for unweighted graph, var of degrees).
    Returns dict mapping node -> diversity.
    """
    div = {}
    degrees = dict(G.degree(weight=None))
    all_degrees = np.array(list(degrees.values()), dtype=float)
    mean = np.mean(all_degrees)
    var_all = np.var(all_degrees, ddof=1)
    for n, d in degrees.items():
        div[n] = var_all
    return div