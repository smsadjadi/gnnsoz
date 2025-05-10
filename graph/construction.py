import numpy as np
import networkx as nx


def construct_graph(time_series: np.ndarray, threshold: float = 0.5) -> nx.Graph:
    """
    Build an undirected connectivity graph from ROI time series.
    Edges exist where |correlation| >= threshold.
    Returns:
        NetworkX Graph object
    """
    # correlation matrix
    corr = np.corrcoef(time_series)
    # threshold
    adj = threshold_matrix(corr, threshold)
    # build graph
    G = nx.from_numpy_array(adj)
    return G


def threshold_matrix(matrix: np.ndarray, threshold: float) -> np.ndarray:
    """
    Apply threshold to connectivity matrix: edges where |value| >= threshold.
    Zero out diagonal.
    """
    adj = (np.abs(matrix) >= threshold).astype(float)
    np.fill_diagonal(adj, 0)
    return adj


def normalize_adjacency(adj: np.ndarray) -> np.ndarray:
    """
    Compute symmetric normalized adjacency matrix A_norm = D^-1/2 (A + I) D^-1/2
    """
    N = adj.shape[0]
    A = adj + np.eye(N)
    degree = np.sum(A, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return A_norm