"""Utility functions for Maximum Independent Set (MIS) problem."""

import numpy as np


def mis_decode_np(predictions, adj_matrix):
    """Decode node predictions to a valid Maximum Independent Set.

    Uses a greedy decoding strategy: select nodes in order of decreasing
    predicted probability, marking neighbors as unavailable.

    Args:
        predictions: Node probability predictions (num_nodes,)
        adj_matrix: Sparse adjacency matrix (scipy.sparse format)

    Returns:
        solution: Binary array indicating MIS membership (num_nodes,)
    """
    solution = np.zeros_like(predictions, dtype=int)
    sorted_predict_labels = np.argsort(-predictions)  # Descending order
    csr_adj_matrix = adj_matrix.tocsr()

    for i in sorted_predict_labels:
        next_node = i

        # Skip if already marked as neighbor of selected node
        if solution[next_node] == -1:
            continue

        # Mark all neighbors as unavailable
        solution[csr_adj_matrix[next_node].nonzero()[1]] = -1
        # Select this node
        solution[next_node] = 1

    return (solution == 1).astype(int)
