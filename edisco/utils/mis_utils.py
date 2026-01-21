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


def mis_decode_torch(predictions, adj_matrix, device='cuda'):
    """Decode node predictions to a valid MIS using PyTorch.

    Similar to mis_decode_np but uses PyTorch tensors for potential GPU acceleration.

    Args:
        predictions: Node probability predictions (num_nodes,) - torch.Tensor
        adj_matrix: Sparse adjacency matrix (scipy.sparse format)
        device: Device to use for computation

    Returns:
        solution: Binary tensor indicating MIS membership (num_nodes,)
    """
    import torch

    predictions_np = predictions.cpu().numpy() if torch.is_tensor(predictions) else predictions
    solution = mis_decode_np(predictions_np, adj_matrix)
    return torch.from_numpy(solution).to(device)


def compute_mis_size(solution):
    """Compute the size of an independent set.

    Args:
        solution: Binary array indicating set membership

    Returns:
        Size of the independent set (sum of selected nodes)
    """
    if hasattr(solution, 'sum'):
        return int(solution.sum())
    return np.sum(solution)


def verify_independent_set(solution, adj_matrix):
    """Verify that a solution is a valid independent set.

    Args:
        solution: Binary array indicating set membership
        adj_matrix: Adjacency matrix (dense or sparse)

    Returns:
        is_valid: True if no two selected nodes are adjacent
    """
    import scipy.sparse

    if scipy.sparse.issparse(adj_matrix):
        adj_matrix = adj_matrix.toarray()

    selected = np.where(solution == 1)[0]
    for i in selected:
        for j in selected:
            if i != j and adj_matrix[i, j] != 0:
                return False
    return True


class MISEvaluator:
    """Evaluator for Maximum Independent Set solutions."""

    def __init__(self, adj_matrix):
        """
        Args:
            adj_matrix: Adjacency matrix of the graph
        """
        self.adj_matrix = adj_matrix

    def evaluate(self, solution):
        """Evaluate an MIS solution.

        Args:
            solution: Binary array indicating set membership

        Returns:
            size: Size of the independent set (negative for minimization)
        """
        return compute_mis_size(solution)

    def is_valid(self, solution):
        """Check if a solution is valid.

        Args:
            solution: Binary array indicating set membership

        Returns:
            True if the solution is a valid independent set
        """
        return verify_independent_set(solution, self.adj_matrix)
