"""Tour evaluation and 2-opt local search for the TSP."""

import scipy.spatial
import torch


class TSPEvaluator(object):
    """Length of a closed tour given as a node sequence that returns to its start."""

    def __init__(self, points):
        self.dist_mat = scipy.spatial.distance_matrix(points, points)

    def evaluate(self, route):
        total_cost = 0
        for i in range(len(route) - 1):
            total_cost += self.dist_mat[route[i], route[i + 1]]
        return total_cost


def batched_two_opt_torch(points, tour, max_iterations=1000, device="cpu"):
    """Best-improvement 2-opt on a batch of tours over the same points.

    EDISCO reports results without local search; this routine is provided only
    for reference comparisons with 2-opt post-processing.

    Args:
        points: (n, 2) array of coordinates
        tour: (batch, n + 1) array of closed tours
        max_iterations: maximum number of improving moves
    Returns:
        the improved tours and the number of moves applied
    """
    iterator = 0
    tour = tour.copy()
    if max_iterations <= 0:
        return tour, 0
    with torch.inference_mode():
        cuda_points = torch.from_numpy(points).to(device)
        cuda_tour = torch.from_numpy(tour).to(device)
        batch_size = cuda_tour.shape[0]
        min_change = -1.0
        while min_change < 0.0:
            points_i = cuda_points[cuda_tour[:, :-1].reshape(-1)].reshape((batch_size, -1, 1, 2))
            points_j = cuda_points[cuda_tour[:, :-1].reshape(-1)].reshape((batch_size, 1, -1, 2))
            points_i_plus_1 = cuda_points[cuda_tour[:, 1:].reshape(-1)].reshape((batch_size, -1, 1, 2))
            points_j_plus_1 = cuda_points[cuda_tour[:, 1:].reshape(-1)].reshape((batch_size, 1, -1, 2))

            A_ij = torch.sqrt(torch.sum((points_i - points_j) ** 2, axis=-1))
            A_i_plus_1_j_plus_1 = torch.sqrt(torch.sum((points_i_plus_1 - points_j_plus_1) ** 2, axis=-1))
            A_i_i_plus_1 = torch.sqrt(torch.sum((points_i - points_i_plus_1) ** 2, axis=-1))
            A_j_j_plus_1 = torch.sqrt(torch.sum((points_j - points_j_plus_1) ** 2, axis=-1))

            change = A_ij + A_i_plus_1_j_plus_1 - A_i_i_plus_1 - A_j_j_plus_1
            valid_change = torch.triu(change, diagonal=2)

            min_change = torch.min(valid_change)
            flatten_argmin_index = torch.argmin(valid_change.reshape(batch_size, -1), dim=-1)
            min_i = torch.div(flatten_argmin_index, len(points), rounding_mode='floor')
            min_j = torch.remainder(flatten_argmin_index, len(points))

            if min_change < -1e-6:
                for i in range(batch_size):
                    cuda_tour[i, min_i[i] + 1:min_j[i] + 1] = torch.flip(cuda_tour[i, min_i[i] + 1:min_j[i] + 1], dims=(0,))
                iterator += 1
            else:
                break

            if iterator >= max_iterations:
                break
        tour = cuda_tour.cpu().numpy()
    return tour, iterator
