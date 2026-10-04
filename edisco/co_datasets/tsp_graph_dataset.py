"""TSP dataset (dense graphs or kNN-sparsified graphs)."""

import numpy as np
import torch

from sklearn.neighbors import KDTree
from torch_geometric.data import Data as GraphData


def knn_indices(points, k, margin=8):
  """k nearest neighbours of every point, ties broken by lower node index.

  Neighbours are ordered by (distance, node index), so the neighbour sets
  depend only on pairwise distances and node indices. Each point is its own
  nearest neighbour (distance zero), following the DIFUSCO sparsification.
  """
  n = points.shape[0]
  k = min(k, n)
  k_query = min(n, k + margin)
  dist, idx = KDTree(points, leaf_size=30, metric='euclidean').query(
      points, k=k_query, return_distance=True)
  order = np.lexsort((idx, dist), axis=-1)
  dist = np.take_along_axis(dist, order, axis=-1)
  idx = np.take_along_axis(idx, order, axis=-1)
  if k_query < n:
    # A tie that reaches the end of the query window may hide equidistant
    # points with lower indices, so those rows are resolved exhaustively.
    for i in np.nonzero(dist[:, k - 1] == dist[:, -1])[0]:
      d = np.linalg.norm(points - points[i], axis=-1)
      idx[i, :k] = np.lexsort((np.arange(n), d))[:k]
  return idx[:, :k]


class TSPGraphDataset(torch.utils.data.Dataset):
  def __init__(self, data_file, sparse_factor=-1):
    self.data_file = data_file
    self.sparse_factor = sparse_factor
    self.file_lines = open(data_file).read().splitlines()
    print(f'Loaded "{data_file}" with {len(self.file_lines)} lines')

  def __len__(self):
    return len(self.file_lines)

  def get_example(self, idx):
    # Select sample
    line = self.file_lines[idx]
    # Clear leading/trailing characters
    line = line.strip()

    # Extract points
    points = line.split(' output ')[0]
    points = points.split(' ')
    points = np.array([[float(points[i]), float(points[i + 1])] for i in range(0, len(points), 2)])
    # Extract tour
    tour = line.split(' output ')[1]
    tour = tour.split(' ')
    tour = np.array([int(t) for t in tour])
    tour -= 1

    return points, tour

  def __getitem__(self, idx):
    points, tour = self.get_example(idx)
    if self.sparse_factor <= 0:
      # Return a densely connected graph
      adj_matrix = np.zeros((points.shape[0], points.shape[0]))
      for i in range(tour.shape[0] - 1):
        adj_matrix[tour[i], tour[i + 1]] = 1
      # return points, adj_matrix, tour
      return (
          torch.LongTensor(np.array([idx], dtype=np.int64)),
          torch.from_numpy(points).float(),
          torch.from_numpy(adj_matrix).float(),
          torch.from_numpy(tour).long(),
      )
    else:
      # Return a sparse graph where each node is connected to its k nearest neighbors
      # k = self.sparse_factor
      sparse_factor = min(self.sparse_factor, points.shape[0])
      idx_knn = knn_indices(points, sparse_factor)

      edge_index_0 = torch.arange(points.shape[0]).reshape((-1, 1)).repeat(1, sparse_factor).reshape(-1)
      edge_index_1 = torch.from_numpy(idx_knn.reshape(-1))

      edge_index = torch.stack([edge_index_0, edge_index_1], dim=0)

      tour_edges = np.zeros(points.shape[0], dtype=np.int64)
      tour_edges[tour[:-1]] = tour[1:]
      tour_edges = torch.from_numpy(tour_edges)
      tour_edges = tour_edges.reshape((-1, 1)).repeat(1, sparse_factor).reshape(-1)
      tour_edges = torch.eq(edge_index_1, tour_edges).reshape(-1, 1)
      graph_data = GraphData(x=torch.from_numpy(points).float(),
                             edge_index=edge_index,
                             edge_attr=tour_edges)

      point_indicator = np.array([points.shape[0]], dtype=np.int64)
      edge_indicator = np.array([edge_index.shape[1]], dtype=np.int64)
      return (
          torch.LongTensor(np.array([idx], dtype=np.int64)),
          graph_data,
          torch.from_numpy(point_indicator).long(),
          torch.from_numpy(edge_indicator).long(),
          torch.from_numpy(tour).long(),
      )
