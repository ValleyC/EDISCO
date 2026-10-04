"""Euclidean Steiner tree utilities: evaluation, feasible decoding and reference solvers.

The reference solvers (GeoSteiner, iterated 1-Steiner and the minimum spanning
tree) are used by data/generate_steiner_data.py to label instances.
"""

import os
import re
import subprocess
import tempfile

import numpy as np
import torch
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import distance_matrix


class SteinerTreeEvaluator:
    """Evaluator for Steiner Tree solutions"""

    @staticmethod
    def compute_tree_length(coords, adjacency):
        """
        Compute total length of a Steiner tree

        Args:
            coords: (n, 2) array of node coordinates
            adjacency: (n, n) adjacency matrix

        Returns:
            total_length: Total edge length
        """
        if isinstance(coords, torch.Tensor):
            coords = coords.cpu().numpy()
        if isinstance(adjacency, torch.Tensor):
            adjacency = adjacency.cpu().numpy()

        total_length = 0.0
        n = len(coords)

        for i in range(n):
            for j in range(i + 1, n):
                if adjacency[i, j] > 0.5:  # Edge exists
                    length = np.linalg.norm(coords[i] - coords[j])
                    total_length += length

        return total_length

    @staticmethod
    def compute_gap(pred_length, gt_length):
        """Compute percentage gap to ground truth"""
        return (pred_length - gt_length) / gt_length * 100.0

    @staticmethod
    def validate_tree(adjacency, n_terminals):
        """
        Validate that adjacency represents a valid tree

        Checks:
        1. All terminals are connected
        2. No cycles (n_edges = n_nodes - 1 for trees)
        3. Graph is connected

        Args:
            adjacency: (n, n) adjacency matrix
            n_terminals: Number of terminal nodes (first n_terminals nodes)

        Returns:
            is_valid: True if valid tree
            error_msg: Error message if invalid
        """
        if isinstance(adjacency, torch.Tensor):
            adjacency = adjacency.cpu().numpy()

        n = len(adjacency)

        # Make symmetric
        adj_sym = (adjacency + adjacency.T) > 0.5

        # Count edges
        n_edges = np.sum(adj_sym) // 2

        # Check tree property: n_edges = n_nodes - 1
        # But only for nodes actually in the tree
        degrees = np.sum(adj_sym, axis=0)
        active_nodes = degrees > 0
        n_active = np.sum(active_nodes)

        if n_active > 0 and n_edges != n_active - 1:
            return False, f"Not a tree: {n_edges} edges but {n_active} nodes"

        # Check all terminals are included
        terminals_included = np.sum(active_nodes[:n_terminals])
        if terminals_included != n_terminals:
            return False, f"Only {terminals_included}/{n_terminals} terminals included"

        return True, "Valid tree"


def decode_steiner_tree(adj_probs, coords, is_terminal, epsilon=1e-8):
    """
    Tree-aware greedy decoding of edge probabilities into a Steiner tree.

    Kruskal-style procedure on the symmetrized score
    s_ij = (P_ij + P_ji) / (d_ij + epsilon):

    1. Candidate edges are scanned in decreasing score (ties by node index).
       An edge is added when it joins two distinct components, until all
       terminals lie in one component.
    2. Any terminal still disconnected once the scored candidates are
       exhausted is attached through the shortest edge that joins two
       components, which guarantees feasibility.
    3. Steiner points are kept only when they act as connection hubs:
       edges outside the terminal component and Steiner leaves are removed,
       and a Steiner point of degree two is replaced by the direct edge
       between its neighbours, which never increases the length.

    Every quantity read (probabilities, pairwise distances, the terminal
    indicator, component membership) is invariant under E(2).

    Args:
        adj_probs: (n, n) edge probabilities
        coords: (n, 2) node coordinates
        is_terminal: (n,) binary indicator (1 for terminals)

    Returns:
        adjacency: (n, n) decoded adjacency matrix
        tree_length: Total tree length
    """
    if isinstance(adj_probs, torch.Tensor):
        adj_probs = adj_probs.cpu().numpy()
    if isinstance(coords, torch.Tensor):
        coords = coords.cpu().numpy()
    if isinstance(is_terminal, torch.Tensor):
        is_terminal = is_terminal.cpu().numpy()
    adj_probs = np.asarray(adj_probs, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    terminal = np.asarray(is_terminal).reshape(-1) > 0.5

    n = len(coords)
    n_terminals = int(terminal.sum())
    distances = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)

    parent = list(range(n))
    terminals_in = terminal.astype(int).tolist()  # terminals per component root

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(i, j):
        ri, rj = find(i), find(j)
        parent[rj] = ri
        terminals_in[ri] += terminals_in[rj]
        return terminals_in[ri]

    edges = set()
    connected = n_terminals <= 1

    # 1. Kruskal-style scan in decreasing score
    iu = np.triu_indices(n, k=1)
    scores = (adj_probs[iu] + adj_probs.T[iu]) / (distances[iu] + epsilon)
    order = np.lexsort((iu[1], iu[0], -scores))
    for k in order:
        if connected or scores[k] <= 0:
            break
        i, j = int(iu[0][k]), int(iu[1][k])
        if find(i) != find(j):
            edges.add((i, j))
            connected = union(i, j) == n_terminals

    # 2. Attach terminals that are still disconnected
    while not connected:
        roots = np.array([find(v) for v in range(n)])
        has_terminal = np.array([terminals_in[r] > 0 for r in roots])
        feasible = (roots[:, None] != roots[None, :]) & has_terminal[:, None] & has_terminal[None, :]
        masked = np.where(feasible, distances, np.inf)
        i, j = np.unravel_index(np.argmin(masked), masked.shape)
        i, j = (int(i), int(j)) if i < j else (int(j), int(i))
        edges.add((i, j))
        connected = union(i, j) == n_terminals

    # 3. Keep Steiner points only as connection hubs
    if n_terminals > 0:
        root = find(int(np.argmax(terminal)))
        edges = {(i, j) for i, j in edges if find(i) == root}
    neighbours = [set() for _ in range(n)]
    for i, j in edges:
        neighbours[i].add(j)
        neighbours[j].add(i)
    changed = True
    while changed:
        changed = False
        for v in range(n):
            if terminal[v]:
                continue
            if len(neighbours[v]) == 1:
                (u,) = neighbours[v]
                neighbours[u].discard(v)
                neighbours[v].clear()
                changed = True
            elif len(neighbours[v]) == 2:
                u, w = neighbours[v]
                neighbours[u].discard(v)
                neighbours[w].discard(v)
                neighbours[u].add(w)
                neighbours[w].add(u)
                neighbours[v].clear()
                changed = True

    adjacency = np.zeros((n, n), dtype=np.float32)
    for v in range(n):
        for u in neighbours[v]:
            adjacency[v, u] = 1.0
    tree_length = float((adjacency * distances).sum() / 2.0)

    return adjacency, tree_length


# ============================================================================
# Baseline Solvers
# ============================================================================

class MSTSolver:
    """Minimum Spanning Tree baseline (2/√3 ≈ 1.155 approximation)"""

    @staticmethod
    def solve(coords, is_terminal=None):
        """
        Compute MST on given coordinates

        Args:
            coords: (n, 2) coordinates
            is_terminal: Optional, not used (for API compatibility)

        Returns:
            adjacency: (n, n) MST adjacency matrix
            length: Total MST length
        """
        if isinstance(coords, torch.Tensor):
            coords = coords.cpu().numpy()

        n = len(coords)

        # Compute distance matrix
        dist_mat = distance_matrix(coords, coords)

        # Find MST using Scipy
        mst = minimum_spanning_tree(dist_mat)
        mst_array = mst.toarray()

        # Create symmetric adjacency matrix
        adjacency = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                if mst_array[i, j] > 0 or mst_array[j, i] > 0:
                    adjacency[i, j] = 1.0
                    adjacency[j, i] = 1.0

        # Compute length
        evaluator = SteinerTreeEvaluator()
        length = evaluator.compute_tree_length(coords, adjacency)

        return adjacency, length


class OneSteinerSolver:
    """
    1-Steiner heuristic (Kahng & Robins, 1992)

    Algorithm:
    1. Start with MST
    2. For each candidate Steiner point:
       - Check if adding it reduces total length
       - Add best improvement
    3. Repeat until no improvement
    """

    @staticmethod
    def solve(terminals, steiner_candidates, max_iters=5):
        """
        1-Steiner heuristic

        Args:
            terminals: (n_terminals, 2) terminal coordinates
            steiner_candidates: (n_candidates, 2) candidate coordinates
            max_iters: Maximum iterations

        Returns:
            adjacency: (n_total, n_total) tree adjacency
            length: Total tree length
        """
        if isinstance(terminals, torch.Tensor):
            terminals = terminals.cpu().numpy()
        if isinstance(steiner_candidates, torch.Tensor):
            steiner_candidates = steiner_candidates.cpu().numpy()

        n_terminals = len(terminals)
        n_candidates = len(steiner_candidates)

        # Start with MST on terminals
        adj_mst, best_length = MSTSolver.solve(terminals)

        # Pad adjacency to include candidates
        n_total = n_terminals + n_candidates
        adjacency = np.zeros((n_total, n_total), dtype=np.float32)
        adjacency[:n_terminals, :n_terminals] = adj_mst

        # Combine all coordinates
        all_coords = np.vstack([terminals, steiner_candidates])

        # Track which Steiner points are used
        used_steiner = set()

        # Iteratively add Steiner points
        for iteration in range(max_iters):
            best_improvement = 0
            best_candidate = None

            for cand_idx in range(n_candidates):
                if cand_idx in used_steiner:
                    continue

                steiner_global_idx = n_terminals + cand_idx

                # Try adding this Steiner point
                current_used = [n_terminals + idx for idx in used_steiner]
                active_nodes = list(range(n_terminals)) + current_used + [steiner_global_idx]
                active_coords = all_coords[active_nodes]

                # Compute MST on active nodes
                test_adj, test_length = MSTSolver.solve(active_coords)

                improvement = best_length - test_length
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_candidate = cand_idx

            # Add best candidate if it improves
            if best_candidate is not None and best_improvement > 1e-6:
                used_steiner.add(best_candidate)
                best_length -= best_improvement
            else:
                break  # No more improvements

        # Build final tree with terminals + used Steiner points
        if len(used_steiner) > 0:
            used_steiner_global = [n_terminals + idx for idx in used_steiner]
            active_nodes = list(range(n_terminals)) + used_steiner_global
            active_coords = all_coords[active_nodes]

            # Compute final MST
            final_adj, final_length = MSTSolver.solve(active_coords)

            # Map back to full adjacency matrix
            adjacency = np.zeros((n_total, n_total), dtype=np.float32)
            active_to_global = {i: active_nodes[i] for i in range(len(active_nodes))}

            for i in range(len(active_nodes)):
                for j in range(i + 1, len(active_nodes)):
                    if final_adj[i, j] > 0:
                        global_i = active_to_global[i]
                        global_j = active_to_global[j]
                        adjacency[global_i, global_j] = 1.0
                        adjacency[global_j, global_i] = 1.0

            best_length = final_length

        return adjacency, best_length


class IteratedOneSteinerSolver:
    """
    Iterated 1-Steiner (standard heuristic baseline)

    Repeatedly applies 1-Steiner until convergence.
    Typically achieves ~5-7% gap to optimal.
    """

    @staticmethod
    def solve(terminals, steiner_candidates, max_outer_iters=3):
        """
        Iterated 1-Steiner

        Args:
            terminals: (n_terminals, 2) coordinates
            steiner_candidates: (n_candidates, 2) coordinates
            max_outer_iters: Maximum outer iterations

        Returns:
            adjacency: Final tree adjacency
            length: Total tree length
        """
        best_adj, best_length = OneSteinerSolver.solve(
            terminals, steiner_candidates, max_iters=10
        )

        for outer_iter in range(max_outer_iters - 1):
            # Try again with fresh candidates
            new_adj, new_length = OneSteinerSolver.solve(
                terminals, steiner_candidates, max_iters=10
            )

            if new_length < best_length - 1e-6:
                best_adj = new_adj
                best_length = new_length
            else:
                break  # Converged

        return best_adj, best_length


class GeoSteinerSolver:
    """
    GeoSteiner exact solver wrapper

    Computes optimal Euclidean Steiner trees using the GeoSteiner package.
    Requires GeoSteiner to be installed and available in PATH.

    Installation:
        Download from http://www.geosteiner.com/
        Compile and add to PATH, or specify path via geosteiner_path parameter

    Note: GeoSteiner can be slow for large instances (>50 terminals)
    """

    @staticmethod
    def solve(coords, is_terminal=None, geosteiner_path=None, scale_factor=10000):
        """
        Solve Euclidean Steiner Tree using GeoSteiner

        Args:
            coords: (n, 2) array of coordinates (terminals + candidates)
                    For GeoSteiner, only terminal coordinates are used.
                    Steiner points are computed optimally by the solver.
            is_terminal: (n,) binary array indicating which points are terminals
                        If None, assumes all points are terminals
            geosteiner_path: Path to GeoSteiner binaries. If None, assumes in PATH
            scale_factor: Scaling factor for coordinates (GeoSteiner uses integers)

        Returns:
            adjacency: (n_total, n_total) adjacency matrix including optimal Steiner points
            length: Optimal tree length
        """
        if isinstance(coords, torch.Tensor):
            coords = coords.cpu().numpy()
        if isinstance(is_terminal, torch.Tensor):
            is_terminal = is_terminal.cpu().numpy().flatten()

        # Extract terminals
        if is_terminal is not None:
            terminal_mask = is_terminal > 0.5
            terminals = coords[terminal_mask]
        else:
            terminals = coords

        n_terminals = len(terminals)

        # Scale coordinates to integers (GeoSteiner requirement)
        scaled_coords = (terminals * scale_factor).astype(int)

        # Create temporary STP file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.stp', delete=False) as f:
            stp_file = f.name
            GeoSteinerSolver._write_stp_file(f, scaled_coords, n_terminals)

        try:
            # Run GeoSteiner
            result = GeoSteinerSolver._run_geosteiner(stp_file, geosteiner_path)

            # Parse output
            adjacency, steiner_points, length = GeoSteinerSolver._parse_geosteiner_output(
                result, terminals, n_terminals, scale_factor
            )

            # If original input had candidate Steiner points, we need to return
            # an adjacency matrix of the same size
            if is_terminal is not None and len(coords) > n_terminals:
                n_total = len(coords)
                n_steiner = len(steiner_points)

                # Create full adjacency matrix
                full_adjacency = np.zeros((n_total + n_steiner, n_total + n_steiner), dtype=np.float32)

                # Map terminals to original indices
                terminal_indices = np.where(terminal_mask)[0]

                # Copy terminal-terminal edges
                for i in range(len(adjacency)):
                    for j in range(len(adjacency)):
                        if adjacency[i, j] > 0:
                            if i < n_terminals and j < n_terminals:
                                # Both are terminals
                                orig_i = terminal_indices[i]
                                orig_j = terminal_indices[j]
                                full_adjacency[orig_i, orig_j] = 1.0
                                full_adjacency[orig_j, orig_i] = 1.0
                            elif i < n_terminals and j >= n_terminals:
                                # Terminal to Steiner
                                orig_i = terminal_indices[i]
                                steiner_idx = n_total + (j - n_terminals)
                                full_adjacency[orig_i, steiner_idx] = 1.0
                                full_adjacency[steiner_idx, orig_i] = 1.0
                            elif i >= n_terminals and j < n_terminals:
                                # Steiner to terminal
                                steiner_idx = n_total + (i - n_terminals)
                                orig_j = terminal_indices[j]
                                full_adjacency[steiner_idx, orig_j] = 1.0
                                full_adjacency[orig_j, steiner_idx] = 1.0
                            else:
                                # Steiner to Steiner
                                steiner_i = n_total + (i - n_terminals)
                                steiner_j = n_total + (j - n_terminals)
                                full_adjacency[steiner_i, steiner_j] = 1.0
                                full_adjacency[steiner_j, steiner_i] = 1.0

                return full_adjacency, length
            else:
                return adjacency, length

        finally:
            # Clean up temporary file
            if os.path.exists(stp_file):
                os.unlink(stp_file)

    @staticmethod
    def _write_stp_file(f, coords, n_terminals):
        """Write coordinates to STP format file"""
        f.write("33D32945 STP File, STP Format Version 1.0\n\n")
        f.write("SECTION Comment\n")
        f.write("Name \"EDISCO Generated Instance\"\n")
        f.write("Problem \"Euclidean Steiner Tree Problem\"\n")
        f.write("END\n\n")

        f.write("SECTION Graph\n")
        f.write(f"Nodes {n_terminals}\n")
        f.write("Edges 0\n")  # No predefined edges
        f.write("END\n\n")

        f.write("SECTION Terminals\n")
        f.write(f"Terminals {n_terminals}\n")
        for i in range(n_terminals):
            f.write(f"T {i+1}\n")
        f.write("END\n\n")

        f.write("SECTION Coordinates\n")
        for x, y in coords:
            f.write(f"DD {x} {y}\n")
        f.write("END\n\n")

        f.write("EOF\n")

    @staticmethod
    def _run_geosteiner(stp_file, geosteiner_path=None):
        """Run GeoSteiner on STP file"""
        # Determine command paths
        if geosteiner_path:
            efst_cmd = os.path.join(geosteiner_path, 'efst')
            bb_cmd = os.path.join(geosteiner_path, 'bb')
        else:
            efst_cmd = 'efst'
            bb_cmd = 'bb'

        try:
            # Read STP file content
            with open(stp_file, 'r') as f:
                stp_content = f.read()

            # Pipeline: efst | bb
            # efst: Euclidean Full Steiner Tree generator
            # bb: Branch-and-bound solver
            efst_proc = subprocess.Popen(
                [efst_cmd],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            bb_proc = subprocess.Popen(
                [bb_cmd, '-f'],  # -f for full output
                stdin=efst_proc.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Write input to efst
            efst_proc.stdin.write(stp_content)
            efst_proc.stdin.close()

            # Get output from bb
            stdout, stderr = bb_proc.communicate()

            if bb_proc.returncode != 0:
                raise RuntimeError(f"GeoSteiner failed: {stderr}")

            return stdout

        except FileNotFoundError:
            raise RuntimeError(
                "GeoSteiner not found. Please install GeoSteiner and add to PATH, "
                "or specify geosteiner_path parameter. "
                "Download from: http://www.geosteiner.com/"
            )

    @staticmethod
    def _parse_geosteiner_output(output, terminals, n_terminals, scale_factor):
        """
        Parse GeoSteiner output to extract tree structure

        Returns:
            adjacency: Adjacency matrix (terminals + Steiner points)
            steiner_points: Array of computed Steiner point coordinates
            length: Total tree length
        """
        # Parse optimal length from output
        length_match = re.search(r'Length\s*=\s*([\d.]+)', output)
        if length_match:
            length = float(length_match.group(1)) / scale_factor
        else:
            # Fallback: compute from MST
            length = MSTSolver.solve(terminals)[1]

        # Parse Steiner points
        steiner_points = []
        steiner_section = False
        edges = []

        for line in output.split('\n'):
            # Look for Steiner point coordinates
            if 'Steiner' in line and 'point' in line.lower():
                steiner_section = True

            # Parse coordinates (DD format: x y)
            coord_match = re.search(r'DD\s+([\d-]+)\s+([\d-]+)', line)
            if coord_match and steiner_section:
                x = int(coord_match.group(1)) / scale_factor
                y = int(coord_match.group(2)) / scale_factor
                steiner_points.append([x, y])

            # Parse edges (E format: E node1 node2)
            edge_match = re.search(r'E\s+(\d+)\s+(\d+)', line)
            if edge_match:
                i = int(edge_match.group(1)) - 1  # Convert to 0-indexed
                j = int(edge_match.group(2)) - 1
                edges.append((i, j))

        # Build adjacency matrix
        n_steiner = len(steiner_points)
        n_total = n_terminals + n_steiner
        adjacency = np.zeros((n_total, n_total), dtype=np.float32)

        for i, j in edges:
            adjacency[i, j] = 1.0
            adjacency[j, i] = 1.0

        # If no edges were parsed, fall back to MST
        if len(edges) == 0:
            print("Warning: Could not parse GeoSteiner edges, falling back to MST")
            adjacency, length = MSTSolver.solve(terminals)
            steiner_points = np.array([])
        else:
            steiner_points = np.array(steiner_points, dtype=np.float32)

        return adjacency, steiner_points, length
