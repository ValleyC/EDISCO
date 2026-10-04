"""Convert TSPLIB instances to the EDISCO TSP text format.

By default the 29 two-dimensional Euclidean instances with 51 to 200 nodes of
the TSPLIB transfer experiment are converted. For every instance the script

1. reads the node coordinates of `<name>.tsp` (fetched from TSPLIB95 with
   --download when the file is missing),
2. rescales them into the unit square with one common factor for both axes,
   which preserves the instance up to a similarity transformation,
3. takes the reference tour from `<name>.opt.tour` when it is available and
   computes it with LKH-3 on the original file otherwise,
4. checks the reference tour against the published optimal length under the
   TSPLIB metric (Euclidean distances rounded to the nearest integer).

Output (in --output_dir):
    <name>.txt      one instance:  x1 y1 ... xn yn output t1 ... tn t1
                    (tour indices are 1-based and the tour is closed)
    tsplib.txt      all converted instances, one per line
    metadata.json   size, reference length and optimality check per instance
"""

import argparse
import gzip
import json
import os
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request

import numpy as np

TSPLIB_URL = "http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/{}"

# Published optimal tour lengths under the TSPLIB metric.
OPTIMAL_LENGTHS = {
    'eil51': 426, 'berlin52': 7542, 'st70': 675, 'eil76': 538, 'pr76': 108159,
    'rat99': 1211, 'kroA100': 21282, 'kroB100': 22141, 'kroC100': 20749,
    'kroD100': 21294, 'kroE100': 22068, 'rd100': 7910, 'eil101': 629,
    'lin105': 14379, 'pr107': 44303, 'pr124': 59030, 'bier127': 118282,
    'ch130': 6110, 'pr136': 96772, 'pr144': 58537, 'ch150': 6528,
    'kroA150': 26524, 'kroB150': 26130, 'pr152': 73682, 'u159': 42080,
    'rat195': 2323, 'd198': 15780, 'kroA200': 29368, 'kroB200': 29437,
}


def download(name, suffix, directory):
    """Fetch `<name><suffix>` from TSPLIB95, which stores files gzip-compressed."""
    target = os.path.join(directory, name + suffix)
    if os.path.exists(target):
        return True
    for extension in ('.gz', ''):
        try:
            with urllib.request.urlopen(TSPLIB_URL.format(name + suffix + extension), timeout=60) as response:
                payload = response.read()
            if extension:
                payload = gzip.decompress(payload)
        except (urllib.error.URLError, OSError):
            continue
        with open(target, 'wb') as f:
            f.write(payload)
        return True
    return False


def read_tsp(path):
    """Node coordinates (n, 2) of a TSPLIB instance with EUC_2D edge weights."""
    header, coords, in_coords = {}, [], False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if in_coords:
                parts = line.split()
                if len(parts) < 3:  # EOF or the next section
                    break
                coords.append((float(parts[1]), float(parts[2])))
            elif line.upper().startswith('NODE_COORD_SECTION'):
                in_coords = True
            elif ':' in line:
                key, value = line.split(':', 1)
                header[key.strip().upper()] = value.strip()
    weight_type = header.get('EDGE_WEIGHT_TYPE', 'EUC_2D').upper()
    if weight_type != 'EUC_2D':
        raise ValueError(f"only EUC_2D instances are supported, got {weight_type}")
    coords = np.array(coords, dtype=np.float64)
    if 'DIMENSION' in header and len(coords) != int(header['DIMENSION']):
        raise ValueError(f"expected {header['DIMENSION']} nodes, read {len(coords)}")
    return coords


def read_tour(path, num_nodes):
    """0-based node order of a TSPLIB tour file."""
    with open(path) as f:
        tokens = f.read().split('TOUR_SECTION')[1].split()
    tour = []
    for token in tokens:
        if token in ('-1', 'EOF'):
            break
        tour.append(int(token) - 1)
    if sorted(tour) != list(range(num_nodes)):
        raise ValueError(f"{path} is not a tour over {num_nodes} nodes")
    return tour


def solve_lkh(lkh_path, tsp_path, num_nodes, runs=10, seed=1234):
    """Solve the original TSPLIB file with LKH-3 in its default configuration."""
    workdir = tempfile.mkdtemp(prefix="lkh_")
    try:
        tour_file = os.path.join(workdir, "problem.tour")
        param_file = os.path.join(workdir, "problem.par")
        with open(param_file, "w") as f:
            f.write(f"PROBLEM_FILE = {os.path.abspath(tsp_path)}\nTOUR_FILE = {tour_file}\n"
                    f"RUNS = {runs}\nSEED = {seed}\nTRACE_LEVEL = 0\n")
        subprocess.run([lkh_path, param_file], check=True, stdin=subprocess.DEVNULL,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return read_tour(tour_file, num_nodes)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def tsplib_length(coords, tour):
    """Tour length with Euclidean distances rounded to the nearest integer."""
    closed = coords[list(tour) + [tour[0]]]
    distances = np.linalg.norm(closed[1:] - closed[:-1], axis=1)
    return int(np.floor(distances + 0.5).sum())


def normalize(coords):
    """Translate and scale the coordinates into the unit square with one common factor."""
    shifted = coords - coords.min(axis=0)
    return shifted / shifted.max()


def format_instance(coords, tour):
    points = " ".join(f"{x} {y}" for x, y in coords)
    closed = " ".join(str(node + 1) for node in list(tour) + [tour[0]])
    return f"{points} output {closed}"


def convert_instance(name, opts):
    """Return (line, metadata) for one instance, or raise if it cannot be converted."""
    tsp_path = os.path.join(opts.input_dir, name + ".tsp")
    if not os.path.exists(tsp_path) and not (opts.download and download(name, ".tsp", opts.input_dir)):
        raise FileNotFoundError(f"{tsp_path} not found (pass --download to fetch it)")
    coords = read_tsp(tsp_path)
    num_nodes = len(coords)
    optimal = OPTIMAL_LENGTHS.get(name)

    # Candidate reference tours: the published optimal tour, then LKH-3.
    if opts.download:
        download(name, ".opt.tour", opts.input_dir)
    candidates = []
    tour_path = os.path.join(opts.input_dir, name + ".opt.tour")
    if os.path.exists(tour_path):
        tour = read_tour(tour_path, num_nodes)
        candidates.append((tsplib_length(coords, tour), "opt.tour", tour))
    if (not candidates or candidates[0][0] != optimal) and opts.lkh_path is not None:
        tour = solve_lkh(opts.lkh_path, tsp_path, num_nodes, opts.lkh_runs)
        candidates.append((tsplib_length(coords, tour), "lkh", tour))
    if not candidates:
        raise RuntimeError(f"no reference tour: provide {name}.opt.tour or --lkh_path")
    length, source, tour = min(candidates, key=lambda candidate: candidate[0])

    metadata = {
        'num_nodes': num_nodes,
        'reference_length': length,
        'reference_source': source,
        'optimal_length': optimal,
        'reference_is_optimal': None if optimal is None else length == optimal,
    }
    return format_instance(normalize(coords), tour), metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert TSPLIB instances to the EDISCO TSP format")
    parser.add_argument("--input_dir", type=str, default="data/tsplib",
                        help="Directory with <name>.tsp (and optional <name>.opt.tour) files")
    parser.add_argument("--output_dir", type=str, default="data/tsplib_processed")
    parser.add_argument("--instances", type=str, nargs="+", default=None,
                        help="Instance names (default: the 29 benchmark instances)")
    parser.add_argument("--download", action="store_true",
                        help="Fetch missing files from TSPLIB95")
    parser.add_argument("--lkh_path", type=str, default=None,
                        help="LKH-3 executable used when no optimal tour file is available")
    parser.add_argument("--lkh_runs", type=int, default=10, help="LKH-3 RUNS (LKH-3 default)")
    opts = parser.parse_args()

    if opts.lkh_path is not None:
        opts.lkh_path = os.path.expanduser(opts.lkh_path)
        if shutil.which(opts.lkh_path) is None:
            raise FileNotFoundError(f"LKH-3 executable not found: {opts.lkh_path}")
    os.makedirs(opts.input_dir, exist_ok=True)
    os.makedirs(opts.output_dir, exist_ok=True)

    lines, metadata, failed = [], {}, {}
    for name in opts.instances or list(OPTIMAL_LENGTHS):
        try:
            line, info = convert_instance(name, opts)
        except Exception as error:  # report and continue with the remaining instances
            failed[name] = str(error)
            print(f"{name}: FAILED ({error})")
            continue
        with open(os.path.join(opts.output_dir, name + ".txt"), "w") as f:
            f.write(line + "\n")
        lines.append(line)
        metadata[name] = info
        status = {True: "optimal", False: "NOT optimal", None: "optimum unknown"}[info['reference_is_optimal']]
        print(f"{name}: {info['num_nodes']} nodes, reference length {info['reference_length']} "
              f"from {info['reference_source']} ({status})")

    with open(os.path.join(opts.output_dir, "tsplib.txt"), "w") as f:
        f.write("".join(line + "\n" for line in lines))
    with open(os.path.join(opts.output_dir, "metadata.json"), "w") as f:
        json.dump({'instances': metadata, 'failed': failed}, f, indent=2)
    print(f"Converted {len(lines)} instances to {opts.output_dir}" +
          (f"; failed: {', '.join(failed)}" if failed else ""))
