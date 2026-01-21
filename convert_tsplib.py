#!/usr/bin/env python3
"""
Convert TSPLIB instances to EDISCO format
Handles downloading, parsing, and converting TSPLIB benchmark instances

Output format (per file): x1 y1 x2 y2 ... xn yn output tour_1 tour_2 ... tour_n tour_1
where coordinates are normalized to [0,1] and tour indices are 1-indexed,
and the tour is explicitly closed by repeating the first index at the end.
"""

import os
import numpy as np
import urllib.request
import argparse
from typing import List, Optional
import json
from tqdm import tqdm
import ssl

# Create SSL context for HTTPS downloads (avoid certificate issues)
ssl._create_default_https_context = ssl._create_unverified_context


class TSPLIBConverter:
    """Converter for TSPLIB instances to EDISCO format"""

    # Known optimal values for common TSPLIB instances
    OPTIMAL_LENGTHS = {
        'eil51': 426.0,
        'berlin52': 7542.0,
        'st70': 675.0,
        'eil76': 538.0,
        'pr76': 108159.0,
        'rat99': 1211.0,
        'kroA100': 21282.0,
        'kroB100': 22141.0,
        'kroC100': 20749.0,
        'kroD100': 21294.0,
        'kroE100': 22068.0,
        'rd100': 7910.0,
        'eil101': 629.0,
        'lin105': 14379.0,
        'pr107': 44303.0,
        'pr124': 59030.0,
        'bier127': 118282.0,
        'ch130': 6110.0,
        'pr136': 96772.0,
        'pr144': 58537.0,
        'ch150': 6528.0,
        'kroA150': 26524.0,
        'kroB150': 26130.0,
        'pr152': 73682.0,
        'u159': 42080.0,
        'rat195': 2323.0,
        'd198': 15780.0,
        'kroA200': 29368.0,
        'kroB200': 29437.0,
    }

    def __init__(self, input_dir='data/tsplib', output_dir='data/tsplib_processed', download=True):
        """
        Initialize converter

        Args:
            input_dir: Directory containing .tsp files
            output_dir: Directory to save converted files
            download: Whether to download missing files
        """
        self.input_dir = os.path.normpath(input_dir)
        self.output_dir = os.path.normpath(output_dir)
        self.download = download

        # Create directories if they don't exist
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"Input directory: {os.path.abspath(self.input_dir)}")
        print(f"Output directory: {os.path.abspath(self.output_dir)}")

    def download_instance(self, name: str) -> bool:
        """
        Download TSPLIB instance from repository

        Args:
            name: Instance name (e.g., 'eil51')

        Returns:
            Success status
        """
        tsp_path = os.path.join(self.input_dir, f"{name}.tsp")
        tsp_path = os.path.normpath(tsp_path)

        # Check if already exists
        if os.path.exists(tsp_path) and os.path.getsize(tsp_path) > 100:
            print(f"File already exists: {tsp_path}")
            return True

        # Try multiple repositories
        urls = [
            f"http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/{name}.tsp",
            f"http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/{name}.tsp.gz",
        ]

        for url in urls:
            try:
                if url.endswith('.gz'):
                    # Download as .gz file first
                    gz_path = os.path.join(self.input_dir, f"{name}.tsp.gz")
                    print(f"Downloading {name}.tsp.gz...")
                    urllib.request.urlretrieve(url, gz_path)

                    # Check if gz file was downloaded properly
                    if not os.path.exists(gz_path) or os.path.getsize(gz_path) < 100:
                        print(f"Downloaded file too small or missing: {gz_path}")
                        if os.path.exists(gz_path):
                            os.remove(gz_path)
                        continue

                    # Extract gzip file
                    import gzip
                    print(f"Extracting {gz_path} to {tsp_path}...")
                    with gzip.open(gz_path, 'rb') as f_in:
                        with open(tsp_path, 'wb') as f_out:
                            content = f_in.read()
                            f_out.write(content)
                            print(f"Extracted {len(content)} bytes")
                    os.remove(gz_path)

                else:
                    print(f"Downloading {name}.tsp...")
                    urllib.request.urlretrieve(url, tsp_path)

                # Verify the file exists and is valid
                if os.path.exists(tsp_path):
                    file_size = os.path.getsize(tsp_path)
                    print(f"Downloaded file size: {file_size} bytes")
                    if file_size > 100:  # Basic sanity check
                        print(f"Successfully downloaded {name}.tsp")
                        return True
                    else:
                        print(f"Downloaded file too small: {tsp_path}")
                        os.remove(tsp_path)
                        continue

            except Exception as e:
                print(f"Error downloading from {url}: {str(e)}")
                if os.path.exists(tsp_path):
                    os.remove(tsp_path)
                continue

        print(f"Failed to download {name}.tsp from all sources")
        return False

    def parse_tsp_file(self, filepath: str) -> Optional[np.ndarray]:
        """
        Parse TSP file to extract coordinates

        Args:
            filepath: Path to .tsp file

        Returns:
            Numpy array of coordinates or None if parsing fails
        """
        coords = []
        edge_weight_type = None
        dimension = None

        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Parse header
        for line in lines:
            line = line.strip()
            if line.upper().startswith('DIMENSION'):
                try:
                    dimension = int(line.split(':')[1].strip())
                except Exception:
                    # some files use space separator
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            dimension = int(parts[-1])
                        except Exception:
                            pass
            elif line.upper().startswith('EDGE_WEIGHT_TYPE'):
                # Keep original case value (EUC_2D, GEO, etc.)
                try:
                    edge_weight_type = line.split(':', 1)[1].strip()
                except Exception:
                    parts = line.split()
                    if len(parts) >= 2:
                        edge_weight_type = parts[-1].strip()

        # Parse coordinates based on format
        if edge_weight_type is not None and edge_weight_type.upper() in ('EUC_2D', 'CEIL_2D'):
            coords = self._parse_euc2d(lines)
        elif edge_weight_type is not None and edge_weight_type.upper() == 'GEO':
            coords = self._parse_geo(lines)
        elif edge_weight_type is not None and edge_weight_type.upper() == 'ATT':
            coords = self._parse_att(lines)
        elif edge_weight_type is not None and edge_weight_type.upper() == 'EXPLICIT':
            print(f"Skipping EXPLICIT edge weight type (matrix-based)")
            return None
        else:
            # Try default EUC_2D parsing
            coords = self._parse_euc2d(lines)

        if coords and dimension is not None and len(coords) == dimension:
            return np.array(coords, dtype=np.float64)
        elif coords:
            if dimension is not None:
                print(f"Warning: Expected {dimension} nodes but parsed {len(coords)}")
            return np.array(coords, dtype=np.float64)
        else:
            return None

    def _parse_euc2d(self, lines: List[str]) -> List[List[float]]:
        """Parse EUC_2D format coordinates"""
        coords = []
        reading_coords = False

        for line in lines:
            raw = line
            line = line.strip()
            if line.upper().startswith('NODE_COORD_SECTION'):
                reading_coords = True
                continue
            elif line == 'EOF' or line.upper().startswith('DISPLAY_DATA_SECTION'):
                break
            elif reading_coords and line:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        # Format: node_id x y
                        x, y = float(parts[1]), float(parts[2])
                        coords.append([x, y])
                    except ValueError:
                        # skip non-numeric lines
                        continue

        return coords

    def _parse_geo(self, lines: List[str]) -> List[List[float]]:
        """Parse GEO format (geographical coordinates)"""
        coords = []
        reading_coords = False

        for line in lines:
            line = line.strip()
            if line.upper().startswith('NODE_COORD_SECTION'):
                reading_coords = True
                continue
            elif line == 'EOF':
                break
            elif reading_coords and line:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        # Convert geographical to Euclidean
                        lat = float(parts[1])
                        lon = float(parts[2])

                        # Convert degrees.minutes to radians
                        lat_deg = int(lat)
                        lat_min = lat - lat_deg
                        lat_rad = np.pi * (lat_deg + 5.0 * lat_min / 3.0) / 180.0

                        lon_deg = int(lon)
                        lon_min = lon - lon_deg
                        lon_rad = np.pi * (lon_deg + 5.0 * lon_min / 3.0) / 180.0

                        # Convert to Euclidean coordinates (approx)
                        x = 6378.388 * np.cos(lat_rad) * np.cos(lon_rad)
                        y = 6378.388 * np.cos(lat_rad) * np.sin(lon_rad)

                        coords.append([x, y])
                    except ValueError:
                        continue

        return coords

    def _parse_att(self, lines: List[str]) -> List[List[float]]:
        """Parse ATT format (pseudo-Euclidean)"""
        # ATT uses the same coordinate section layout as EUC_2D (node_id x y)
        return self._parse_euc2d(lines)

    def parse_optimal_tour(self, name: str) -> Optional[List[int]]:
        """
        Parse optimal tour from .opt.tour file if available

        Args:
            name: Instance name

        Returns:
            List of node indices (0-indexed) in optimal tour order
        """
        tour_path = os.path.join(self.input_dir, f"{name}.opt.tour")

        if not os.path.exists(tour_path):
            # Try to download
            url = f"http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/{name}.opt.tour"
            try:
                urllib.request.urlretrieve(url, tour_path)
            except Exception:
                return None

        if os.path.exists(tour_path):
            tour = []
            reading_tour = False

            with open(tour_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.upper().startswith('TOUR_SECTION'):
                        reading_tour = True
                        continue
                    elif line == '-1' or line == 'EOF':
                        break
                    elif reading_tour and line:
                        try:
                            # Convert to 0-indexed
                            node = int(line) - 1
                            if node >= 0:
                                tour.append(node)
                        except ValueError:
                            continue

            return tour if tour else None

        return None

    def compute_tour_length(self, coords: np.ndarray, tour: List[int]) -> float:
        """
        Compute tour length for given coordinates and tour

        Args:
            coords: Node coordinates
            tour: Order of nodes in tour

        Returns:
            Total tour length
        """
        length = 0.0
        n = len(tour)
        for i in range(n):
            current = tour[i]
            next_node = tour[(i + 1) % n]
            dist = np.linalg.norm(coords[current] - coords[next_node])
            length += dist
        return length

    def normalize_coordinates(self, coords: np.ndarray) -> np.ndarray:
        """
        Normalize coordinates to [0, 1] range

        Args:
            coords: Original coordinates

        Returns:
            Normalized coordinates
        """
        min_vals = coords.min(axis=0)
        max_vals = coords.max(axis=0)
        range_vals = max_vals - min_vals

        # Avoid division by zero
        range_vals[range_vals == 0] = 1.0

        normalized = (coords - min_vals) / range_vals
        return normalized

    def convert_instance(self, name: str) -> bool:
        """
        Convert a single TSPLIB instance to EDISCO format

        Args:
            name: Instance name

        Returns:
            Success status
        """
        # Download if needed
        if self.download:
            if not self.download_instance(name):
                print(f"Failed to download {name}")
                return False

        # Parse TSP file with proper path handling
        tsp_path = os.path.join(self.input_dir, f"{name}.tsp")
        tsp_path = os.path.normpath(tsp_path)  # Normalize path for OS

        if not os.path.exists(tsp_path):
            print(f"File not found: {tsp_path}")
            return False

        coords = self.parse_tsp_file(tsp_path)
        if coords is None:
            print(f"Failed to parse {name}.tsp")
            return False

        # Normalize coordinates
        coords_normalized = self.normalize_coordinates(coords)

        # Get optimal tour and length
        optimal_tour = self.parse_optimal_tour(name)
        optimal_length = self.OPTIMAL_LENGTHS.get(name, None)

        # If we have the tour but not the length, compute it
        if optimal_tour and optimal_length is None:
            optimal_length = self.compute_tour_length(coords, optimal_tour)
            optimal_length = round(optimal_length)  # Round for TSPLIB compatibility

        # Create EDISCO format string
        # Format: x1 y1 x2 y2 ... xn yn output tour_indices (1-indexed)
        # and explicitly close the tour by repeating the first index at the end.

        # Coordinates (normalized) - flatten as "x1 y1 x2 y2 ..."
        # Use Python's default float->str conversion to be consistent with generator style.
        coord_flat = []
        for (x, y) in coords_normalized:
            coord_flat.append(str(float(x)))
            coord_flat.append(str(float(y)))
        coord_str = ' '.join(coord_flat)

        n_nodes = coords_normalized.shape[0]

        # Tour (1-indexed)
        if optimal_tour:
            tour_list = [int(idx) + 1 for idx in optimal_tour]  # convert to 1-indexed ints
        else:
            tour_list = list(range(1, n_nodes + 1))

        # Close the tour by repeating the first index at the end
        if len(tour_list) > 0:
            tour_closed = tour_list + [tour_list[0]]
        else:
            tour_closed = tour_list

        tour_str = ' '.join(map(str, tour_closed))

        # Final output line (single-line per file)
        output_line = f"{coord_str} output {tour_str}"

        # Save to file with proper path handling
        output_path = os.path.join(self.output_dir, f"{name}.txt")
        output_path = os.path.normpath(output_path)  # Normalize path for OS

        with open(output_path, 'w') as f:
            f.write(output_line + '\n')

        print(f"Converted {name}: {n_nodes} cities, optimal length: {optimal_length}")
        return True

    def convert_all(self, instance_names: Optional[List[str]] = None):
        """
        Convert multiple TSPLIB instances

        Args:
            instance_names: List of instance names, or None for all known instances
        """
        if instance_names is None:
            # Use all known instances
            instance_names = list(self.OPTIMAL_LENGTHS.keys())

        success_count = 0
        failed_instances = []

        print(f"Converting {len(instance_names)} TSPLIB instances...")
        print("=" * 70)

        for name in tqdm(instance_names, desc="Converting"):
            if self.convert_instance(name):
                success_count += 1
            else:
                failed_instances.append(name)

        print("=" * 70)
        print(f"Successfully converted: {success_count}/{len(instance_names)}")

        if failed_instances:
            print(f"Failed instances: {', '.join(failed_instances)}")

        # Save metadata
        metadata = {
            'total_instances': len(instance_names),
            'successful': success_count,
            'failed': failed_instances,
            'optimal_lengths': {
                name: self.OPTIMAL_LENGTHS.get(name, None)
                for name in instance_names
                if name not in failed_instances
            }
        }

        metadata_path = os.path.join(self.output_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Metadata saved to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert TSPLIB instances to EDISCO format'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default='data/tsplib',
        help='Directory containing .tsp files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/tsplib_processed',
        help='Directory to save converted files'
    )
    parser.add_argument(
        '--instances',
        type=str,
        nargs='+',
        default=None,
        help='Specific instances to convert (e.g., eil51 berlin52)'
    )
    parser.add_argument(
        '--download',
        action='store_true',
        default=True,
        help='Download missing instances'
    )
    parser.add_argument(
        '--no_download',
        action='store_false',
        dest='download',
        help='Do not download missing instances'
    )
    parser.add_argument(
        '--format',
        type=str,
        default='edisco',
        choices=['edisco', 'difusco'],
        help='Output format (edisco or difusco compatible)'
    )

    args = parser.parse_args()

    # Initialize converter
    converter = TSPLIBConverter(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        download=args.download
    )

    # Common benchmark instances used in papers
    if args.instances is None:
        # Default to the 29 benchmark instances from the evaluation code
        instance_names = [
            'eil51', 'berlin52', 'st70', 'eil76', 'pr76',
            'rat99', 'kroA100', 'kroB100', 'kroC100', 'kroD100', 'kroE100',
            'rd100', 'eil101', 'lin105', 'pr107', 'pr124',
            'bier127', 'ch130', 'pr136', 'pr144', 'ch150',
            'kroA150', 'kroB150', 'pr152', 'u159',
            'rat195', 'd198', 'kroA200', 'kroB200'
        ]
    else:
        instance_names = args.instances

    # Convert instances
    converter.convert_all(instance_names)

    print("\nConversion complete!")
    print(f"Converted files saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
