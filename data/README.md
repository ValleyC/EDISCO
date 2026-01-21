# Data Generation and Preparation for EDISCO

This folder contains scripts and instructions for generating training and evaluation data for EDISCO experiments.

## Traveling Salesman Problem (TSP)

### TSP-50 & TSP-100

Both the training and evaluation data of TSP-50 and TSP-100 are taken from [chaitjo/learning-tsp](https://github.com/chaitjo/learning-tsp).

For generating training data with EDISCO's reduced data requirements:

#### TSP-50
```bash
python -u generate_data.py \
  --min_nodes 50 \
  --max_nodes 50 \
  --num_samples 500000 \
  --batch_size 128 \
  --filename "data/tsp/tsp50_train_concorde.txt" \
  --seed 1234 \
  --solver "concorde"
```

#### TSP-100
```bash
python -u generate_data.py \
  --min_nodes 100 \
  --max_nodes 100 \
  --num_samples 500000 \
  --batch_size 128 \
  --filename "data/tsp/tsp100_train_concorde.txt" \
  --seed 1234 \
  --solver "concorde"
```

### TSP-500, TSP-1000, and TSP-10000

The evaluation data of TSP-500, TSP-1000, and TSP-10000 are taken from [Spider-scnu/TSP](https://github.com/Spider-scnu/TSP).

**Note**: EDISCO requires significantly less training data (33-50% reduction) compared to baseline methods due to its efficient equivariant architecture.

#### TSP-500
```bash
python -u generate_data.py \
  --min_nodes 500 \
  --max_nodes 500 \
  --num_samples 60000 \
  --batch_size 128 \
  --filename "data/tsp/tsp500_train_lkh.txt" \
  --seed 1234 \
  --solver "lkh3"
```

#### TSP-1000
```bash
python -u generate_data.py \
  --min_nodes 1000 \
  --max_nodes 1000 \
  --num_samples 30000 \
  --batch_size 64 \
  --filename "data/tsp/tsp1000_train_lkh.txt" \
  --seed 1234 \
  --solver "lkh3"
```

#### TSP-10000
```bash
python -u generate_data.py \
  --min_nodes 10000 \
  --max_nodes 10000 \
  --num_samples 3000 \
  --batch_size 8 \
  --filename "data/tsp/tsp10000_train_lkh.txt" \
  --seed 1234 \
  --solver "lkh3"
```

## TSPLIB Benchmark Instances

For evaluation on real-world TSP instances from TSPLIB:

1. Download TSPLIB instances from [TSPLIB95](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/)
2. Place `.tsp` files in `data/tsplib/`
3. Convert to the required format:

```bash
python -u convert_tsplib.py \
  --input_dir "data/tsplib/" \
  --output_dir "data/tsplib_processed/" \
  --format "edisco"
```

## Capacitated Vehicle Routing Problem (CVRP)

For CVRP data generation, we follow the standard protocol from the Attention Model paper.

### Installation

First, clone the Attention Model repository:
```bash
git clone https://github.com/wouterkool/attention-learn-to-route.git
cd attention-learn-to-route
```

### Generate CVRP Training Data

#### CVRP-20 (20 customers)
```bash
python generate_data.py \
  --problem cvrp \
  --name cvrp20_train \
  --dataset_size 500000 \
  --graph_size 20
```

#### CVRP-50 (50 customers)
```bash
python generate_data.py \
  --problem cvrp \
  --name cvrp50_train \
  --dataset_size 500000 \
  --graph_size 50
```

#### CVRP-100 (100 customers)
```bash
python generate_data.py \
  --problem cvrp \
  --name cvrp100_train \
  --dataset_size 200000 \
  --graph_size 100
```

### Generate CVRP Validation and Test Data

```bash
# Validation sets
for size in 20 50 100; do
  python generate_data.py \
    --problem cvrp \
    --name cvrp${size}_validation \
    --dataset_size 10000 \
    --graph_size $size \
    --seed 4321
done

# Test sets
for size in 20 50 100; do
  python generate_data.py \
    --problem cvrp \
    --name cvrp${size}_test \
    --dataset_size 10000 \
    --graph_size $size \
    --seed 1234
done
```
## Data Format

### TSP Format
Each line contains a TSP instance in the following format:
```
x1 y1 x2 y2 ... xn yn output tour_1 tour_2 ... tour_n
```
- Coordinates are normalized to [0, 1]
- The word "output" serves as a separator
- Tour indices are 1-indexed (1, 2, 3, ..., n)
- Example for TSP-20:
  ```
  0.191 0.622 0.437 0.785 ... 0.300 output 1 15 6 8 11 4 19 20 14 2 7 18 10 5 13 16 3 17 9 12 1
  ```

### CVRP Format
Each line contains a CVRP instance:
```
depot_x depot_y | x1 y1 d1 x2 y2 d2 ... xn yn dn | capacity | routes | total_distance
```
- First pair: depot coordinates
- Following triplets: customer coordinates and demands
- Routes: semicolon-separated sequences of customer indices
- Total distance: sum of all route distances


## Euclidean Steiner Tree Problem (ESTP)

For Euclidean Steiner Tree data generation:

### Steiner10, Steiner20, Steiner50

```bash
# Generate Steiner-10 (10 terminals + 10 candidates)
python -u generate_steiner_data.py \
  --problem_size 10 \
  --num_samples 10000 \
  --filename "steiner10_train.txt" \
  --solver "iterated_1steiner" \
  --seed 1234

python -u generate_steiner_data.py \
  --problem_size 10 \
  --num_samples 1000 \
  --filename "steiner10_valid.txt" \
  --solver "iterated_1steiner" \
  --seed 4321

python -u generate_steiner_data.py \
  --problem_size 10 \
  --num_samples 1000 \
  --filename "steiner10_test.txt" \
  --solver "iterated_1steiner" \
  --seed 5678

# Generate Steiner-20 (20 terminals + 20 candidates)
python -u generate_steiner_data.py \
  --problem_size 20 \
  --num_samples 10000 \
  --filename "steiner20_train.txt" \
  --solver "iterated_1steiner" \
  --seed 1234

python -u generate_steiner_data.py \
  --problem_size 20 \
  --num_samples 1000 \
  --filename "steiner20_valid.txt" \
  --solver "iterated_1steiner" \
  --seed 4321

python -u generate_steiner_data.py \
  --problem_size 20 \
  --num_samples 1000 \
  --filename "steiner20_test.txt" \
  --solver "iterated_1steiner" \
  --seed 5678

# Generate Steiner-50 (50 terminals + 50 candidates)
python -u generate_steiner_data.py \
  --problem_size 50 \
  --num_samples 10000 \
  --filename "steiner50_train.txt" \
  --solver "iterated_1steiner" \
  --seed 1234

python -u generate_steiner_data.py \
  --problem_size 50 \
  --num_samples 1000 \
  --filename "steiner50_valid.txt" \
  --solver "iterated_1steiner" \
  --seed 4321

python -u generate_steiner_data.py \
  --problem_size 50 \
  --num_samples 1000 \
  --filename "steiner50_test.txt" \
  --solver "iterated_1steiner" \
  --seed 5678
```

### Steiner Tree Data Format

Each line contains a Steiner Tree instance in the following format:
```
x1_t y1_t x2_t y2_t ... xn_t yn_t SEP x1_c y1_c x2_c y2_c ... xm_c ym_c output adj_00 adj_01 ... adj_nn
```
- First section (before "SEP"): terminal coordinates (n terminals)
- Second section (between "SEP" and "output"): candidate Steiner point coordinates (m candidates)
- Third section (after "output"): flattened (n+m)×(n+m) adjacency matrix representing the Steiner tree solution
- Coordinates are normalized to [0, 1]
- Adjacency matrix is binary (0 or 1), row-major order
- Example for Steiner-10 (10 terminals + 10 candidates, 20 total nodes):
  ```
  0.191 0.622 0.437 0.785 ... 0.300 SEP 0.543 0.112 0.891 0.445 ... 0.678 output 0 1 0 0 0 0 ... 0 1 0
  ```

**Ground Truth Solvers**:
- **Iterated 1-Steiner** (default): Fast heuristic with ~3-4% gap from optimal
- **GeoSteiner** (optional): Exact optimal solver for small instances (≤50 terminals)

### Using GeoSteiner for Optimal Ground Truth

For generating data with optimal solutions (small instances only):

1. **Install GeoSteiner**:
   ```bash
   # Download from http://www.geosteiner.com/
   wget http://www.geosteiner.com/geosteiner-5.3.tar.gz
   tar -xzf geosteiner-5.3.tar.gz
   cd geosteiner-5.3
   ./configure
   make
   sudo make install
   ```

2. **Generate data with GeoSteiner**:
   ```bash
   python -u generate_steiner_data.py \
     --problem_size 10 \
     --num_samples 1000 \
     --filename "steiner10_train_optimal.txt" \
     --solver "geosteiner" \
     --seed 1234
   ```

**Note**: GeoSteiner provides truly optimal solutions but can be slow for instances >20 terminals. For larger instances (50+), use `iterated_1steiner` which provides high-quality solutions efficiently


## Maximum Independent Set (MIS)

**Note**: MIS is NOT a geometric problem, so EDISCO uses the standard GNN encoder (not EGNN) for MIS. The continuous-time diffusion still provides benefits.

### MIS Benchmark Framework

We use the MIS benchmark framework from [ICLR 2022](https://openreview.net/forum?id=mk0HzdqY7i1). The framework is located in `data/mis-benchmark-framework/`.

### Setup

```bash
cd data/mis-benchmark-framework
conda env create -f environment.yml
conda activate mis-benchmark
bash setup_bm_env.sh
```

### Generate Random Graphs (Erdős-Rényi)

The standard benchmark uses Erdős-Rényi (ER) random graphs with varying sizes:

```bash
cd data/mis-benchmark-framework

# Generate ER graphs with 700-800 nodes (SATLIB-like)
python main.py gendata random . ../mis/er_700_800_train \
  --model er \
  --min_n 700 \
  --max_n 800 \
  --num_graphs 4000 \
  --er_p 0.15 \
  --gen_labels

python main.py gendata random . ../mis/er_700_800_test \
  --model er \
  --min_n 700 \
  --max_n 800 \
  --num_graphs 500 \
  --er_p 0.15 \
  --gen_labels
```

### Generate Smaller Graphs for Quick Experiments

```bash
# ER graphs with 100 nodes
python main.py gendata random . ../mis/er_100_train \
  --model er \
  --min_n 100 \
  --max_n 100 \
  --num_graphs 10000 \
  --er_p 0.15 \
  --gen_labels

python main.py gendata random . ../mis/er_100_test \
  --model er \
  --min_n 100 \
  --max_n 100 \
  --num_graphs 500 \
  --er_p 0.15 \
  --gen_labels

# ER graphs with 200-300 nodes
python main.py gendata random . ../mis/er_200_300_train \
  --model er \
  --min_n 200 \
  --max_n 300 \
  --num_graphs 5000 \
  --er_p 0.15 \
  --gen_labels
```

### Other Random Graph Models

```bash
# Barabási-Albert (BA) graphs
python main.py gendata random . ../mis/ba_500_train \
  --model ba \
  --min_n 500 \
  --max_n 500 \
  --num_graphs 4000 \
  --ba_m 4 \
  --gen_labels

# Watts-Strogatz (WS) small-world graphs
python main.py gendata random . ../mis/ws_500_train \
  --model ws \
  --min_n 500 \
  --max_n 500 \
  --num_graphs 4000 \
  --ws_k 4 \
  --ws_p 0.5 \
  --gen_labels

# Holme-Kim (HK) powerlaw cluster graphs
python main.py gendata random . ../mis/hk_500_train \
  --model hk \
  --min_n 500 \
  --max_n 500 \
  --num_graphs 4000 \
  --hk_m 4 \
  --hk_p 0.5 \
  --gen_labels
```

### Generate SATLIB Graphs

SATLIB graphs are derived from SAT instances and are commonly used for MIS benchmarking:

```bash
# Download and convert SATLIB instances
python main.py gendata sat <satlib_cnf_folder> ../mis/satlib \
  --gen_labels
```

### MIS Data Format

MIS data is stored as NetworkX graphs in `.gpickle` format:
- **Nodes**: Each node has an optional `label` attribute (1 if in MIS, 0 otherwise)
- **Edges**: Undirected edges defining the graph structure
- **Loading**: Use `pickle.load()` or `pickle5.load()` to read

Example loading:
```python
import pickle
import networkx as nx

with open("graph.gpickle", "rb") as f:
    G = pickle.load(f)

num_nodes = G.number_of_nodes()
edges = list(G.edges())
labels = [G.nodes[i].get('label', 0) for i in range(num_nodes)]
```

### Training EDISCO on MIS

EDISCO uses continuous-time diffusion with ODE solvers (PNDM, DEIS, etc.) for MIS.

#### Training on SATLIB graphs

```bash
python edisco/train.py \
  --task mis \
  --wandb_logger_name "edisco_mis_satlib" \
  --diffusion_type categorical \
  --do_train \
  --do_test \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "/path/to/data" \
  --training_split "mis/satlib_train/*.gpickle" \
  --validation_split "mis/satlib_test/*.gpickle" \
  --test_split "mis/satlib_test/*.gpickle" \
  --batch_size 16 \
  --num_epochs 50 \
  --solver_type pndm \
  --solver_steps 50 \
  --time_schedule linear \
  --adaptive_mixing \
  --use_activation_checkpoint
```

#### Training on ER-[700-800] graphs

```bash
python edisco/train.py \
  --task mis \
  --wandb_logger_name "edisco_mis_er" \
  --diffusion_type categorical \
  --do_train \
  --do_test \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "/path/to/data" \
  --training_split "mis/er_700_800_train/*.gpickle" \
  --validation_split "mis/er_700_800_test/*.gpickle" \
  --test_split "mis/er_700_800_test/*.gpickle" \
  --batch_size 8 \
  --num_epochs 50 \
  --solver_type pndm \
  --solver_steps 50 \
  --time_schedule linear \
  --adaptive_mixing \
  --use_activation_checkpoint
```

#### Evaluation with Different Solvers

```bash
# Using PNDM solver (default)
python edisco/train.py \
  --task mis \
  --do_test \
  --storage_path "/path/to/data" \
  --test_split "mis/satlib_test/*.gpickle" \
  --solver_type pndm \
  --solver_steps 50 \
  --parallel_sampling 4 \
  --ckpt_path "/path/to/checkpoint.ckpt"

# Using DEIS solver
python edisco/train.py \
  --task mis \
  --do_test \
  --storage_path "/path/to/data" \
  --test_split "mis/satlib_test/*.gpickle" \
  --solver_type deis \
  --solver_steps 50 \
  --ckpt_path "/path/to/checkpoint.ckpt"
```

### EDISCO MIS Settings Reference

| Setting | Default | Description |
|---------|---------|-------------|
| `solver_type` | pndm | ODE solver (euler, ddim, pndm, dpm2, deis, rk4, heun) |
| `solver_steps` | 50 | Number of solver steps |
| `time_schedule` | linear | Time schedule (linear, cosine, quadratic) |
| `adaptive_mixing` | True | Use adaptive mixing strategy |
| `beta_min` | 0.1 | Minimum noise rate |
| `beta_max` | 1.5 | Maximum noise rate |
| `n_layers` | 12 | Number of GNN layers |
| `hidden_dim` | 256 | Hidden dimension |

### Pre-generated Data

You can download pre-generated MIS data from:
- [HPI OwnCloud](https://owncloud.hpi.de/s/cv6szEJtSs8UGju) - Random graphs with labels and trained models
- [Backup location](https://mboether.com/paper-models-randomgraphs.zip)

