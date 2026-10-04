# Data

Datasets for all EDISCO experiments: where to obtain the public sets, how to
generate the others, and the file formats. Commands are run from the repository
root. Paths passed to `edisco/train.py` are relative to its `--storage_path`.

| Script | Purpose |
|--------|---------|
| `generate_tsp_data.py` | TSP instances (uniform, cluster, explosion, implosion, gaussian) labelled with LKH-3, Concorde or Farthest Insertion |
| `generate_cvrp_data.py` | CVRP instances labelled with HGS-CVRP |
| `generate_steiner_data.py` | Euclidean Steiner tree instances labelled with iterated 1-Steiner or GeoSteiner |
| `convert_tsplib.py` | TSPLIB instances converted to the TSP text format |
| `mis-benchmark-framework/` | MIS graph generation and KaMIS labelling ([Böther et al., 2022](https://openreview.net/forum?id=mk0HzdqY7i1)) |

## Traveling Salesman Problem (TSP)

All TSP instances have coordinates drawn uniformly from the unit square. The
layout below is the one used by [reproducing_scripts.md](../reproducing_scripts.md).

```
data/tsp/train/   tsp50_train_concorde_500k.txt   tsp100_train_concorde_500k.txt
                  tsp500_train_lkh_60k.txt   tsp1000_train_lkh_30k.txt   tsp10000_train_lkh_3k.txt
data/tsp/valid/   tsp50_valid_concorde_1280.txt   tsp100_valid_concorde_1280.txt
                  tsp500_valid_lkh_128.txt   tsp1000_valid_lkh_128.txt   tsp10000_valid_lkh_16.txt
data/tsp/test/    tsp50_test_concorde.txt   tsp100_test_concorde.txt
                  tsp500_test_concorde.txt   tsp1000_test_concorde.txt   tsp10000_test_lkh.txt
```

### TSP-50 and TSP-100 (Concorde)

The training and evaluation data of TSP-50 and TSP-100 are taken from
[chaitjo/learning-tsp](https://github.com/chaitjo/learning-tsp), as in DIFUSCO.
The archive contains 1,502,000 Concorde-solved training instances per size and
the standard 1,280-instance test sets.

```bash
gdown "https://drive.google.com/uc?id=152mpCze-v4d0m9kdsCeVkLdHFkjeDeF5" -O tsp-data.tar.gz
tar -xzf tsp-data.tar.gz data/tsp/tsp50_train_concorde.txt data/tsp/tsp100_train_concorde.txt \
  data/tsp/tsp50_test_concorde.txt data/tsp/tsp100_test_concorde.txt
```

EDISCO trains on the first 500,000 instances of each file. The last 1,280
instances, which are disjoint from that subset, are held out for validation.

```bash
cd data/tsp && mkdir -p train valid test
for n in 50 100; do
  head -n 500000 tsp${n}_train_concorde.txt > train/tsp${n}_train_concorde_500k.txt
  tail -n 1280   tsp${n}_train_concorde.txt > valid/tsp${n}_valid_concorde_1280.txt
  mv tsp${n}_test_concorde.txt test/
done
```

To generate Concorde-labelled instances instead, install
[pyconcorde](https://github.com/jvkersch/pyconcorde) and run
`data/generate_tsp_data.py --solver concorde`.

### TSP-500, TSP-1000 and TSP-10000 (LKH)

Training data are the public LKH-labelled uniform sets of
[ML4CO/ML4CO-Bench-101-SL](https://huggingface.co/datasets/ML4CO/ML4CO-Bench-101-SL/tree/main/train_dataset/tsp)
(1000 LKH trials at TSP-500 and TSP-1000). EDISCO trains on the first
60,000 / 30,000 / 3,000 instances at TSP-500 / 1000 / 10000 and holds out the
next 128 / 128 / 16 instances for validation.

| Scale | Source file | Training instances | Validation instances |
|-------|-------------|--------------------|----------------------|
| TSP-500 | `tsp_500/tsp500_uniform_lkh-1000_320k_01.txt` | lines 1-60000 | lines 60001-60128 |
| TSP-1000 | `tsp_1k/tsp1k_uniform_lkh-1000_128k_01.txt` | lines 1-30000 | lines 30001-30128 |
| TSP-10000 | `tsp_10k/tsp10000_uniform_1.6k_1.txt` followed by `tsp10000_uniform_1.6k_2.txt` | lines 1-3000 | lines 3001-3016 |

```bash
HF=https://huggingface.co/datasets/ML4CO/ML4CO-Bench-101-SL/resolve/main/train_dataset/tsp
wget $HF/tsp_500/tsp500_uniform_lkh-1000_320k_01.txt $HF/tsp_1k/tsp1k_uniform_lkh-1000_128k_01.txt \
     $HF/tsp_10k/tsp10000_uniform_1.6k_1.txt $HF/tsp_10k/tsp10000_uniform_1.6k_2.txt

head -n 60000 tsp500_uniform_lkh-1000_320k_01.txt          > train/tsp500_train_lkh_60k.txt
sed -n '60001,60128p' tsp500_uniform_lkh-1000_320k_01.txt  > valid/tsp500_valid_lkh_128.txt
head -n 30000 tsp1k_uniform_lkh-1000_128k_01.txt           > train/tsp1000_train_lkh_30k.txt
sed -n '30001,30128p' tsp1k_uniform_lkh-1000_128k_01.txt   > valid/tsp1000_valid_lkh_128.txt
cat tsp10000_uniform_1.6k_1.txt tsp10000_uniform_1.6k_2.txt | head -n 3000        > train/tsp10000_train_lkh_3k.txt
cat tsp10000_uniform_1.6k_1.txt tsp10000_uniform_1.6k_2.txt | sed -n '3001,3016p' > valid/tsp10000_valid_lkh_16.txt
```

Instances can also be generated and labelled locally with LKH-3. Build it from
<http://webhotel4.ruc.dk/~keld/research/LKH-3/> and pass the executable with
`--lkh_path`. Only `MAX_TRIALS` and `RUNS` are set; every other LKH-3 parameter
keeps its default. Generation is deterministic (instance `i` uses seed
`seed + i`) and resumable.

```bash
python -u data/generate_tsp_data.py --min_nodes 500 --max_nodes 500 --num_samples 60000 \
  --solver lkh --lkh_trails 1000 --lkh_path /path/to/LKH --num_workers 64 \
  --seed 1234 --filename data/tsp/train/tsp500_train_lkh_60k.txt
```

The evaluation data of TSP-500, TSP-1000 and TSP-10000 (128 / 128 / 16
instances) originate from [Spider-scnu/TSP](https://github.com/Spider-scnu/TSP).
The same test sets are mirrored in the text format used here under
`test_dataset/tsp` of ML4CO-Bench-101-SL (`tsp500_concorde_16.546.txt`,
`tsp1000_concorde_23.118.txt`, `tsp10000_lkh_500_71.782.txt`) and are stored as
`test/tsp500_test_concorde.txt`, `test/tsp1000_test_concorde.txt` and
`test/tsp10000_test_lkh.txt`.

### Cross-distribution test sets (TSP-100)

The Cluster, Explosion and Implosion distributions of Bi et al. (2022) are
generated with `--distribution` (here 1,280 instances per distribution, the
size of the uniform TSP-100 test set):

```bash
for distribution in cluster explosion implosion; do
  python -u data/generate_tsp_data.py --min_nodes 100 --max_nodes 100 --num_samples 1280 \
    --distribution $distribution --solver lkh --lkh_path /path/to/LKH --seed 4321 \
    --filename data/tsp/test/tsp100_${distribution}_test.txt
done
cp data/tsp/test/tsp100_test_concorde.txt data/tsp/test/tsp100_uniform_test.txt
```

### Heuristic labels (training-data study)

```bash
python -u data/generate_tsp_data.py --solver farthest_insertion \
  --min_nodes 50 --max_nodes 50 --num_samples 500000 --seed 1234 \
  --filename data/tsp/train/tsp50_train_farthest_insertion.txt
```

### TSPLIB

`convert_tsplib.py` converts the 29 two-dimensional Euclidean
[TSPLIB95](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/) instances
with 51 to 200 nodes used in the TSPLIB experiment. Place the `.tsp` files (and
`.opt.tour` files where TSPLIB provides them) in `data/tsplib/`, or pass
`--download` to fetch them.

```bash
python -u data/convert_tsplib.py --input_dir data/tsplib --output_dir data/tsplib_processed \
  --download --lkh_path /path/to/LKH
```

Coordinates are rescaled into the unit square with one common factor for both
axes, so the instance is preserved up to a similarity transformation and
optimality gaps are unchanged. The reference tour of an instance is its
published optimal tour when TSPLIB provides one and an LKH-3 solution of the
original file otherwise; the script checks it against the published optimal
length under the TSPLIB metric and records the result in `metadata.json`.
The output is one file per instance (`<name>.txt`) and `tsplib.txt` with all
instances.

### TSP format

One instance per line:

```
x1 y1 x2 y2 ... xn yn output t1 t2 ... tn t1
```

Coordinates lie in the unit square, the word `output` separates coordinates
from the reference tour, tour indices are 1-based and the tour is closed by
repeating its first node.

## Capacitated Vehicle Routing Problem (CVRP)

CVRP instances follow the standard uniform protocol: depot and customers are
uniform on the unit square, demands are integers in {1, ..., 9}, and the vehicle
capacity is Q = 40 / 50 / 80 / 100 for N = 50 / 100 / 200 / 500 customers.
Reference solutions are computed with [HGS-CVRP](https://github.com/vidalt/HGS-CVRP)
through the `hygese` package.

`generate_cvrp_data.py` generates and labels instances in parallel. The default
HGS-CVRP time limit per instance is 1 / 20 / 60 / 240 seconds for
N = 50 / 100 / 200 / 500; generation is deterministic (instance `i` uses seed
`seed + i`).

### Training, validation and test sets

```bash
# training sets: 500K / 250K / 16K / 6K instances
python -u data/generate_cvrp_data.py --num_customers 50  --num_samples 500000 --seed 1234 --filename data/cvrp/cvrp50_train.pkl
python -u data/generate_cvrp_data.py --num_customers 100 --num_samples 250000 --seed 1234 --filename data/cvrp/cvrp100_train.pkl
python -u data/generate_cvrp_data.py --num_customers 200 --num_samples 16000  --seed 1234 --filename data/cvrp/cvrp200_train.pkl
python -u data/generate_cvrp_data.py --num_customers 500 --num_samples 6000   --seed 1234 --filename data/cvrp/cvrp500_train.pkl

# validation sets (held out) and test sets: 10K / 10K / 100 / 100 test instances
for n in 50 100 200 500; do
  python -u data/generate_cvrp_data.py --num_customers $n --num_samples 128 --seed 90000000 --filename data/cvrp/cvrp${n}_valid.pkl
done
python -u data/generate_cvrp_data.py --num_customers 50  --num_samples 10000 --seed 80000000 --filename data/cvrp/cvrp50_test.pkl
python -u data/generate_cvrp_data.py --num_customers 100 --num_samples 10000 --seed 80000000 --filename data/cvrp/cvrp100_test.pkl
python -u data/generate_cvrp_data.py --num_customers 200 --num_samples 100   --seed 80000000 --filename data/cvrp/cvrp200_test.pkl
python -u data/generate_cvrp_data.py --num_customers 500 --num_samples 100   --seed 80000000 --filename data/cvrp/cvrp500_test.pkl
```

### Large-scale sets (CVRP-1000 / CVRP-2000)

Instances for the partition-diffusion pipeline are generated with the same
script by passing the capacity explicitly:

```bash
python -u data/generate_cvrp_data.py --num_customers 1000 --num_samples 100 --capacity 200 \
  --time_limit 240 --seed 82000000 --filename data/cvrp/cvrp1000_test.pkl
python -u data/generate_cvrp_data.py --num_customers 2000 --num_samples 100 --capacity 300 \
  --time_limit 480 --seed 82000000 --filename data/cvrp/cvrp2000_test.pkl
```

### Capacity-shift sets (CVRP-100)

```bash
# mixed-capacity training and validation: one capacity per instance, uniform on {10, ..., 500}
python -u data/generate_cvrp_data.py --num_customers 100 --num_samples 250000 --seed 2234 \
  --capacity_min 10 --capacity_max 500 --filename data/cvrp/cvrp100_mixed_train.pkl
python -u data/generate_cvrp_data.py --num_customers 100 --num_samples 128 --seed 91000000 \
  --capacity_min 10 --capacity_max 500 --filename data/cvrp/cvrp100_mixed_valid.pkl

# test bins: 10,000 instances at each absolute capacity
for C in 10 50 100 200 300 400 500; do
  python -u data/generate_cvrp_data.py --num_customers 100 --num_samples 10000 --seed 81000000 \
    --capacity $C --filename data/cvrp/cvrp100_C${C}_test.pkl
done
```

### CVRP format

A pickle file holding a list of instances, each a dictionary:

- `coords`: `(N + 1, 2)` float array, node 0 is the depot
- `demands`: `(N + 1,)` float array, the depot demand is 0
- `capacity`: vehicle capacity Q
- `n_customers`, `n_nodes`: N and N + 1
- `solution`: `{'routes': [[customer, ...], ...], 'total_distance': float, 'solver': 'hgs'}`,
  with customers indexed from 1 and each route starting and ending at the depot

## Euclidean Steiner Tree Problem (ESTP)

Every instance has `--problem_size` terminals and the same number of candidate
Steiner points, all uniform on the unit square. Steiner-10 / 20 / 50 use 10,000
training instances and 1,000 validation and test instances each:

```bash
for n in 10 20 50; do
  python -u data/generate_steiner_data.py --problem_size $n --num_samples 10000 --seed 1234 --filename data/steiner/steiner${n}_train.txt
  python -u data/generate_steiner_data.py --problem_size $n --num_samples 1000  --seed 4321 --filename data/steiner/steiner${n}_valid.txt
  python -u data/generate_steiner_data.py --problem_size $n --num_samples 1000  --seed 5678 --filename data/steiner/steiner${n}_test.txt
done
```

Reference trees come from iterated 1-Steiner on the candidate set (default,
`--solver iterated_1steiner`) or from the exact solver
[GeoSteiner](http://www.geosteiner.com/) (`--solver geosteiner`, which requires
the GeoSteiner executables on the `PATH`).

### Steiner format

One instance per line:

```
x1 y1 ... xn yn SEP u1 v1 ... um vm output a_11 a_12 ... a_(n+m)(n+m)
```

Terminal coordinates come before `SEP`, candidate Steiner point coordinates
between `SEP` and `output`, followed by the row-major binary adjacency matrix
of the reference tree over the `n + m` nodes (terminals first).

## Maximum Independent Set (MIS)

RB-[200-300] and ER-[700-800] graphs with KaMIS labels are available from
[ML4CO/ML4CO-Bench-101-SL](https://huggingface.co/datasets/ML4CO/ML4CO-Bench-101-SL)
as text files with one graph per line, which `edisco/train.py --task mis` reads
directly:

- training: `train_dataset/mis/mis_rb_small/mis_rb-small_kamis-10s_16k_*.txt`,
  `train_dataset/mis/mis_er_700_800/mis_er700-800_1.6k_*.txt`
- test: `test_dataset/mis/mis_rb-small_kamis-60s_20.090.txt`,
  `test_dataset/mis/mis_er-700-800_kamis-60s_44.969.txt`

Graphs can also be generated and labelled locally with the MIS benchmark
framework in `data/mis-benchmark-framework/` (see its `readme.md` for the
KaMIS setup):

```bash
cd data/mis-benchmark-framework
conda env create -f environment.yml && conda activate mis-benchmark && bash setup_bm_env.sh

python main.py gendata random . ../mis/er_700_800_train \
  --model er --min_n 700 --max_n 800 --num_graphs 4000 --er_p 0.15 --gen_labels
```

### MIS formats

- Text: one graph per line, `u1 v1 u2 v2 ... label l_0 l_1 ... l_{n-1}`, an edge
  list with 0-based node indices followed by the binary node labels.
- A quoted glob of NetworkX `.gpickle` graphs such as `"data/mis/er_700_800_train/*.gpickle"`;
  each node carries a `label` attribute (1 if it belongs to the reference
  independent set), or labels are read from `--training_split_label_dir`.
