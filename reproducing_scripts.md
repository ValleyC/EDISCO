# Reproducing EDISCO

Training and evaluation commands for every experiment of the paper. All
commands are run from the repository root on a single 48 GB GPU. Replace the
`/your/...` paths with local data and checkpoint locations; data splits are
given relative to `--storage_path`, and checkpoints and logs are written to
`<storage_path>/models`.

```bash
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline   # optional: log locally without a Weights & Biases account
```

The main-table configuration for TSP and CVRP is the 5-step DEIS-2 solver with
NEE decoding. The 50-step PNDM configuration is the slower, higher-quality
variant.

| Experiment | Paper | Section below |
|------------|-------|---------------|
| TSP-50 to TSP-10000 | Tables 1, 11 | [TSP](#tsp) |
| Cross-distribution generalization | Table 2 | [Cross-distribution generalization](#cross-distribution-generalization-section-42-table-2) |
| TSPLIB | Table 12 | [TSPLIB](#tsplib-appendix-f4-table-12) |
| Solver sweep, noise schedules, hyperparameters, model sizes | Tables 10, 13-17 | [TSP](#tsp) |
| Cross-size generalization | Figure 2 | [Cross-size generalization](#cross-size-generalization-section-42-figure-2) |
| Training-data variations | Figures 4, 5 | [Training-data variations](#training-data-variations-section-44-figures-4-and-5) |
| CVRP-50 to CVRP-500, capacity shift | Tables 3, 4 | [CVRP](#cvrp) |
| CVRP-1000 / 2000 (partition diffusion) | Table 8 | [Large-scale CVRP](#large-scale-cvrp-appendix-a2-partition-diffusion) |
| Euclidean Steiner tree | Table 7 | [Euclidean Steiner Tree](#euclidean-steiner-tree) |
| Maximum independent set | Table 9 | [Maximum Independent Set](#maximum-independent-set-appendix-a3) |
| Encoder and decoder ablations, consistency probe | Tables 5, 6 | [Ablations](#ablations-section-45) |

## TSP

Data preparation (public Concorde sets for TSP-50/100, public LKH sets for larger
scales, standard test sets) is described in [data/README.md](data/README.md).
Validation uses held-out instances, never the test set, and the checkpoint with
the best validation tour length is kept for evaluation and curriculum
initialization (its path is written to `checkpoints/best_checkpoint.txt`).

### Training

All TSP models share the optimizer and diffusion settings of the paper:
AdamW, learning rate `2e-4`, weight decay `1e-5`, cosine schedule, gradient
clipping at unit norm, linear `beta(t)` from 0.1 to 1.5 and the
`(1 - sqrt(t))`-weighted cross-entropy objective. These are the defaults of
`edisco/train.py`.

The effective batch size is `--batch_size` times `--accumulate_grad_batches`.
Gradient accumulation and `--use_activation_checkpoint` (which recomputes layer
activations in the backward pass) leave the optimizer step of the full batch
unchanged and keep peak memory within a single 48 GB GPU in 32-bit precision.

| Scale | Training set | Graph | Effective batch | Epochs | Initialization |
|-------|--------------|-------|-----------------|--------|----------------|
| TSP-50 | 500K (Concorde) | dense | 64 | 100 | scratch |
| TSP-100 | 500K (Concorde) | dense | 32 = 16 x 2 | 100 | scratch |
| TSP-500 | 60K (LKH) | kNN, k=50 | 16 = 8 x 2 | 50 | TSP-100 best |
| TSP-1000 | 30K (LKH) | kNN, k=100 | 8 | 50 | TSP-100 best |
| TSP-10000 | 3K (LKH) | kNN, k=100 | 4 = 2 x 2 | 50 | TSP-500 best |

TSP-50 (dense graphs):

```bash
python -u edisco/train.py \
  --task tsp \
  --do_train --do_test \
  --learning_rate 0.0002 --weight_decay 0.00001 --lr_scheduler cosine-decay \
  --storage_path /your/storage/path \
  --training_split data/tsp/train/tsp50_train_concorde_500k.txt \
  --validation_split data/tsp/valid/tsp50_valid_concorde_1280.txt \
  --test_split data/tsp/test/tsp50_test_concorde.txt \
  --batch_size 64 --num_epochs 100 \
  --n_layers 12 --hidden_dim 256 \
  --solver_type pndm --solver_steps 50 \
  --time_schedule linear --beta_min 0.1 --beta_max 1.5
```

TSP-100 uses the same command with the TSP-100 files and
`--batch_size 16 --accumulate_grad_batches 2`.

TSP-500 (kNN-sparsified, curriculum from the TSP-100 checkpoint with the best
validation performance):

```bash
python -u edisco/train.py \
  --task tsp \
  --do_train --do_test \
  --learning_rate 0.0002 --weight_decay 0.00001 --lr_scheduler cosine-decay \
  --storage_path /your/storage/path \
  --training_split data/tsp/train/tsp500_train_lkh_60k.txt \
  --validation_split data/tsp/valid/tsp500_valid_lkh_128.txt \
  --test_split data/tsp/test/tsp500_test_concorde.txt \
  --sparse_factor 50 \
  --batch_size 8 --accumulate_grad_batches 2 --num_epochs 50 \
  --n_layers 12 --hidden_dim 256 \
  --solver_type pndm --solver_steps 50 \
  --time_schedule linear --beta_min 0.1 --beta_max 1.5 \
  --ckpt_path /your/tsp100_best.ckpt --resume_weight_only
```

For TSP-1000 use `--sparse_factor 100 --batch_size 8 --use_activation_checkpoint`
(curriculum from TSP-100). For TSP-10000 use
`--sparse_factor 100 --batch_size 2 --accumulate_grad_batches 2 --validation_examples 16 --use_activation_checkpoint`
(curriculum from TSP-500).

To continue an interrupted run, pass its `checkpoints/last.ckpt` as
`--ckpt_path` without `--resume_weight_only`, together with `--resume_id`.

### Evaluation (NEE decoding, headline configuration)

```bash
python -u edisco/train.py \
  --task tsp \
  --do_test \
  --storage_path /your/storage/path \
  --test_split data/tsp/test/tsp100_test_concorde.txt \
  --validation_split data/tsp/valid/tsp100_valid_concorde_1280.txt \
  --n_layers 12 --hidden_dim 256 \
  --solver_type deis --solver_steps 5 \
  --decoder nee --two_opt_iterations 0 \
  --ckpt_path /your/tsp100_best.ckpt --resume_weight_only
```

A single-evaluation solver (Euler, DDIM, DEIS-2, PNDM) with `T` steps uses
`T + 1` network evaluations: `T` reverse steps followed by the final
clean-state prediction at `t = 0` that is passed to the decoder. Heun and DPM-2
use `2T` and RK4 uses `4T` evaluations.

Reported results average five runs with `--seed 1` to `--seed 5`. The test loop
logs `test/time`, the per-instance wall-clock time of sampling and decoding.
The timing protocol uses one instance at a time (the TSP test loader does this
by default) and `--single_thread`.

Use `--decoder greedy --two_opt_iterations 0` for the probability-only greedy
decoder. Both decoders enforce feasibility once on each final clean-edge
heatmap. NEE ranks edges by the symmetrized probability divided by the distance
plus `1e-8`. Sampling-based decoding draws
`--parallel_sampling` x `--sequential_sampling` candidates per instance and
keeps the best; `--two_opt_iterations` adds 2-opt local search for reference
comparisons.

### Cross-size generalization (Section 4.2, Figure 2)

A checkpoint trained at one scale is evaluated at another scale by passing the
test set and the graph setting of the target scale (`--sparse_factor 50` at
TSP-500, `--sparse_factor 100` at TSP-1000, dense graphs up to TSP-100) together
with `--decoder greedy`. Dense and sparse models share their parameters, so
every checkpoint loads at every scale. For example, the TSP-100 model on
TSP-500:

```bash
python -u edisco/train.py \
  --task tsp \
  --do_test \
  --storage_path /your/storage/path \
  --validation_split data/tsp/valid/tsp500_valid_lkh_128.txt \
  --test_split data/tsp/test/tsp500_test_concorde.txt --sparse_factor 50 \
  --n_layers 12 --hidden_dim 256 \
  --solver_type deis --solver_steps 5 --decoder greedy \
  --ckpt_path /your/tsp100_best.ckpt --resume_weight_only
```

### Cross-distribution generalization (Section 4.2, Table 2)

The TSP-100 model trained on uniform instances is evaluated on Cluster,
Explosion and Implosion instances (generation in [data/README.md](data/README.md))
with greedy decoding and 2-opt:

```bash
for distribution in uniform cluster explosion implosion; do
  python -u edisco/train.py \
    --task tsp \
    --do_test \
    --storage_path /your/storage/path \
    --validation_split data/tsp/valid/tsp100_valid_concorde_1280.txt \
    --test_split data/tsp/test/tsp100_${distribution}_test.txt \
    --n_layers 12 --hidden_dim 256 \
    --solver_type deis --solver_steps 5 \
    --decoder greedy --two_opt_iterations 5000 \
    --ckpt_path /your/tsp100_best.ckpt --resume_weight_only
done
```

### TSPLIB (Appendix F.4, Table 12)

The same TSP-100 checkpoint is evaluated on the 29 TSPLIB instances with 51 to
200 nodes (conversion in [data/README.md](data/README.md)) with 4x sampling and
2-opt. `tsplib.txt` holds all instances; pass `<name>.txt` as `--test_split`
for a single instance.

```bash
python -u edisco/train.py \
  --task tsp \
  --do_test \
  --storage_path /your/storage/path \
  --validation_split data/tsplib_processed/tsplib.txt \
  --test_split data/tsplib_processed/tsplib.txt \
  --n_layers 12 --hidden_dim 256 \
  --solver_type deis --solver_steps 5 \
  --parallel_sampling 4 --two_opt_iterations 5000 \
  --ckpt_path /your/tsp100_best.ckpt --resume_weight_only
```

### Solver sweep (Appendix F.2, Table 10)

```bash
for solver in euler ddim pndm dpm2 deis rk4 heun; do
  python -u edisco/train.py \
    --task tsp \
    --do_test \
    --storage_path /your/storage/path \
    --validation_split data/tsp/valid/tsp500_valid_lkh_128.txt \
    --test_split data/tsp/test/tsp500_test_concorde.txt --sparse_factor 50 \
    --decoder greedy --n_layers 12 --hidden_dim 256 \
    --solver_type "$solver" --solver_steps 50 \
    --time_schedule linear \
    --ckpt_path /your/tsp500_best.ckpt --resume_weight_only
done
```

### Noise schedules (Appendix F.5, Table 13)

The forward process uses the linear schedule `beta(t) = beta_min + t (beta_max - beta_min)`
with `--beta_min 0.1 --beta_max 1.5` by default. The schedule comparison trains
TSP-50 models on 10,000 instances for 50 epochs with one of:

```bash
--beta_schedule linear --beta_min 0.1 --beta_max 2.0          # also 1.5 (default) and 1.0
--beta_schedule exponential --beta_exp_a 0.5 --beta_exp_b 4.0  # also (0.3, 3.0) and (0.8, 5.0)
--beta_schedule cosine --beta_min 0.01 --beta_max 5.0          # also (0.001, 10.0) and (0.1, 3.0)
```

Pass the same schedule flags at evaluation: the reverse sampler uses the
transition probabilities of the schedule the model was trained with.

### Architectural hyperparameters (Appendix F.6, Tables 14-16)

The coordinate step size and the weight temperature are set with
`--coord_update_alpha` (default 0.1) and `--weight_temp` (default 10).

### Model sizes (Appendix F.7, Table 17)

EDISCO-Full uses `--n_layers 12 --hidden_dim 256`, EDISCO-Medium
`--n_layers 12 --hidden_dim 128` and EDISCO-Small `--n_layers 8 --hidden_dim 128`,
evaluated with `--solver_type pndm --solver_steps 50 --decoder greedy`.

### Training-data variations (Section 4.4, Figures 4 and 5)

Training on a fraction of the data uses the first lines of the TSP-50 training
file (Figure 4) or of the TSP-100 training file (Figure 5), for example
`head -n 100000 tsp50_train_concorde_500k.txt`. Training on heuristic labels
uses Farthest Insertion tours:

```bash
python -u data/generate_tsp_data.py --solver farthest_insertion \
  --min_nodes 50 --max_nodes 50 --num_samples 500000 --seed 1234 \
  --filename data/tsp/train/tsp50_train_farthest_insertion.txt
```

Both are evaluated with `--decoder greedy`.

## CVRP

Data generation with HGS-CVRP labels is described in [data/README.md](data/README.md).
CVRP models use the capacity-conditioned encoder (capacity-normalized node and
edge features plus FiLM modulation of the scalar messages) and the same
optimizer and diffusion settings as TSP. Graphs are dense for N <= 100 and
kNN-sparsified (k = 50) for CVRP-200 and CVRP-500. Decoding uses the
capacity-feasible edge expansion: candidate edges are ranked by
`(P_ij + P_ji) / (2 (d_ij + 1e-8))` and accepted subject to customer degree two,
one depot edge pair per route and the route capacity.

### CVRP-50 / 100 / 200 / 500 training

| Scale | Training set | Graph | Effective batch | Epochs | Initialization |
|-------|--------------|-------|-----------------|--------|----------------|
| CVRP-50 | 500K | dense | 64 | 50 | scratch |
| CVRP-100 | 250K | dense | 32 = 16 x 2 | 50 | scratch |
| CVRP-200 | 16K | kNN, k=50 | 16 | 50 | CVRP-100 best |
| CVRP-500 | 6K | kNN, k=50 | 8 | 50 | CVRP-200 best |

```bash
python -u edisco/train.py \
  --task cvrp \
  --do_train --do_test \
  --learning_rate 0.0002 --weight_decay 0.00001 --lr_scheduler cosine-decay \
  --storage_path /your/storage/path \
  --training_split data/cvrp/cvrp100_train.pkl \
  --validation_split data/cvrp/cvrp100_valid.pkl \
  --test_split data/cvrp/cvrp100_test.pkl \
  --batch_size 16 --accumulate_grad_batches 2 --num_epochs 50 \
  --n_layers 12 --hidden_dim 256 \
  --solver_type pndm --solver_steps 50 \
  --time_schedule linear --beta_min 0.1 --beta_max 1.5
```

For CVRP-50 use `--batch_size 64`. For CVRP-200 use
`--sparse_factor 50 --batch_size 16 --ckpt_path /your/cvrp100_best.ckpt --resume_weight_only`.
For CVRP-500 use
`--sparse_factor 50 --batch_size 8 --ckpt_path /your/cvrp200_best.ckpt --resume_weight_only`.
The dense CVRP-100 checkpoint initializes the sparse models directly.

### Constraint-shift evaluation (Section 4.3, Table 4)

Mixed-capacity training (per-instance capacity drawn from 10 to 500, reference
capacity 50), capacity-conditioned:

```bash
python -u edisco/train.py \
  --task cvrp \
  --do_train --do_test \
  --storage_path /your/storage/path \
  --training_split data/cvrp/cvrp100_mixed_train.pkl \
  --validation_split data/cvrp/cvrp100_mixed_valid.pkl \
  --test_split data/cvrp/cvrp100_C50_test.pkl \
  --default_capacity 50 \
  --batch_size 16 --accumulate_grad_batches 2 --num_epochs 50 \
  --n_layers 12 --hidden_dim 256 \
  --solver_type pndm --solver_steps 50 \
  --time_schedule linear --beta_min 0.1 --beta_max 1.5
```

Add `--disable_capacity_conditioning` for the mixed unconditioned variant. The
default-only variant uses `--disable_capacity_conditioning` with the fixed-capacity
CVRP-100 training set, and the per-capacity specialists use the conditioned
command with one fixed-capacity training set each.

Per-capacity evaluation across `C ∈ {10, 50, 100, 200, 300, 400, 500}`:

```bash
for C in 10 50 100 200 300 400 500; do
  python -u edisco/train.py \
    --task cvrp \
    --do_test \
    --storage_path /your/storage/path \
    --validation_split data/cvrp/cvrp100_mixed_valid.pkl \
    --test_split data/cvrp/cvrp100_C${C}_test.pkl \
    --default_capacity 50 \
    --batch_size 32 --n_layers 12 --hidden_dim 256 \
    --solver_type deis --solver_steps 5 \
    --ckpt_path /your/cvrp100_mixed_best.ckpt --resume_weight_only
done
```

Add `--disable_capacity_conditioning` when evaluating an unconditioned checkpoint.

### Large-scale CVRP (Appendix A.2, partition diffusion)

CVRP-1000 / CVRP-2000 use a two-stage pipeline (`--task cvrp_partition`).
Stage 1 diffuses the symmetric same-route indicator over customer pairs with
the same categorical CTMC and capacity-conditioned encoder as end-to-end CVRP;
each HGS-CVRP reference route defines one cluster. The denoised affinities are
projected onto capacity-feasible clusters by spectral clustering followed by
capacity-feasible re-balancing. Stage 2 solves every cluster together with the
depot as a small TSP with the EDISCO TSP score network and NEE.

Training the partition score network (same flags as CVRP; shown for CVRP-500):

```bash
python -u edisco/train.py \
  --task cvrp_partition \
  --do_train --do_test \
  --learning_rate 0.0002 --weight_decay 0.00001 --lr_scheduler cosine-decay \
  --storage_path /your/storage/path \
  --training_split data/cvrp/cvrp500_train.pkl \
  --validation_split data/cvrp/cvrp500_valid.pkl \
  --test_split data/cvrp/cvrp500_test.pkl \
  --sparse_factor 50 --batch_size 8 --num_epochs 50 \
  --n_layers 12 --hidden_dim 256 \
  --solver_type pndm --solver_steps 50 \
  --time_schedule linear --beta_min 0.1 --beta_max 1.5
```

Solving CVRP-1000 / CVRP-2000 with the full pipeline:

```bash
python -u edisco/train.py \
  --task cvrp_partition \
  --do_test \
  --storage_path /your/storage/path \
  --validation_split data/cvrp/cvrp1000_test.pkl \
  --test_split data/cvrp/cvrp1000_test.pkl \
  --sparse_factor 100 --batch_size 1 --n_layers 12 --hidden_dim 256 \
  --solver_type deis --solver_steps 5 \
  --ckpt_path /your/partition_best.ckpt --resume_weight_only \
  --sub_tsp_ckpt /your/tsp50_best.ckpt
```

`--partitioner kmeans` replaces the diffusion partitioner with Lloyd's algorithm
on raw coordinates and keeps the projection and the sub-TSP solver unchanged
(ablation). Without `--sub_tsp_ckpt` each cluster is ordered by NEE on
distances alone, which is what validation during partition training uses. The
number of clusters is the smallest number of vehicles that can carry the total
demand.

## Euclidean Steiner Tree

ESTP reuses the TSP configuration: the same 12-layer EGNN with the terminal
indicator as invariant node input, the same optimizer and diffusion settings,
and dense graphs over the terminals and candidate Steiner points. Decoding uses
the Kruskal-style tree decoder on `(P_ij + P_ji) / (d_ij + 1e-8)`.

### Steiner-10 / 20 / 50

```bash
python -u edisco/train.py \
  --task steiner \
  --do_train --do_test \
  --learning_rate 0.0002 --weight_decay 0.00001 --lr_scheduler cosine-decay \
  --storage_path /your/storage/path \
  --training_split data/steiner/steiner20_train.txt \
  --validation_split data/steiner/steiner20_valid.txt \
  --test_split data/steiner/steiner20_test.txt \
  --batch_size 64 --num_epochs 100 \
  --n_layers 12 --hidden_dim 256 \
  --solver_type pndm --solver_steps 50 \
  --time_schedule linear --beta_min 0.1 --beta_max 1.5
```

Steiner-50 has 100 nodes per instance and uses
`--batch_size 16 --accumulate_grad_batches 2`, as TSP-100. Evaluate the 5-step
configuration with `--do_test --solver_type deis --solver_steps 5`.

## Maximum Independent Set (Appendix A.3)

MIS is non-Euclidean and admits no E(2) action, so the equivariance contributions do not apply. The non-equivariant GNN (`edisco/models/gnn_encoder.py`) is paired with the same categorical CTMC (forward kernel, weighted objective and reverse solver) to evaluate the generality of the diffusion engine.

RB-[200-300]:

```bash
python -u edisco/train.py \
  --task mis \
  --do_train --do_test \
  --learning_rate 0.0002 --weight_decay 0.0001 --lr_scheduler cosine-decay \
  --storage_path /your/storage/path \
  --training_split data/mis/mis_rb_small_train.txt \
  --validation_split data/mis/mis_rb_small_valid.txt \
  --test_split data/mis/mis_rb_small_test.txt \
  --batch_size 16 --num_epochs 50 \
  --n_layers 12 --hidden_dim 256 \
  --solver_type pndm --solver_steps 50 \
  --time_schedule linear --beta_min 0.1 --beta_max 1.5 \
  --use_activation_checkpoint
```

ER-[700-800] uses the same flags with the ER data paths and `--batch_size 4`.
Data splits may be text files (one graph per line) or a quoted glob of
`.gpickle` graphs such as `"data/mis/er_train/*gpickle"`. Greedy decoding is the
default. Sampling-based decoding draws 128 candidates and keeps the best, e.g.
`--parallel_sampling 16 --sequential_sampling 8`.

## Ablations (Section 4.5)

### Encoder substitution (Table 5, decoder fixed at greedy)

EDISCO Full (architectural E(2)):

```bash
python -u edisco/train.py --task tsp --do_test \
  --storage_path /your/storage/path \
  --validation_split data/tsp/valid/tsp500_valid_lkh_128.txt \
  --test_split data/tsp/test/tsp500_test_concorde.txt --sparse_factor 50 \
  --solver_type deis --solver_steps 5 --decoder greedy \
  --ckpt_path /your/tsp500_full_best.ckpt --resume_weight_only \
  --n_layers 12 --hidden_dim 256
```

Non-equivariant GNN, parameter-matched (Table 5 row "None"):

```bash
python -u edisco/train.py --task tsp --disable_equivariance --do_train --do_test \
  --storage_path /your/storage/path \
  --training_split data/tsp/train/tsp500_train_lkh_60k.txt \
  --validation_split data/tsp/valid/tsp500_valid_lkh_128.txt \
  --test_split data/tsp/test/tsp500_test_concorde.txt \
  --sparse_factor 50 --batch_size 16 --num_epochs 50 --use_activation_checkpoint \
  --n_layers 12 --hidden_dim 256 \
  --solver_type pndm --solver_steps 50 --time_schedule linear
```

For TSP-1000 use `--sparse_factor 100 --batch_size 8 --use_activation_checkpoint` with the TSP-1000 files. Evaluate with `--disable_equivariance --decoder greedy`.

Add `--data_augmentation e2` to enable E(2) data augmentation (Table 5 row "Augmentation"): each training instance is rotated by a uniform angle about the centre of the unit square, reflected with probability one half and translated by a uniform shift in `[-0.5, 0.5]^2`. Add `--symmetry_loss` to enable the soft symmetry regularizer (Table 5 row "Soft regularizer"): the mean squared difference between the edge probabilities predicted on an instance and on a randomly transformed copy, weighted by `--symmetry_loss_weight` (default 1).

### Decoder choice (Table 5, encoder fixed at full EDISCO)

Greedy only:

```bash
python -u edisco/train.py --task tsp --do_test \
  --decoder greedy \
  --storage_path /your/storage/path \
  --validation_split data/tsp/valid/tsp500_valid_lkh_128.txt \
  --test_split data/tsp/test/tsp500_test_concorde.txt --sparse_factor 50 \
  --solver_type deis --solver_steps 5 \
  --ckpt_path /your/tsp500_full_best.ckpt --resume_weight_only
```

Greedy + 2-opt: same command with `--decoder greedy --two_opt_iterations 5000`.

Native Edge Expansion (proposed): same command with `--decoder nee`.

### Edge-probability consistency probe (Table 6)

Add `--test_equivariance` to a `--do_test` run on a dense TSP checkpoint. The
run reports `test/consistency_max_abs_dp`, the mean over test instances and 16
random E(2) elements of the largest change in predicted edge probability, with
the noisy edge state and diffusion time held fixed. Add `--disable_equivariance`
for a non-equivariant checkpoint.

## Tests

```bash
python -m pytest tests -q
```

The tests check the forward kernel and training objective against their closed
forms, the exact posterior against enumeration, E(2)-invariance of the edge
logits (dense, sparse, capacity-conditioned and with terminal indicators),
E(2)-invariance and feasibility of the TSP, CVRP and Steiner decoders, the
network-evaluation counts of all solvers and the model sizes.
