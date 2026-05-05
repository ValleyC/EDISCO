# Reproducing EDISCO

All commands assume the repository root is on `PYTHONPATH` and a single 48 GB GPU per process. Replace the `/your/...` paths with local data and checkpoint locations.

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
```

The TSP/CVRP main-table configuration is the 5-step DEIS-2 sampler paired with NEE decoding. The slower 50-step PNDM configuration is reported as the higher-quality variant.

## TSP

### Training

TSP-50 / TSP-100 (dense graphs):

```bash
python -u edisco/train.py \
  --task tsp \
  --diffusion_type categorical --continuous_time --equivariant \
  --do_train --do_test \
  --learning_rate 0.0002 --weight_decay 0.00001 --lr_scheduler cosine-decay \
  --storage_path /your/storage/path \
  --training_split /your/tsp100_train_concorde.txt \
  --validation_split /your/tsp100_valid_concorde.txt \
  --test_split /your/tsp100_test_concorde.txt \
  --batch_size 32 --num_epochs 100 \
  --n_layers 12 --hidden_dim 256 \
  --solver_type pndm --solver_steps 50 \
  --time_schedule linear --beta_min 0.1 --beta_max 1.5
```

TSP-500 / TSP-1000 (kNN-sparsified, curriculum from TSP-100):

```bash
python -u edisco/train.py \
  --task tsp \
  --diffusion_type categorical --continuous_time --equivariant \
  --do_train --do_test \
  --learning_rate 0.0002 --weight_decay 0.00001 --lr_scheduler cosine-decay \
  --storage_path /your/storage/path \
  --training_split /your/tsp500_train_lkh.txt \
  --validation_split /your/tsp500_valid_lkh.txt \
  --test_split /your/tsp500_test_lkh.txt \
  --sparse_factor 50 \
  --batch_size 16 --num_epochs 50 \
  --n_layers 12 --hidden_dim 256 \
  --solver_type pndm --solver_steps 50 \
  --time_schedule linear --beta_min 0.1 --beta_max 1.5 \
  --ckpt_path /your/tsp100_best.ckpt --resume_weight_only
```

For TSP-1000 use `--sparse_factor 100 --batch_size 8 --use_activation_checkpoint`. For TSP-10000 curriculum from TSP-500 with `--sparse_factor 100 --batch_size 4 --use_activation_checkpoint`.

### Evaluation (greedy with NEE decoding, headline configuration)

```bash
python -u edisco/train.py \
  --task tsp \
  --diffusion_type categorical --continuous_time --equivariant \
  --do_test \
  --storage_path /your/storage/path \
  --test_split /your/tsp100_test_concorde.txt \
  --batch_size 32 --n_layers 12 --hidden_dim 256 \
  --solver_type deis --solver_steps 5 \
  --time_schedule linear \
  --ckpt_path /your/tsp100_best.ckpt --resume_weight_only
```

For sampling-based decoding append `--parallel_sampling 4 --two_opt_iterations 5000`. For TSPLIB transfer use the same TSP-100 checkpoint with `--parallel_sampling 4`.

### Solver sweep (Appendix G.2)

```bash
for solver in euler ddim pndm dpm2 deis rk4 heun; do
  python -u edisco/train.py \
    --task tsp \
    --diffusion_type categorical --continuous_time --equivariant \
    --do_test --compare_solvers \
    --storage_path /your/storage/path \
    --test_split /your/tsp500_test_lkh.txt --sparse_factor 50 \
    --batch_size 32 --n_layers 12 --hidden_dim 256 \
    --solver_type "$solver" --solver_steps 50 \
    --time_schedule linear \
    --ckpt_path /your/tsp500_best.ckpt --resume_weight_only
done
```

## CVRP

### CVRP-50 / 100 / 200 / 500 training

Capacity conditioning (invariant-feature inputs and FiLM modulation on scalar message channels) is enabled automatically for the CVRP task. Curriculum from CVRP-100 is used at CVRP-200/500.

```bash
python -u edisco/train.py \
  --task cvrp \
  --diffusion_type categorical --continuous_time --equivariant \
  --do_train --do_test \
  --learning_rate 0.0002 --weight_decay 0.00001 --lr_scheduler cosine-decay \
  --storage_path /your/storage/path \
  --training_split /your/cvrp100_train.txt \
  --validation_split /your/cvrp100_valid.txt \
  --test_split /your/cvrp100_test.txt \
  --batch_size 32 --num_epochs 50 \
  --n_layers 12 --hidden_dim 256 \
  --solver_type pndm --solver_steps 50 \
  --time_schedule linear --beta_min 0.1 --beta_max 1.5
```

### Constraint-shift evaluation (Section 4.4)

Mixed-capacity training:

```bash
python -u edisco/train.py \
  --task cvrp \
  --diffusion_type categorical --continuous_time --equivariant \
  --do_train --do_test \
  --storage_path /your/storage/path \
  --training_split /your/cvrp100_mixed_capacity_train.txt \
  --validation_split /your/cvrp100_mixed_capacity_valid.txt \
  --test_split /your/cvrp100_mixed_capacity_test.txt \
  --batch_size 32 --num_epochs 50 \
  --n_layers 12 --hidden_dim 256 \
  --solver_type pndm --solver_steps 50 \
  --time_schedule linear --beta_min 0.1 --beta_max 1.5
```

Per-capacity evaluation across `C ∈ {10, 50, 100, 200, 300, 400, 500}`:

```bash
for C in 10 50 100 200 300 400 500; do
  python -u edisco/train.py \
    --task cvrp \
    --diffusion_type categorical --continuous_time --equivariant \
    --do_test \
    --storage_path /your/storage/path \
    --test_split /your/cvrp100_C${C}_test.txt \
    --batch_size 32 --n_layers 12 --hidden_dim 256 \
    --solver_type deis --solver_steps 5 \
    --ckpt_path /your/cvrp100_mixed_best.ckpt --resume_weight_only
done
```

### Large-scale CVRP (Appendix B.2, partition diffusion)

CVRP-1000 / CVRP-2000 use the partition-diffusion extension (categorical CTMC over a pairwise same-route indicator + capacity-feasible projection + EDISCO sub-TSP solver). Training and evaluation scripts live in `edisco/cvrp_partition/` and follow the same flag conventions.

## Euclidean Steiner Tree

### Steiner-10 / 20 / 50

```bash
python -u edisco/train.py \
  --task steiner \
  --diffusion_type categorical --continuous_time --equivariant \
  --do_train --do_test \
  --learning_rate 0.0001 --weight_decay 0.00001 --lr_scheduler cosine-decay \
  --storage_path /your/storage/path \
  --training_split steiner20_train.txt \
  --validation_split steiner20_valid.txt \
  --test_split steiner20_test.txt \
  --batch_size 16 --num_epochs 100 \
  --n_layers 8 --hidden_dim 128 \
  --solver_type pndm --solver_steps 50 \
  --time_schedule linear --beta_min 0.1 --beta_max 1.5
```

For Steiner-50 add `--sparse_factor 20 --learning_rate 0.00005 --num_epochs 150 --n_layers 10`.

## Maximum Independent Set (engine-generality evidence, Appendix B.3)

MIS is non-Euclidean and admits no E(2) action; the equivariance contributions do not apply. The non-equivariant GNN backbone (`gnn_encoder.py`) is paired with the same categorical CTMC machinery to evaluate engine generality.

SATLIB:

```bash
python -u edisco/train.py \
  --task mis \
  --diffusion_type categorical \
  --do_train --do_test \
  --learning_rate 0.0002 --weight_decay 0.0001 --lr_scheduler cosine-decay \
  --storage_path /your/storage/path \
  --training_split "/your/train_mis_sat/*gpickle" \
  --validation_split "/your/test_mis_sat/*gpickle" \
  --test_split "/your/test_mis_sat/*gpickle" \
  --batch_size 16 --num_epochs 50 \
  --n_layers 12 --hidden_dim 256 \
  --solver_type pndm --solver_steps 50 \
  --time_schedule linear --beta_min 0.1 --beta_max 1.5 \
  --use_activation_checkpoint
```

ER-[700-800] uses the same flags with `--diffusion_type gaussian` and the ER data paths. Sampling-based decoding uses `--parallel_sampling 4`.

## Ablations (Section 4.6)

### Encoder substitution (Table 5, decoder fixed at greedy)

EDISCO Full (architectural E(2)):

```bash
python -u edisco/train.py --task tsp --equivariant --do_test \
  --test_split /your/tsp500_test_lkh.txt --sparse_factor 50 \
  --solver_type deis --solver_steps 5 \
  --ckpt_path /your/tsp500_full_best.ckpt --resume_weight_only \
  --n_layers 12 --hidden_dim 256
```

Non-equivariant GNN, parameter-matched (Table 5 row "None"):

```bash
python -u edisco/train.py --task tsp --disable_equivariance --do_train --do_test \
  --training_split /your/tsp500_train_lkh.txt \
  --validation_split /your/tsp500_valid_lkh.txt \
  --test_split /your/tsp500_test_lkh.txt \
  --sparse_factor 50 --batch_size 16 --num_epochs 50 \
  --n_layers 12 --hidden_dim 256 \
  --solver_type pndm --solver_steps 50 --time_schedule linear
```

Add `--data_augmentation e2` to enable E(2) data augmentation (Table 5 row "Augmentation"). Add `--symmetry_loss` to enable the Sym-NCO soft regularizer (Table 5 row "Soft regularizer").

### Decoder choice (Table 6, encoder fixed at full EDISCO)

Greedy only:

```bash
python -u edisco/train.py --task tsp --equivariant --do_test \
  --decoder greedy \
  --test_split /your/tsp500_test_lkh.txt --sparse_factor 50 \
  --solver_type deis --solver_steps 5 \
  --ckpt_path /your/tsp500_full_best.ckpt --resume_weight_only
```

Greedy + 2-opt: same command with `--decoder greedy --two_opt_iterations 5000`.

Native Edge Expansion (proposed): same command with `--decoder nee`.

## Equivariance verification

```bash
python -m unittest tests.test_native_decoder_equivariance
python -m unittest tests.test_capacity_conditioning_equivariance
```

The TSP edge-logit consistency probe used in Table 7 is also exposed as a `--test_equivariance` flag of `edisco/train.py` for any trained checkpoint.
