# Reproduce Results for EDISCO

## Training

### Training on TSP50

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

python -u edisco/train.py \
  --task "tsp" \
  --wandb_logger_name "edisco_tsp50_continuous_categorical" \
  --diffusion_type "categorical" \
  --continuous_time \
  --equivariant \
  --do_train \
  --do_test \
  --learning_rate 0.0002 \
  --weight_decay 0.00001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "/your/storage/path" \
  --training_split "/your/tsp50_train_concorde.txt" \
  --validation_split "/your/tsp50_valid_concorde.txt" \
  --test_split "/your/tsp50_test_concorde.txt" \
  --batch_size 64 \
  --num_epochs 50 \
  --validation_examples 8 \
  --n_layers 12 \
  --hidden_dim 256 \
  --solver_type "pndm" \
  --solver_steps 50 \
  --time_schedule "linear" \
  --adaptive_mixing \
  --beta_min 0.1 \
  --beta_max 1.5
```

### Training on TSP100

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

python -u edisco/train.py \
  --task "tsp" \
  --wandb_logger_name "edisco_tsp100_continuous_categorical" \
  --diffusion_type "categorical" \
  --continuous_time \
  --equivariant \
  --do_train \
  --do_test \
  --learning_rate 0.0002 \
  --weight_decay 0.00001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "/your/storage/path" \
  --training_split "/your/tsp100_train_concorde.txt" \
  --validation_split "/your/tsp100_valid_concorde.txt" \
  --test_split "/your/tsp100_test_concorde.txt" \
  --batch_size 32 \
  --num_epochs 50 \
  --validation_examples 8 \
  --n_layers 12 \
  --hidden_dim 256 \
  --solver_type "pndm" \
  --solver_steps 50 \
  --time_schedule "linear" \
  --adaptive_mixing \
  --beta_min 0.1 \
  --beta_max 1.5
```

### Training on TSP500 (with curriculum learning from TSP100)

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

python -u edisco/train.py \
  --task "tsp" \
  --wandb_logger_name "edisco_tsp500_continuous_categorical" \
  --diffusion_type "categorical" \
  --continuous_time \
  --equivariant \
  --do_train \
  --do_test \
  --learning_rate 0.0002 \
  --weight_decay 0.00001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "/your/storage/path" \
  --training_split "/your/tsp500_train_lkh.txt" \
  --validation_split "/your/tsp500_valid_lkh.txt" \
  --test_split "/your/tsp500_test_lkh.txt" \
  --sparse_factor 50 \
  --batch_size 16 \
  --num_epochs 50 \
  --validation_examples 8 \
  --n_layers 12 \
  --hidden_dim 256 \
  --solver_type "pndm" \
  --solver_steps 50 \
  --time_schedule "linear" \
  --adaptive_mixing \
  --beta_min 0.1 \
  --beta_max 1.5 \
  --ckpt_path "/your/tsp100_model/ckpt_path/last.ckpt" \
  --resume_weight_only
```

### Training on TSP1000 (with curriculum learning from TSP100)

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

python -u edisco/train.py \
  --task "tsp" \
  --wandb_logger_name "edisco_tsp1000_continuous_categorical" \
  --diffusion_type "categorical" \
  --continuous_time \
  --equivariant \
  --do_train \
  --do_test \
  --learning_rate 0.0002 \
  --weight_decay 0.00001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "/your/storage/path" \
  --training_split "/your/tsp1000_train_lkh.txt" \
  --validation_split "/your/tsp1000_valid_lkh.txt" \
  --test_split "/your/tsp1000_test_lkh.txt" \
  --sparse_factor 100 \
  --batch_size 8 \
  --num_epochs 50 \
  --validation_examples 8 \
  --n_layers 12 \
  --hidden_dim 256 \
  --solver_type "pndm" \
  --solver_steps 50 \
  --time_schedule "linear" \
  --adaptive_mixing \
  --beta_min 0.1 \
  --beta_max 1.5 \
  --ckpt_path "/your/tsp100_model/ckpt_path/last.ckpt" \
  --resume_weight_only \
  --use_activation_checkpoint
```

### Training on TSP10000 (with curriculum learning from TSP500)

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

python -u edisco/train.py \
  --task "tsp" \
  --wandb_logger_name "edisco_tsp10000_continuous_categorical" \
  --diffusion_type "categorical" \
  --continuous_time \
  --equivariant \
  --do_train \
  --do_test \
  --learning_rate 0.0002 \
  --weight_decay 0.00001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "/your/storage/path" \
  --training_split "/your/tsp10000_train_lkh.txt" \
  --validation_split "/your/tsp10000_valid_lkh.txt" \
  --test_split "/your/tsp10000_test_lkh.txt" \
  --sparse_factor 100 \
  --batch_size 4 \
  --num_epochs 50 \
  --validation_examples 8 \
  --n_layers 12 \
  --hidden_dim 256 \
  --solver_type "pndm" \
  --solver_steps 50 \
  --time_schedule "linear" \
  --adaptive_mixing \
  --beta_min 0.1 \
  --beta_max 1.5 \
  --two_opt_iterations 5000 \
  --ckpt_path "/your/tsp500_model/ckpt_path/last.ckpt" \
  --resume_weight_only \
  --use_activation_checkpoint
```

### Training on CVRP (Capacitated Vehicle Routing Problem)

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

python -u edisco/train.py \
  --task "cvrp" \
  --wandb_logger_name "edisco_cvrp_continuous_categorical" \
  --diffusion_type "categorical" \
  --continuous_time \
  --equivariant \
  --do_train \
  --do_test \
  --learning_rate 0.0002 \
  --weight_decay 0.00001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "/your/storage/path" \
  --training_split "/your/cvrp_train_data.txt" \
  --validation_split "/your/cvrp_valid_data.txt" \
  --test_split "/your/cvrp_test_data.txt" \
  --batch_size 32 \
  --num_epochs 50 \
  --validation_examples 8 \
  --n_layers 8 \
  --hidden_dim 128 \
  --solver_type "pndm" \
  --solver_steps 50 \
  --time_schedule "linear" \
  --adaptive_mixing \
  --beta_min 0.1 \
  --beta_max 1.5
```

## Evaluation

### Evaluation on TSP100 with Greedy Decoding

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -u edisco/train.py \
  --task "tsp" \
  --wandb_logger_name "edisco_tsp100_test_greedy" \
  --diffusion_type "categorical" \
  --continuous_time \
  --equivariant \
  --do_test \
  --storage_path "/your/storage/path" \
  --training_split "/your/tsp100_train_concorde.txt" \
  --validation_split "/your/tsp100_valid_concorde.txt" \
  --test_split "/your/tsp100_test_concorde.txt" \
  --batch_size 32 \
  --n_layers 12 \
  --hidden_dim 256 \
  --solver_type "pndm" \
  --solver_steps 50 \
  --time_schedule "linear" \
  --adaptive_mixing \
  --ckpt_path "/your/edisco_tsp100/ckpt_path/best.ckpt" \
  --resume_weight_only
```

### Evaluation on TSP500 with Sampling (4x Parallel)

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -u edisco/train.py \
  --task "tsp" \
  --wandb_logger_name "edisco_tsp500_test_parallel4" \
  --diffusion_type "categorical" \
  --continuous_time \
  --equivariant \
  --do_test \
  --storage_path "/your/storage/path" \
  --training_split "/your/tsp500_train_lkh.txt" \
  --validation_split "/your/tsp500_valid_lkh.txt" \
  --test_split "/your/tsp500_test_lkh.txt" \
  --sparse_factor 50 \
  --batch_size 32 \
  --n_layers 12 \
  --hidden_dim 256 \
  --solver_type "pndm" \
  --solver_steps 50 \
  --time_schedule "linear" \
  --adaptive_mixing \
  --parallel_sampling 4 \
  --ckpt_path "/your/edisco_tsp500/ckpt_path/best.ckpt" \
  --resume_weight_only
```

### Evaluation on TSP1000 with Sampling (4x Sequential)

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -u edisco/train.py \
  --task "tsp" \
  --wandb_logger_name "edisco_tsp1000_test_sequential4" \
  --diffusion_type "categorical" \
  --continuous_time \
  --equivariant \
  --do_test \
  --storage_path "/your/storage/path" \
  --training_split "/your/tsp1000_train_lkh.txt" \
  --validation_split "/your/tsp1000_valid_lkh.txt" \
  --test_split "/your/tsp1000_test_lkh.txt" \
  --sparse_factor 100 \
  --batch_size 1 \
  --n_layers 12 \
  --hidden_dim 256 \
  --solver_type "pndm" \
  --solver_steps 50 \
  --time_schedule "linear" \
  --adaptive_mixing \
  --sequential_sampling 4 \
  --two_opt_iterations 5000 \
  --ckpt_path "/your/edisco_tsp1000/ckpt_path/best.ckpt" \
  --resume_weight_only
```

### Solver Comparison Evaluation

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Compare different ODE solvers
for solver in euler ddim pndm dpm2 deis rk4 heun; do
  python -u edisco/train.py \
    --task "tsp" \
    --wandb_logger_name "edisco_tsp500_solver_${solver}" \
    --diffusion_type "categorical" \
    --continuous_time \
    --equivariant \
    --do_test \
    --compare_solvers \
    --storage_path "/your/storage/path" \
    --test_split "/your/tsp500_test_lkh.txt" \
    --sparse_factor 50 \
    --batch_size 32 \
    --n_layers 12 \
    --hidden_dim 256 \
    --solver_type "${solver}" \
    --solver_steps 50 \
    --time_schedule "linear" \
    --adaptive_mixing \
    --ckpt_path "/your/edisco_tsp500/ckpt_path/best.ckpt" \
    --resume_weight_only
done
```

## Ablation Studies

### Ablation: Without E(2)-Equivariance

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

python -u edisco/train.py \
  --task "tsp" \
  --wandb_logger_name "edisco_tsp500_ablation_no_equivariance" \
  --diffusion_type "categorical" \
  --continuous_time \
  --disable_equivariance \
  --do_train \
  --do_test \
  --learning_rate 0.0002 \
  --weight_decay 0.00001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "/your/storage/path" \
  --training_split "/your/tsp500_train_lkh.txt" \
  --validation_split "/your/tsp500_valid_lkh.txt" \
  --test_split "/your/tsp500_test_lkh.txt" \
  --sparse_factor 50 \
  --batch_size 16 \
  --num_epochs 50 \
  --validation_examples 8 \
  --n_layers 12 \
  --hidden_dim 256 \
  --solver_type "pndm" \
  --solver_steps 50 \
  --time_schedule "linear" \
  --adaptive_mixing
```

### Ablation: Without Continuous-Time (Discrete-Time)

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

python -u edisco/train.py \
  --task "tsp" \
  --wandb_logger_name "edisco_tsp500_ablation_discrete_time" \
  --diffusion_type "categorical" \
  --disable_continuous_time \
  --equivariant \
  --do_train \
  --do_test \
  --learning_rate 0.0002 \
  --weight_decay 0.00001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "/your/storage/path" \
  --training_split "/your/tsp500_train_lkh.txt" \
  --validation_split "/your/tsp500_valid_lkh.txt" \
  --test_split "/your/tsp500_test_lkh.txt" \
  --sparse_factor 50 \
  --batch_size 16 \
  --num_epochs 50 \
  --validation_examples 8 \
  --n_layers 12 \
  --hidden_dim 256 \
  --inference_schedule "cosine" \
  --inference_diffusion_steps 50
```

### Ablation: Without Adaptive Mixing

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

python -u edisco/train.py \
  --task "tsp" \
  --wandb_logger_name "edisco_tsp500_ablation_no_adaptive_mixing" \
  --diffusion_type "categorical" \
  --continuous_time \
  --equivariant \
  --do_train \
  --do_test \
  --learning_rate 0.0002 \
  --weight_decay 0.00001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "/your/storage/path" \
  --training_split "/your/tsp500_train_lkh.txt" \
  --validation_split "/your/tsp500_valid_lkh.txt" \
  --test_split "/your/tsp500_test_lkh.txt" \
  --sparse_factor 50 \
  --batch_size 16 \
  --num_epochs 50 \
  --validation_examples 8 \
  --n_layers 12 \
  --hidden_dim 256 \
  --solver_type "pndm" \
  --solver_steps 50 \
  --time_schedule "linear" \
  --no_adaptive_mixing
```

## Model Efficiency Experiments

### EDISCO-Medium (2.6M parameters)

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

python -u edisco/train.py \
  --task "tsp" \
  --wandb_logger_name "edisco_medium_tsp500" \
  --diffusion_type "categorical" \
  --continuous_time \
  --equivariant \
  --do_train \
  --do_test \
  --learning_rate 0.0002 \
  --weight_decay 0.00001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "/your/storage/path" \
  --training_split "/your/tsp500_train_lkh.txt" \
  --validation_split "/your/tsp500_valid_lkh.txt" \
  --test_split "/your/tsp500_test_lkh.txt" \
  --sparse_factor 50 \
  --batch_size 16 \
  --num_epochs 50 \
  --validation_examples 8 \
  --n_layers 12 \
  --hidden_dim 128 \
  --solver_type "pndm" \
  --solver_steps 50 \
  --time_schedule "linear" \
  --adaptive_mixing
```

### EDISCO-Small (1.4M parameters)

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

python -u edisco/train.py \
  --task "tsp" \
  --wandb_logger_name "edisco_small_tsp500" \
  --diffusion_type "categorical" \
  --continuous_time \
  --equivariant \
  --do_train \
  --do_test \
  --learning_rate 0.0002 \
  --weight_decay 0.00001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "/your/storage/path" \
  --training_split "/your/tsp500_train_lkh.txt" \
  --validation_split "/your/tsp500_valid_lkh.txt" \
  --test_split "/your/tsp500_test_lkh.txt" \
  --sparse_factor 50 \
  --batch_size 16 \
  --num_epochs 50 \
  --validation_examples 8 \
  --n_layers 8 \
  --hidden_dim 128 \
  --solver_type "pndm" \
  --solver_steps 50 \
  --time_schedule "linear" \
  --adaptive_mixing
```

## Generalization Experiments

### Cross-Size Evaluation (Train on TSP1000, Test on All)

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Test the TSP1000-trained model on different problem sizes
for size in 50 100 500 10000; do
  python -u edisco/train.py \
    --task "tsp" \
    --wandb_logger_name "edisco_generalization_tsp1000_to_tsp${size}" \
    --diffusion_type "categorical" \
    --continuous_time \
    --equivariant \
    --do_test \
    --storage_path "/your/storage/path" \
    --test_split "/your/tsp${size}_test_data.txt" \
    --batch_size 32 \
    --n_layers 12 \
    --hidden_dim 256 \
    --solver_type "pndm" \
    --solver_steps 50 \
    --time_schedule "linear" \
    --adaptive_mixing \
    --ckpt_path "/your/edisco_tsp1000/ckpt_path/best.ckpt" \
    --resume_weight_only
done
```

## Robustness Experiments

### Training with Limited Data (10%, 20%, 40%, 60%, 80%)

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

for percent in 10 20 40 60 80; do
  # shellcheck disable=SC2155
  export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
  echo "Training with ${percent}% data, WANDB_ID is $WANDB_RUN_ID"
  
  python -u edisco/train.py \
    --task "tsp" \
    --wandb_logger_name "edisco_tsp50_data_${percent}percent" \
    --diffusion_type "categorical" \
    --continuous_time \
    --equivariant \
    --do_train \
    --do_test \
    --learning_rate 0.0002 \
    --weight_decay 0.00001 \
    --lr_scheduler "cosine-decay" \
    --storage_path "/your/storage/path" \
    --training_split "/your/tsp50_train_concorde_${percent}percent.txt" \
    --validation_split "/your/tsp50_valid_concorde.txt" \
    --test_split "/your/tsp50_test_concorde.txt" \
    --batch_size 64 \
    --num_epochs 50 \
    --validation_examples 8 \
    --n_layers 12 \
    --hidden_dim 256 \
    --solver_type "pndm" \
    --solver_steps 50 \
    --time_schedule "linear" \
    --adaptive_mixing
done
```

### Training with Suboptimal Data (Farthest Insertion Heuristic)

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

python -u edisco/train.py \
  --task "tsp" \
  --wandb_logger_name "edisco_tsp50_farthest_insertion" \
  --diffusion_type "categorical" \
  --continuous_time \
  --equivariant \
  --do_train \
  --do_test \
  --learning_rate 0.0002 \
  --weight_decay 0.00001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "/your/storage/path" \
  --training_split "/your/tsp50_train_farthest_insertion.txt" \
  --validation_split "/your/tsp50_valid_concorde.txt" \
  --test_split "/your/tsp50_test_concorde.txt" \
  --batch_size 64 \
  --num_epochs 50 \
  --validation_examples 8 \
  --n_layers 12 \
  --hidden_dim 256 \
  --solver_type "pndm" \
  --solver_steps 50 \
  --time_schedule "linear" \
  --adaptive_mixing
```

## Test Equivariance Property

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

python -u edisco/train.py \
  --task "tsp" \
  --wandb_logger_name "edisco_equivariance_test" \
  --diffusion_type "categorical" \
  --continuous_time \
  --equivariant \
  --test_equivariance \
  --storage_path "/your/storage/path" \
  --test_split "/your/tsp100_test_concorde.txt" \
  --batch_size 1 \
  --n_layers 12 \
  --hidden_dim 256 \
  --solver_type "pndm" \
  --solver_steps 50 \
  --ckpt_path "/your/edisco_tsp100/ckpt_path/best.ckpt" \
  --resume_weight_only
```

---

## Euclidean Steiner Tree Experiments

### Training on Steiner10 (10 terminals)

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

python -u edisco/train.py \
  --task "steiner" \
  --wandb_logger_name "edisco_steiner10_continuous_categorical" \
  --diffusion_type "categorical" \
  --continuous_time \
  --equivariant \
  --do_train \
  --do_test \
  --learning_rate 0.0001 \
  --weight_decay 0.00001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "/your/storage/path" \
  --training_split "steiner10_train.txt" \
  --validation_split "steiner10_valid.txt" \
  --test_split "steiner10_test.txt" \
  --batch_size 32 \
  --num_epochs 100 \
  --validation_examples 100 \
  --n_layers 8 \
  --hidden_dim 128 \
  --solver_type "pndm" \
  --solver_steps 50 \
  --time_schedule "linear" \
  --adaptive_mixing \
  --beta_min 0.1 \
  --beta_max 1.5
```

### Training on Steiner20 (20 terminals)

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

python -u edisco/train.py \
  --task "steiner" \
  --wandb_logger_name "edisco_steiner20_continuous_categorical" \
  --diffusion_type "categorical" \
  --continuous_time \
  --equivariant \
  --do_train \
  --do_test \
  --learning_rate 0.0001 \
  --weight_decay 0.00001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "/your/storage/path" \
  --training_split "steiner20_train.txt" \
  --validation_split "steiner20_valid.txt" \
  --test_split "steiner20_test.txt" \
  --batch_size 16 \
  --num_epochs 100 \
  --validation_examples 100 \
  --n_layers 8 \
  --hidden_dim 128 \
  --solver_type "pndm" \
  --solver_steps 50 \
  --time_schedule "linear" \
  --adaptive_mixing \
  --beta_min 0.1 \
  --beta_max 1.5
```

### Training on Steiner50 (50 terminals, with sparse mode)

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

python -u edisco/train.py \
  --task "steiner" \
  --wandb_logger_name "edisco_steiner50_continuous_categorical" \
  --diffusion_type "categorical" \
  --continuous_time \
  --equivariant \
  --do_train \
  --do_test \
  --learning_rate 0.00005 \
  --weight_decay 0.00001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "/your/storage/path" \
  --training_split "steiner50_train.txt" \
  --validation_split "steiner50_valid.txt" \
  --test_split "steiner50_test.txt" \
  --sparse_factor 20 \
  --batch_size 8 \
  --num_epochs 150 \
  --validation_examples 100 \
  --n_layers 10 \
  --hidden_dim 128 \
  --solver_type "pndm" \
  --solver_steps 50 \
  --time_schedule "linear" \
  --adaptive_mixing \
  --beta_min 0.1 \
  --beta_max 1.5
```

### Evaluation on Steiner Tree with E(2)-Equivariance Verification

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

python -u edisco/train.py \
  --task "steiner" \
  --wandb_logger_name "edisco_steiner_equivariance_test" \
  --diffusion_type "categorical" \
  --continuous_time \
  --equivariant \
  --test_equivariance \
  --storage_path "/your/storage/path" \
  --test_split "steiner10_test.txt" \
  --batch_size 1 \
  --n_layers 8 \
  --hidden_dim 128 \
  --solver_type "pndm" \
  --solver_steps 50 \
  --ckpt_path "/your/edisco_steiner10/ckpt_path/best.ckpt" \
  --resume_weight_only
```

## Maximum Independent Set (MIS)

**Note**: MIS is NOT a geometric problem. EDISCO uses standard GNN (not EGNN) for MIS.
The E(2)-equivariance is automatically disabled for MIS, but continuous-time diffusion
with ODE solvers (PNDM, DEIS, etc.) is still used for superior sampling quality.

### Training on SATLIB graphs with Categorical Diffusion

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

python -u edisco/train.py \
  --task "mis" \
  --wandb_logger_name "edisco_mis_categorical_satlib" \
  --diffusion_type "categorical" \
  --do_train \
  --do_test \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "/your/storage/path" \
  --training_split "/your/train_mis_sat/*gpickle" \
  --validation_split "/your/test_mis_sat/*gpickle" \
  --test_split "/your/test_mis_sat/*gpickle" \
  --batch_size 16 \
  --num_epochs 50 \
  --validation_examples 8 \
  --n_layers 12 \
  --hidden_dim 256 \
  --solver_type "pndm" \
  --solver_steps 50 \
  --time_schedule "linear" \
  --adaptive_mixing \
  --beta_min 0.1 \
  --beta_max 1.5 \
  --use_activation_checkpoint
```

### Training on ER-[700-800] graphs with Gaussian Diffusion

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# shellcheck disable=SC2155
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "WANDB_ID is $WANDB_RUN_ID"

python -u edisco/train.py \
  --task "mis" \
  --wandb_logger_name "edisco_mis_gaussian_er" \
  --diffusion_type "gaussian" \
  --do_train \
  --do_test \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "/your/storage/path" \
  --training_split "/your/data_er/train/*gpickle" \
  --training_split_label_dir "/your/data_er/train_annotations/" \
  --validation_split "/your/data_er/validation/*gpickle" \
  --test_split "/your/data_er/test/*gpickle" \
  --batch_size 4 \
  --num_epochs 50 \
  --validation_examples 8 \
  --n_layers 12 \
  --hidden_dim 256 \
  --solver_type "pndm" \
  --solver_steps 50 \
  --time_schedule "linear" \
  --adaptive_mixing \
  --beta_min 0.1 \
  --beta_max 1.5 \
  --use_activation_checkpoint
```

### Evaluation on SATLIB graphs with Categorical Diffusion and Greedy Decoding

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -u edisco/train.py \
  --task "mis" \
  --wandb_logger_name "edisco_mis_categorical_satlib_test" \
  --diffusion_type "categorical" \
  --do_test \
  --storage_path "/your/storage/path" \
  --training_split "/your/train_mis_sat/*gpickle" \
  --validation_split "/your/test_mis_sat/*gpickle" \
  --test_split "/your/test_mis_sat/*gpickle" \
  --batch_size 16 \
  --n_layers 12 \
  --hidden_dim 256 \
  --solver_type "pndm" \
  --solver_steps 50 \
  --time_schedule "linear" \
  --adaptive_mixing \
  --beta_min 0.1 \
  --beta_max 1.5 \
  --ckpt_path "/your/mis_sat_categorical/ckpt_path/last.ckpt" \
  --resume_weight_only
```

### Evaluation on ER-[700-800] graphs with Gaussian Diffusion and Parallel Sampling (4x)

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -u edisco/train.py \
  --task "mis" \
  --wandb_logger_name "edisco_mis_gaussian_er_test_parallel4" \
  --diffusion_type "gaussian" \
  --do_test \
  --storage_path "/your/storage/path" \
  --training_split "/your/data_er/train/*gpickle" \
  --training_split_label_dir "/your/data_er/train_annotations/" \
  --validation_split "/your/data_er/validation/*gpickle" \
  --test_split "/your/data_er/test/*gpickle" \
  --batch_size 4 \
  --n_layers 12 \
  --hidden_dim 256 \
  --solver_type "pndm" \
  --solver_steps 50 \
  --time_schedule "linear" \
  --adaptive_mixing \
  --beta_min 0.1 \
  --beta_max 1.5 \
  --parallel_sampling 4 \
  --use_activation_checkpoint \
  --ckpt_path "/your/mis_er_gaussian/ckpt_path/last.ckpt" \
  --resume_weight_only
```

### MIS Solver Comparison (ODE Solvers)

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

# Compare different ODE solvers for MIS
for solver in euler ddim pndm dpm2 deis rk4 heun; do
  python -u edisco/train.py \
    --task "mis" \
    --wandb_logger_name "edisco_mis_solver_${solver}" \
    --diffusion_type "categorical" \
    --do_test \
    --storage_path "/your/storage/path" \
    --training_split "/your/train_mis_sat/*gpickle" \
    --validation_split "/your/test_mis_sat/*gpickle" \
    --test_split "/your/test_mis_sat/*gpickle" \
    --batch_size 16 \
    --n_layers 12 \
    --hidden_dim 256 \
    --solver_type "${solver}" \
    --solver_steps 50 \
    --time_schedule "linear" \
    --adaptive_mixing \
    --beta_min 0.1 \
    --beta_max 1.5 \
    --ckpt_path "/your/mis_sat_categorical/ckpt_path/last.ckpt" \
    --resume_weight_only
done
```

