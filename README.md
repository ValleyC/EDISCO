# EDISCO: Equivariant Discrete Diffusion for Euclidean Combinatorial Optimization

**Anonymous Implementation for NeurIPS 2026 Submission**

EDISCO is a discrete diffusion model for Euclidean combinatorial optimization problems whose generative distribution over node-index solutions is exactly E(2)-invariant by construction. The pipeline composes an E(2)-equivariant score network, a categorical continuous-time Markov chain over edge variables, and Native Edge Expansion (NEE) decoding that operates only on invariant quantities.

## Visualization

![EDISCO Visualization](edisco_visualization.jpg)

## Setup

```bash
conda env create -f environment.yml
conda activate edisco
```

EDISCO uses a Cython extension for diffusion-heatmap merging:

```bash
cd edisco/utils/cython_merge
python setup.py build_ext --inplace
cd -
```

## Data

See the `data` folder for dataset preparation instructions.

## Reproduction

See [reproducing_scripts.md](reproducing_scripts.md) for training and evaluation commands across TSP, CVRP, ESTP, and MIS.
