"""Debug checkpoint loading to understand the exact configuration."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'edisco'))

import torch

ckpt_path = 'pretrained/edisco_tsp20.ckpt'
checkpoint = torch.load(ckpt_path, map_location='cpu')

print("=" * 60)
print("Checkpoint Analysis")
print("=" * 60)

print("\n1. Top-level keys:")
for k in checkpoint.keys():
    print(f"  {k}")

# Check hyperparameters
print("\n2. Hyperparameters:")
hparams = checkpoint.get('hyper_parameters', {})
if hparams:
    param_args = hparams.get('param_args', None)
    if param_args:
        print(f"  n_layers: {getattr(param_args, 'n_layers', 'N/A')}")
        print(f"  hidden_dim: {getattr(param_args, 'hidden_dim', 'N/A')}")
        print(f"  node_dim: {getattr(param_args, 'node_dim', 'N/A')}")
        print(f"  edge_dim: {getattr(param_args, 'edge_dim', 'N/A')}")
        print(f"  time_dim: {getattr(param_args, 'time_dim', 'N/A')}")
        print(f"  sparse_factor: {getattr(param_args, 'sparse_factor', 'N/A')}")
        print(f"  equivariant: {getattr(param_args, 'equivariant', 'N/A')}")
        print(f"  continuous_time: {getattr(param_args, 'continuous_time', 'N/A')}")
        print(f"  beta_min: {getattr(param_args, 'beta_min', 'N/A')}")
        print(f"  beta_max: {getattr(param_args, 'beta_max', 'N/A')}")
    else:
        print("  param_args not found in hyper_parameters")
        print(f"  Raw hyper_parameters: {hparams}")
else:
    print("  No hyper_parameters found")

# Check state dict structure
print("\n3. State dict prefixes:")
state_dict = checkpoint.get('state_dict', {})
prefixes = {}
for k in state_dict.keys():
    prefix = k.split('.')[0]
    if prefix not in prefixes:
        prefixes[prefix] = 0
    prefixes[prefix] += 1
for prefix, count in prefixes.items():
    print(f"  {prefix}: {count} keys")

# Check specific layer structure
print("\n4. Layer structure (first model layer):")
for k in sorted(state_dict.keys()):
    if 'model.layers.0.' in k:
        print(f"  {k}: {state_dict[k].shape}")

# Check output layer
print("\n5. Output layer:")
for k in sorted(state_dict.keys()):
    if 'model.out.' in k:
        print(f"  {k}: {state_dict[k].shape}")

# Test loading into EGNNEncoder
print("\n6. Testing model loading...")
from models.egnn_encoder import EGNNEncoder

# Get dimensions from weights
layer_keys = [k for k in state_dict.keys() if 'model.layers.' in k]
layer_nums = set(int(k.split('.')[2]) for k in layer_keys if k.split('.')[2].isdigit())
n_layers = max(layer_nums) + 1 if layer_nums else 12

hidden_dim = state_dict.get('model.time_embed.0.weight', torch.zeros(256, 128)).shape[0]
node_dim = state_dict.get('model.node_embed', torch.zeros(1, 1, 64)).shape[-1]
edge_dim = state_dict.get('model.edge_embed.weight', torch.zeros(64, 1)).shape[0]
time_dim = state_dict.get('model.time_embed.2.weight', torch.zeros(128, 256)).shape[0]

print(f"  Inferred: n_layers={n_layers}, hidden_dim={hidden_dim}, node_dim={node_dim}, edge_dim={edge_dim}, time_dim={time_dim}")

# Create encoder
encoder = EGNNEncoder(
    n_layers=n_layers,
    hidden_dim=hidden_dim,
    node_dim=node_dim,
    edge_dim=edge_dim,
    time_dim=time_dim,
    coord_dim=2,
    out_channels=2,
    sparse=False,
    dense_only=True,
)

# Prepare state dict
encoder_state = {}
for k, v in state_dict.items():
    if k.startswith('model.'):
        new_key = k[6:]
        encoder_state[new_key] = v

# Check what keys the encoder expects
print("\n7. Expected vs provided keys:")
encoder_keys = set(encoder.state_dict().keys())
provided_keys = set(encoder_state.keys())

missing = encoder_keys - provided_keys
unexpected = provided_keys - encoder_keys

print(f"  Encoder expects: {len(encoder_keys)} keys")
print(f"  Checkpoint provides: {len(provided_keys)} keys")
print(f"  Missing: {len(missing)} keys")
print(f"  Unexpected: {len(unexpected)} keys")

if missing:
    print(f"\n  Missing keys (first 10):")
    for k in list(missing)[:10]:
        print(f"    {k}")

if unexpected:
    print(f"\n  Unexpected keys (first 10):")
    for k in list(unexpected)[:10]:
        print(f"    {k}")

# Load and check
missing_keys, unexpected_keys = encoder.load_state_dict(encoder_state, strict=False)
print(f"\n  load_state_dict result:")
print(f"    Missing: {len(missing_keys)}")
print(f"    Unexpected: {len(unexpected_keys)}")

# Test forward pass
print("\n8. Testing forward pass...")
encoder.eval()
with torch.no_grad():
    coords = torch.rand(1, 20, 2)
    adj = torch.rand(1, 20, 20)
    t = torch.tensor([0.5])

    output = encoder(coords, adj, t, None)
    print(f"  Input coords: {coords.shape}")
    print(f"  Input adj: {adj.shape}")
    print(f"  Input t: {t.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output logits range: [{output.min():.4f}, {output.max():.4f}]")
    print(f"  Output logits mean: {output.mean():.4f}")

    # Check class probabilities
    probs = torch.softmax(output, dim=-1)
    print(f"  Class 0 prob mean: {probs[..., 0].mean():.4f}")
    print(f"  Class 1 prob mean: {probs[..., 1].mean():.4f}")

    # Check if model predicts edges
    predictions = output.argmax(dim=-1)
    print(f"  Predicted edges: {predictions.sum().item()}")
