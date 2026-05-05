"""Debug individual diffusion steps to understand what's happening."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'edisco'))

import torch
import torch.nn.functional as F
from models.egnn_encoder import EGNNEncoder

# Load checkpoint
ckpt_path = 'pretrained/edisco_tsp20.ckpt'
checkpoint = torch.load(ckpt_path, map_location='cuda')
state_dict = checkpoint['state_dict']

# Create encoder
encoder = EGNNEncoder(
    n_layers=12, hidden_dim=256, node_dim=64, edge_dim=64, time_dim=128,
    coord_dim=2, out_channels=2, sparse=False, dense_only=True,
)

# Load weights
encoder_state = {k[6:]: v for k, v in state_dict.items() if k.startswith('model.')}
encoder.load_state_dict(encoder_state, strict=True)
encoder.eval().cuda()

print("=" * 60)
print("Debug: Model behavior at different timesteps")
print("=" * 60)

# Fixed coords
torch.manual_seed(42)
coords = torch.rand(1, 20, 2, device='cuda')

# Test 1: Random binary input (like diffusion initialization)
print("\n1. Random binary input (50% ones - like x_T):")
adj_binary = torch.randint(0, 2, (1, 20, 20), device='cuda', dtype=torch.float32)
print(f"   Input: {adj_binary.sum().item():.0f} ones ({adj_binary.mean().item()*100:.1f}%)")

for t_val in [1.0, 0.5, 0.1, 0.0]:
    t = torch.tensor([t_val], device='cuda')
    with torch.no_grad():
        logits = encoder(coords, adj_binary, t, None)
    probs = F.softmax(logits, dim=-1)
    pred_edges = (probs[..., 1] > 0.5).sum().item()
    mean_edge_prob = probs[..., 1].mean().item()
    print(f"   t={t_val}: mean_edge_prob={mean_edge_prob:.4f}, pred_edges={pred_edges}")

# Test 2: Sparse binary input (like ground truth TSP)
print("\n2. Sparse binary input (~20 edges - like ground truth):")
adj_sparse = torch.zeros(1, 20, 20, device='cuda')
# Create a random tour
perm = torch.randperm(20)
for i in range(20):
    adj_sparse[0, perm[i], perm[(i+1)%20]] = 1
    adj_sparse[0, perm[(i+1)%20], perm[i]] = 1
print(f"   Input: {adj_sparse.sum().item():.0f} ones ({adj_sparse.mean().item()*100:.1f}%)")

for t_val in [1.0, 0.5, 0.1, 0.0]:
    t = torch.tensor([t_val], device='cuda')
    with torch.no_grad():
        logits = encoder(coords, adj_sparse, t, None)
    probs = F.softmax(logits, dim=-1)
    pred_edges = (probs[..., 1] > 0.5).sum().item()
    mean_edge_prob = probs[..., 1].mean().item()
    print(f"   t={t_val}: mean_edge_prob={mean_edge_prob:.4f}, pred_edges={pred_edges}")

# Test 3: All zeros input
print("\n3. All zeros input:")
adj_zeros = torch.zeros(1, 20, 20, device='cuda')
print(f"   Input: 0 ones (0%)")

for t_val in [1.0, 0.5, 0.1, 0.0]:
    t = torch.tensor([t_val], device='cuda')
    with torch.no_grad():
        logits = encoder(coords, adj_zeros, t, None)
    probs = F.softmax(logits, dim=-1)
    pred_edges = (probs[..., 1] > 0.5).sum().item()
    mean_edge_prob = probs[..., 1].mean().item()
    print(f"   t={t_val}: mean_edge_prob={mean_edge_prob:.4f}, pred_edges={pred_edges}")

# Test 4: All ones input
print("\n4. All ones input:")
adj_ones = torch.ones(1, 20, 20, device='cuda')
print(f"   Input: 400 ones (100%)")

for t_val in [1.0, 0.5, 0.1, 0.0]:
    t = torch.tensor([t_val], device='cuda')
    with torch.no_grad():
        logits = encoder(coords, adj_ones, t, None)
    probs = F.softmax(logits, dim=-1)
    pred_edges = (probs[..., 1] > 0.5).sum().item()
    mean_edge_prob = probs[..., 1].mean().item()
    print(f"   t={t_val}: mean_edge_prob={mean_edge_prob:.4f}, pred_edges={pred_edges}")

# Test 5: Simulate a few diffusion steps
print("\n5. Simulating diffusion (starting from random binary):")
from diffusion.exact_ctmc import ExactCTMCPosterior
from utils.ode_solvers import get_time_schedule

posterior = ExactCTMCPosterior(beta_min=0.1, beta_max=1.5)
timesteps = get_time_schedule('linear', 50).cuda()

x_t = torch.randint(0, 2, (1, 20, 20), device='cuda', dtype=torch.float32)
print(f"   Initial x_T: {x_t.sum().item():.0f} ones")

# Run a few steps
for step in [0, 5, 10, 25, 49]:
    t = timesteps[step]
    t_next = timesteps[step + 1]

    with torch.no_grad():
        logits = encoder(coords, x_t, t, None)

    probs = F.softmax(logits, dim=-1)
    x0_pred = probs[..., 1].clamp(0, 1)

    mean_x0_pred = x0_pred.mean().item()
    x0_edges = (x0_pred > 0.5).sum().item()

    # Sample from posterior
    x_t = posterior.sample(x_t, x0_pred, t, t_next, deterministic=False)
    x_t_ones = x_t.sum().item()

    print(f"   Step {step}: t={t.item():.3f}->t_next={t_next.item():.3f}, "
          f"mean_x0_pred={mean_x0_pred:.4f}, x0_edges={x0_edges}, x_t_ones={x_t_ones:.0f}")

# Final prediction
with torch.no_grad():
    logits = encoder(coords, x_t, torch.tensor([0.0], device='cuda'), None)
probs = F.softmax(logits, dim=-1)
final_edges = (probs[..., 1] > 0.5).sum().item()
print(f"   Final: edges predicted = {final_edges}")
