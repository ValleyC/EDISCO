"""Visualization utilities for EDISCO"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import seaborn as sns


def visualize_diffusion_process(model, coords, adj_matrix, tour_gt, device='cuda', n_steps=6):
    """
    Visualize the continuous-time diffusion process
    
    Args:
        model: trained EDISCO model
        coords: TSP coordinates (n_nodes, 2)
        adj_matrix: ground truth adjacency matrix (n_nodes, n_nodes)
        tour_gt: ground truth tour (n_nodes,)
        device: computation device
        n_steps: number of time points to visualize
    
    Returns:
        fig: matplotlib figure
    """
    model.eval()
    
    # Prepare inputs
    if not torch.is_tensor(coords):
        coords = torch.tensor(coords)
    if not torch.is_tensor(adj_matrix):
        adj_matrix = torch.tensor(adj_matrix)
    if not torch.is_tensor(tour_gt):
        tour_gt = torch.tensor(tour_gt)
    
    coords = coords.unsqueeze(0).to(device)
    adj_matrix = adj_matrix.unsqueeze(0).to(device)
    n_nodes = coords.shape[1]
    
    with torch.no_grad():
        # Time points to visualize
        time_points = np.linspace(1.0, 0.0, n_steps)
        states = []
        predictions = []
        
        # Forward diffusion and predictions at different times
        for t in time_points[:-1]:
            t_tensor = torch.tensor([t], device=device)
            
            # Sample noisy state
            xt = model.diffusion.sample_forward(adj_matrix, t_tensor, device)
            
            # Predict X_0 from noisy state
            x0_logits = model.forward(coords, xt, t_tensor)
            x0_probs = F.softmax(x0_logits, dim=-1)[0, :, :, 1]  # Prob of edge=1
            
            states.append((t, xt[0].cpu().numpy()))
            predictions.append(x0_probs.cpu().numpy())
        
        # Final sampling
        adj_probs, _ = model.sample(coords, n_steps=50, device=device)
        
        # Decode tour
        from .tsp_utils import merge_tours_dense, compute_tour_length
        pred_tour = merge_tours_dense(adj_probs[0].cpu(), coords[0].cpu())
    
    # Compute metrics
    coords_np = coords[0].cpu().numpy()
    tour_gt_np = tour_gt.numpy() if torch.is_tensor(tour_gt) else tour_gt
    pred_length = compute_tour_length(coords_np, pred_tour)
    gt_length = compute_tour_length(coords_np, tour_gt_np)
    gap = (pred_length - gt_length) / gt_length * 100
    
    # Create visualization
    fig = plt.figure(figsize=(20, 12))
    
    # Plot diffusion states
    for i, ((t_val, state), pred) in enumerate(zip(states[:4], predictions[:4])):
        # Noisy state X_t
        ax = plt.subplot(3, 5, i + 1)
        im = ax.imshow(state, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(f'$X_t$ at $t={t_val:.1f}$')
        ax.set_xlabel('To')
        ax.set_ylabel('From')
        
        # Predicted X_0 probabilities
        ax = plt.subplot(3, 5, i + 6)
        im = ax.imshow(pred, cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(f'$P(X_0=1|X_t)$ at $t={t_val:.1f}$')
        ax.set_xlabel('To')
        ax.set_ylabel('From')
    
    # Ground truth tour
    ax = plt.subplot(3, 5, 11)
    plot_tour(ax, coords_np, tour_gt_np, 'Ground Truth', gt_length)
    
    # Predicted tour
    ax = plt.subplot(3, 5, 12)
    plot_tour(ax, coords_np, pred_tour, f'Predicted (Gap: {gap:.2f}%)', pred_length)
    
    # Final edge probabilities
    ax = plt.subplot(3, 5, 13)
    im = ax.imshow(adj_probs[0].cpu().numpy(), cmap='hot', interpolation='nearest', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title('Final Edge Probabilities')
    ax.set_xlabel('To')
    ax.set_ylabel('From')
    
    # Time evolution plot
    ax = plt.subplot(3, 5, 14)
    times = [t for t, _ in states]
    mean_probs = [pred.mean() for pred in predictions]
    std_probs = [pred.std() for pred in predictions]
    ax.errorbar(times, mean_probs, yerr=std_probs, marker='o')
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('Mean Edge Probability')
    ax.set_title('Evolution of Edge Predictions')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'EDISCO Continuous-Time Diffusion Process (n={n_nodes} cities)', fontsize=16)
    plt.tight_layout()
    
    return fig


def plot_tour(ax, coords, tour, title, length):
    """
    Plot a TSP tour
    
    Args:
        ax: matplotlib axis
        coords: node coordinates (n_nodes, 2)
        tour: tour sequence (n_nodes,)
        title: plot title
        length: tour length
    """
    n_nodes = len(tour)
    
    # Plot edges
    for i in range(n_nodes):
        start = coords[tour[i]]
        end = coords[tour[(i + 1) % n_nodes]]
        ax.plot([start[0], end[0]], [start[1], end[1]], 'b-', alpha=0.6)
    
    # Plot nodes
    ax.scatter(coords[:, 0], coords[:, 1], c='red', s=100, zorder=2)
    
    # Add node labels
    for i in range(n_nodes):
        ax.text(coords[i, 0], coords[i, 1], str(i), 
                fontsize=8, ha='center', va='center')
    
    ax.set_title(f'{title}\nLength: {length:.4f}')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')


def visualize_tour(coords, tour, title='TSP Tour'):
    """
    Visualize a single TSP tour
    
    Args:
        coords: node coordinates (n_nodes, 2)
        tour: tour sequence (n_nodes,)
        title: plot title
    
    Returns:
        fig: matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    if torch.is_tensor(coords):
        coords = coords.numpy()
    if torch.is_tensor(tour):
        tour = tour.numpy()
    
    from .tsp_utils import compute_tour_length
    length = compute_tour_length(coords, tour)
    
    plot_tour(ax, coords, tour, title, length)
    
    return fig


def plot_training_curves(train_losses, val_gaps, val_epochs=None):
    """
    Plot training loss and validation gap curves
    
    Args:
        train_losses: list of training losses
        val_gaps: list of validation gaps
        val_epochs: epochs where validation was performed
    
    Returns:
        fig: matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training loss
    ax1.plot(train_losses, label='Training Loss')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Validation gap
    if val_epochs is None:
        val_epochs = list(range(len(val_gaps)))
    
    ax2.plot(val_epochs, val_gaps, marker='o', label='Validation Gap')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Gap (%)')
    ax2.set_title('Validation Gap')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.suptitle('EDISCO Training Progress')
    plt.tight_layout()
    
    return fig


def visualize_attention_weights(model, coords, adj_matrix, layer_idx=0):
    """
    Visualize attention/message weights in EGNN layers
    
    Args:
        model: EDISCO model
        coords: node coordinates
        adj_matrix: adjacency matrix
        layer_idx: which EGNN layer to visualize
    
    Returns:
        fig: matplotlib figure
    """
    model.eval()
    
    if not torch.is_tensor(coords):
        coords = torch.tensor(coords)
    if not torch.is_tensor(adj_matrix):
        adj_matrix = torch.tensor(adj_matrix)
    
    # Hook to capture intermediate activations
    activations = {}
    
    def hook_fn(module, input, output):
        activations['messages'] = output
    
    # Register hook
    hook = model.model.egnn.layers[layer_idx].message_mlp.register_forward_hook(hook_fn)
    
    # Forward pass
    with torch.no_grad():
        t = torch.tensor([0.5])  # Middle of diffusion
        _ = model.forward(coords.unsqueeze(0), adj_matrix.unsqueeze(0), t)
    
    # Remove hook
    hook.remove()
    
    # Visualize message strengths
    messages = activations['messages'][0]  # Remove batch dimension
    message_strength = messages.norm(dim=-1).numpy()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(message_strength, cmap='hot', interpolation='nearest')
    plt.colorbar(im, ax=ax)
    ax.set_title(f'Message Strengths in EGNN Layer {layer_idx}')
    ax.set_xlabel('To Node')
    ax.set_ylabel('From Node')
    
    return fig


def create_diffusion_animation(model, coords, adj_matrix, n_frames=50, device='cuda'):
    """
    Create an animation of the reverse diffusion process
    
    Args:
        model: trained EDISCO model
        coords: TSP coordinates
        adj_matrix: ground truth adjacency matrix
        n_frames: number of animation frames
        device: computation device
    
    Returns:
        anim: matplotlib animation
    """
    model.eval()
    
    if not torch.is_tensor(coords):
        coords = torch.tensor(coords)
    if not torch.is_tensor(adj_matrix):
        adj_matrix = torch.tensor(adj_matrix)
    
    coords = coords.unsqueeze(0).to(device)
    adj_matrix = adj_matrix.unsqueeze(0).to(device)
    
    # Run reverse diffusion and collect states
    states = []
    
    with torch.no_grad():
        # Initialize with noise
        xt = torch.randint(0, 2, adj_matrix.shape, device=device).float()
        
        # Time schedule
        timesteps = torch.linspace(1.0, 0.0, n_frames + 1, device=device)[:-1]
        
        for i in range(len(timesteps) - 1):
            t = timesteps[i].item()
            t_next = timesteps[i + 1].item()
            dt = t_next - t
            
            t_tensor = torch.full((1,), t, device=device)
            
            # Predict X_0
            x0_logits = model.forward(coords, xt, t_tensor)
            
            # Store state
            x0_probs = F.softmax(x0_logits, dim=-1)[0, :, :, 1]
            states.append(x0_probs.cpu().numpy())
            
            # Sample next state
            if t_next > 0.01:
                xt = model.diffusion.sample_reverse_simple(xt, x0_logits, t, dt, device)
            else:
                xt = x0_logits.argmax(dim=-1).float()
    
    # Create animation
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(states[0], cmap='hot', vmin=0, vmax=1, animated=True)
    plt.colorbar(im, ax=ax)
    ax.set_title('EDISCO Reverse Diffusion Process')
    ax.set_xlabel('To Node')
    ax.set_ylabel('From Node')
    
    def update(frame):
        im.set_array(states[frame])
        ax.set_title(f'EDISCO Reverse Diffusion (t={1-frame/n_frames:.2f})')
        return [im]
    
    anim = FuncAnimation(fig, update, frames=n_frames, interval=100, blit=True)
    plt.close(fig)
    
    return anim