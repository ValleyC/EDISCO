"""Check checkpoint structure to understand key names."""
import torch

ckpt_path = 'pretrained/edisco_tsp20.ckpt'
checkpoint = torch.load(ckpt_path, map_location='cpu')

print("Top-level keys:")
for k in checkpoint.keys():
    print(f"  {k}")

print("\nState dict keys (first 20):")
state_dict = checkpoint.get('state_dict', {})
for i, k in enumerate(state_dict.keys()):
    print(f"  {k}: {state_dict[k].shape}")
    if i >= 20:
        print(f"  ... and {len(state_dict) - 20} more")
        break

print("\nHyper parameters:")
hparams = checkpoint.get('hyper_parameters', {})
for k, v in hparams.items():
    print(f"  {k}: {v}")
