"""Check checkpoint structure to understand key names."""
import torch

ckpt_path = 'pretrained/edisco_tsp20.ckpt'
checkpoint = torch.load(ckpt_path, map_location='cpu')

print("Top-level keys:")
for k in checkpoint.keys():
    print(f"  {k}")

print("\nState dict keys - ALL unique prefixes:")
state_dict = checkpoint.get('state_dict', {})
prefixes = set()
for k in state_dict.keys():
    prefix = k.split('.')[0]
    prefixes.add(prefix)
print(f"  Prefixes: {prefixes}")

print("\nState dict keys with 'model.' prefix (first 10):")
for i, k in enumerate(state_dict.keys()):
    if k.startswith('model.'):
        print(f"  {k}: {state_dict[k].shape}")
        if i >= 10:
            break

print("\nState dict keys with 'score_network.' prefix (first 10):")
count = 0
for k in state_dict.keys():
    if k.startswith('score_network.'):
        print(f"  {k}: {state_dict[k].shape}")
        count += 1
        if count >= 10:
            break

print(f"\nTotal keys: {len(state_dict)}")
print(f"Keys starting with 'model.': {sum(1 for k in state_dict if k.startswith('model.'))}")
print(f"Keys starting with 'score_network.': {sum(1 for k in state_dict if k.startswith('score_network.'))}")
