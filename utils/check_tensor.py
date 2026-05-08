# @title
import torch 

def check_tensor(name, tensor):
    if tensor is None:
        print(f"{name}: None")
        return

    print(f"\n{name}")
    print(f"shape: {tensor.shape}")
    #print(f"dtype: {tensor.dtype}")
    #print(f"device: {tensor.device}")
    #print(f"min: {tensor.min().item():.6f}")
    #print(f"max: {tensor.max().item():.6f}")
    #print(f"mean: {tensor.mean().item():.6f}")

    if torch.isnan(tensor).any():
        print(f"WARNING: NaN detected in {name}")

    if torch.isinf(tensor).any():
        print(f"WARNING: Inf detected in {name}")