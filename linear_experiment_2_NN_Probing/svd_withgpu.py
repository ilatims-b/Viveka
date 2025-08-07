import torch as t
import os
from typing import Dict

def load_and_reduce_activations(save_base: str, start: int, target_dim: int = 512, device: str = "cuda") -> Dict[int, t.Tensor]:

    reduced_acts = {}

    for filename in os.listdir(save_base):
        if filename.endswith(f"_{start}.pt") and filename.startswith("layer_"):
            # Extract layer index from filename
            parts = filename.split("_")
            if len(parts) < 3:
                continue  # skip malformed filenames
            layer_idx = int(parts[1])
            path = os.path.join(save_base, filename)

            # Load activation tensor and move to GPU
            acts = t.load(path).to(device)

            if acts.shape[1] < target_dim:
                raise ValueError(f"Cannot reduce from {acts.shape[1]} to {target_dim}")

            # Mean-center before SVD
            mean = acts.mean(dim=0, keepdim=True)
            centered = acts - mean

            # Perform SVD on GPU
            U, S, Vh = t.linalg.svd(centered, full_matrices=False)
            reduced = centered @ Vh[:target_dim].T  # Shape: (30, target_dim)

            # Optionally move reduced result to CPU
            reduced_acts[layer_idx] = reduced.float().cpu()

    return reduced_acts
