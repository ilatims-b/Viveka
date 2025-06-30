# File: project_svd.py
import os
import numpy as np
import torch


def load_correction_vectors(layer_id, train_path):
    data = np.load(train_path)
    h_c = data['correct'][:, layer_id, :]  # [N, D]
    h_i = data['incorrect'][:, layer_id, :]  # [N, D]
    d_q = h_c - h_i  # [N, D]
    return torch.tensor(d_q, dtype=torch.float32)


def compute_svd_basis(d_q, top_k=20):
    # d_q: [N, D] → U: [N, N], S: [N], Vh: [D, D]
    _, _, Vh = torch.linalg.svd(d_q, full_matrices=False)
    basis = Vh[:top_k]  # [k, D]
    return basis  # Each row is a basis vector


def save_basis(basis, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(basis, save_path)
    print(f"✓ Saved top-{basis.size(0)} truth directions to {save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--save_path", type=str, default="svd_basis/layer_{layer}.pt")
    args = parser.parse_args()

    d_q = load_correction_vectors(args.layer, args.train_path)
    basis = compute_svd_basis(d_q, top_k=args.top_k)

    save_path = args.save_path.replace("{layer}", str(args.layer))
    save_basis(basis, save_path)
