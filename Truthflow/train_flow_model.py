import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def load_data(layer_id, split="train", data_dir="data"):
    path = os.path.join(data_dir, f"truthflow_{split}_hiddenstates.npz")
    data = np.load(path)

    # [N, L, D] → take layer ℓ across N samples
    h_q_all = data["query"]      # query last token [N, L, D]
    h_c_all = data["correct"]    # correct mean [N, L, D]
    h_i_all = data["incorrect"]  # incorrect mean [N, L, D]

    h_q = torch.tensor(h_q_all[:, layer_id, :], dtype=torch.float32)
    h_c = torch.tensor(h_c_all[:, layer_id, :], dtype=torch.float32)
    h_i = torch.tensor(h_i_all[:, layer_id, :], dtype=torch.float32)

    d_q = h_c - h_i
    return h_q, d_q

class FixedFlowModel(nn.Module):
    """Fixed version of FlowModel that handles dimension mismatches"""
    def __init__(self, hidden_dim):
        super().__init__()
        # Use a reasonable time dimension that scales with hidden_dim
        time_dim = min(128, max(32, hidden_dim // 16))
        self.time_dim = time_dim
        self.net = self.create_fixed_unet(hidden_dim, time_dim)

    def create_fixed_unet(self, hidden_dim, time_dim):
        """Create a UNet that properly handles the dimensions"""
        class SinusoidalTimeEmbedding(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim

            def forward(self, t):
                device = t.device
                half_dim = self.dim // 2
                emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
                emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
                if t.dim() == 2 and t.size(1) == 1:
                    t = t.squeeze(1)
                emb = t[:, None] * emb[None, :]
                return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        class FixedUNet1D(nn.Module):
            def __init__(self, hidden_dim, time_dim):
                super().__init__()
                self.time_embed = SinusoidalTimeEmbedding(time_dim)

                # Simple architecture that avoids dimension issues
                self.input_proj = nn.Linear(hidden_dim + time_dim, hidden_dim)

                # Encoder
                self.encoder = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.LayerNorm(hidden_dim // 2),
                    nn.ReLU(),
                )

                # Bottleneck
                self.bottleneck = nn.Sequential(
                    nn.Linear(hidden_dim // 2, hidden_dim // 4),
                    nn.LayerNorm(hidden_dim // 4),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 4, hidden_dim // 2),
                    nn.LayerNorm(hidden_dim // 2),
                    nn.ReLU(),
                )

                # Decoder
                self.decoder = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                )

                self.output_proj = nn.Linear(hidden_dim, hidden_dim)

            def forward(self, t, x):
                # Time embedding
                t_embed = self.time_embed(t)

                # Concatenate input with time embedding
                x_input = torch.cat([x, t_embed], dim=-1)
                x_proj = self.input_proj(x_input)

                # Encoder
                x_enc = self.encoder(x_proj)

                # Bottleneck
                x_bottle = self.bottleneck(x_enc)

                # Decoder with skip connection
                x_skip = torch.cat([x_bottle, x_enc], dim=-1)
                x_dec = self.decoder(x_skip)

                # Output
                output = self.output_proj(x_dec)

                return output

        return FixedUNet1D(hidden_dim, time_dim)

    def forward(self, t, z):
        return self.net(t, z)

def train_flow_model(h_q, d_q, hidden_dim, epochs=25, batch_size=128, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Hidden dimension: {hidden_dim}")

    # Use the fixed model instead
    model = FixedFlowModel(hidden_dim).to(device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    dataset = TensorDataset(h_q, d_q)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        for hq, dq in loader:
            hq, dq = hq.to(device), dq.to(device)

            # Sample random time steps
            t = torch.rand(hq.size(0), 1, device=device)  # [B, 1]

            # Linear interpolation between h_q and h_c
            zt = t * dq + (1 - t) * hq

            # Target is the direction from h_q to h_c
            target = dq - hq

            # Forward pass
            pred = model(t, zt)
            loss = criterion(pred, target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = total_loss / num_batches
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f} - LR: {current_lr:.2e}")

    return model

def main(layer, data_dir="data", epochs=25, save_path="models"):
    print(f"Training flow model for layer {layer}")

    # Load data
    h_q, d_q = load_data(layer, split="train", data_dir=data_dir)
    print(f"h_q shape: {h_q.shape}")
    print(f"d_q shape: {d_q.shape}")

    _, hidden_dim = h_q.shape
    print(f"Hidden dimension: {hidden_dim}")

    # Train model
    model = train_flow_model(h_q, d_q, hidden_dim, epochs=epochs)

    # Save model
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, f"flow_model_layer{layer}.pt")
    torch.save(model.state_dict(), save_file)
    print(f"✓ Saved model for layer {layer} to {save_file}")

    return model


'''
if __name__ == "__main__":
    model = main(
        layer=20,
        data_dir="/kaggle/input/truthflow-hidden-states",
        epochs=30,
        save_path="/kaggle/working/flow_models"
    )


'''
