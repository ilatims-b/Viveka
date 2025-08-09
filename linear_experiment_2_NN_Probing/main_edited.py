from utils import load_model, load_statements
from hook import generate_and_label_answers, get_truth_probe_activations
from classifier import ProbingNetwork, hparams
from svd_withgpu import perform_global_svd

import argparse
import glob
import os
import torch as t
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# === SVD Projection Loader ========================================

def load_and_project_activations(activations_dir, layer_idx, device):
    """
    Loads raw activation files and the SVD projection matrix for a specific layer,
    then returns the projected data and labels.
    """
    svd_dir = os.path.join(os.path.dirname(activations_dir), 'svd_components')
    projection_matrix_path = os.path.join(svd_dir, f"projection_matrix_layer_{layer_idx}.pt")

    if not os.path.exists(projection_matrix_path):
        raise FileNotFoundError(
            f"SVD projection matrix not found for layer {layer_idx}. Please run the 'svd' stage first."
        )

    projection_matrix = t.load(projection_matrix_path).to(device)

    activations_list, labels_list = [], []
    file_pattern = os.path.join(activations_dir, f'layer_{layer_idx}_stmt_*.pt')

    for fname in tqdm(glob.glob(file_pattern), desc=f"Loading & Projecting L{layer_idx}", leave=False):
        data = t.load(fname)
        raw_activations = data['activations'].to(device)
        projected_activations = (projection_matrix @ raw_activations.T).T
        activations_list.append(projected_activations)
        labels_list.append(data['labels'])

    if not activations_list:
        return None

    return TensorDataset(
        t.cat(activations_list, dim=0),
        t.cat(labels_list, dim=0).float().unsqueeze(1)
    )

# === Training Function ============================================

def load_preprojected_dataset(projected_dir, layer_idx):
    """
    Loads already SVD-processed activation files for a specific layer
    from the activations_svd directory.
    """
    activations_list, labels_list = [], []
    file_pattern = os.path.join(projected_dir, f'layer{layer_idx}_stmt*_svd_processed.pt')

    for fname in tqdm(glob.glob(file_pattern), desc=f"Loading pre-projected L{layer_idx}", leave=False):
        data = t.load(fname)
        activations_list.append(data['activations'])
        labels_list.append(data['labels'])

    if not activations_list:
        return None

    return TensorDataset(
        t.cat(activations_list, dim=0),
        t.cat(labels_list, dim=0).float().unsqueeze(1)
    )


def train_probing_network(dataset_dir, train_layers, device):
    """
    Trains a probing network for each specified layer on either:
    - precomputed SVD-projected activations from activations_svd, or
    - raw activations projected on-the-fly using saved projection matrices.
    """
    model_name_safe = hparams.model_name.replace('/', '_')
    activations_dir = os.path.join(dataset_dir, 'activations', model_name_safe)
    projected_dir = os.path.join(dataset_dir, 'activations_svd', model_name_safe)
    probes_dir = os.path.join(dataset_dir, 'trained_probes', model_name_safe)
    os.makedirs(probes_dir, exist_ok=True)

    for l_idx in tqdm(train_layers, desc="Training probe per layer"):
        # Prefer precomputed projected activations if available
        if os.path.exists(projected_dir) and glob.glob(os.path.join(projected_dir, f'layer{l_idx}_stmt*_svd_processed.pt')):
            dataset = load_preprojected_dataset(projected_dir, l_idx)
        else:
            dataset = load_and_project_activations(activations_dir, l_idx, device)

        if dataset is None:
            print(f"No data for layer {l_idx}. Skipping.")
            continue

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = t.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=hparams.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=hparams.batch_size)

        model = ProbingNetwork(hparams.model_name).to(device)
        optimizer = t.optim.Adam(model.parameters(), lr=hparams.lr)
        criterion = t.nn.BCELoss()
        scheduler = t.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=hparams.warmup_steps)

        print(f"\n--- Training probe for Layer {l_idx} ---")
        for epoch in range(hparams.num_epochs):
            model.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{hparams.num_epochs} [Training]", leave=False)
            for X_batch, y_batch in pbar:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                if scheduler.get_last_lr()[0] < hparams.lr:
                    scheduler.step()
                pbar.set_postfix({'loss': loss.item()})

            # Validation
            model.eval()
            val_loss, correct, total = 0, 0, 0
            with t.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{hparams.num_epochs} [Validation]", leave=False)
                for X_batch, y_batch in val_pbar:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    val_loss += criterion(outputs, y_batch).item()
                    preds = (outputs > 0.5).float()
                    total += y_batch.size(0)
                    correct += (preds == y_batch).sum().item()

            print(f"Layer {l_idx} | Epoch {epoch+1} | Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {correct/total:.4f}")

        t.save(model.state_dict(), os.path.join(probes_dir, f'probe_model_layer_{l_idx}.pt'))

# === Main Execution Block =========================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a multi-stage pipeline to generate data, extract activations, run SVD, and train a truth probe.")
    # --- Core Arguments ---
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to the dataset CSV file.")
    parser.add_argument('--model_repo_id', type=str, required=True, help="Hugging Face model repository ID.")
    parser.add_argument('--device', type=str, default='cuda' if t.cuda.is_available() else 'cpu')

    # --- Pipeline Stage Control ---
    parser.add_argument('--stage', type=str, choices=['generate', 'activate', 'svd', 'train', 'all'], default='all',
                        help="Which stage of the probing pipeline to run.")

    # --- Arguments for Parallelization ---
    parser.add_argument('--start-index', type=int, default=0, help="The starting row index of the dataset to process.")
    parser.add_argument('--end-index', type=int, default=None, help="The ending row index of the dataset to process. Processes to the end if not specified.")

    # --- Configuration Arguments ---
    parser.add_argument('--layers', nargs='+', type=int, default=[-1], help="List of layer indices to probe. -1 for all layers.")
    parser.add_argument('--probe_output_dir', type=str, default='probes_data', help="Directory to save generated data and activations.")
    parser.add_argument('--num_generations', type=int, default=30, help="Number of answers to generate per statement for probing.")

    # --- SVD & Training ---
    parser.add_argument('--svd_layers', nargs='+', type=int, help="Layers for 'svd' stage.")
    parser.add_argument('--train_layers', nargs='+', type=int, help="Layers for 'train' stage.")
    parser.add_argument('--svd_dim', type=int, default=576)

    args = parser.parse_args()
    hparams.model_name = args.model_repo_id

    # --- Model Loading ---
    print(f"Loading model: {args.model_repo_id}...")
    tokenizer, model, layer_modules = load_model(args.model_repo_id, args.device)
    num_layers = len(layer_modules)

    # --- Stage Routing ---
    if args.stage in ['generate', 'activate', 'all']:
        if -1 in args.layers:
            args.layers = list(range(num_layers))

        print(f"Loading dataset from: {args.dataset_path}")
        df, all_statements, all_correct_answers = load_statements(args.dataset_path)

        start = args.start_index
        end = args.end_index if args.end_index is not None else len(all_statements)
        statements_to_process = all_statements[start:end]
        answers_to_process = all_correct_answers[start:end]

        if args.stage in ['generate', 'all']:
            generate_and_label_answers(
                statements=statements_to_process,
                correct_answers=answers_to_process,
                tokenizer=tokenizer,
                model=model,
                device=args.device,
                num_generations=args.num_generations,
                output_dir=args.probe_output_dir
            )

        if args.stage in ['activate', 'all']:
            get_truth_probe_activations(
                statements=statements_to_process,
                tokenizer=tokenizer,
                model=model,
                layers=layer_modules,
                layer_indices=args.layers,
                device=args.device,
                output_dir=args.probe_output_dir,
                start_index=start
            )

    if args.stage in ['svd', 'all']:
        if not args.svd_layers:
            parser.error("--svd_layers is required for 'svd' stage.")
        activations_dir = os.path.join(args.probe_output_dir, 'activations', args.model_repo_id.replace('/', '_'))
        perform_global_svd(activations_dir, args.svd_dim, args.svd_layers, args.device)

    if args.stage in ['train', 'all']:
        if not args.train_layers:
            parser.error("--train_layers is required for 'train' stage.")
        train_probing_network(args.probe_output_dir, args.train_layers, args.device)
