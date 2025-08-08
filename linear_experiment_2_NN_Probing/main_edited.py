from utils import (load_model, load_statements)
import argparse
import glob
import os
import torch as t
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Import the DECOUPLED pipeline functions from the modified hook.py
from hook import generate_and_label_answers, get_truth_probe_activations
# Import the training and data loading functions
from classifier import ProbingNetwork, hparams

def load_activation_dataset(activations_dir):
    """
    Load all .pt files from an activations directory into a single TensorDataset.
    """
    activations_list, labels_list = [], []
    # Added tqdm to the file loading process
    for fname in tqdm(glob.glob(os.path.join(activations_dir, 'layer_*_stmt_*.pt')), desc="Loading activation files"):
        data = t.load(fname)
        activations_list.append(data['activations'])
        labels_list.append(data['labels'])
    
    if not activations_list:
        return None

    X = t.cat(activations_list, dim=0)
    y = t.cat(labels_list, dim=0).float().unsqueeze(1)
    return TensorDataset(X, y)


def train_probing_network(dataset_dir, device):
    """
    Instantiate and train a ProbingNetwork on the saved activations.
    """
    model_name_safe = hparams.model_name.replace('/', '_')
    activations_dir = os.path.join(dataset_dir, 'activations', model_name_safe)
    
    print(f"Loading dataset from: {activations_dir}")
    dataset = load_activation_dataset(activations_dir)
    
    if dataset is None or len(dataset) == 0:
        print("No activation data found. Skipping training.")
        return

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = t.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=hparams.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=hparams.batch_size)

    model = ProbingNetwork(hparams.model_name).to(device)
    optimizer = t.optim.Adam(model.parameters(), lr=hparams.lr)
    criterion = t.nn.BCELoss()
    scheduler = t.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=hparams.warmup_steps)

    print("Starting training of the probing network...")
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
        
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with t.no_grad():
            # MODIFICATION: Added tqdm to the validation loop
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{hparams.num_epochs} [Validation]", leave=False)
            for X_batch, y_batch in val_pbar:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                val_loss += criterion(outputs, y_batch).item()
                preds = (outputs > 0.5).float()
                total += y_batch.size(0)
                correct += (preds == y_batch).sum().item()
        
        print(f"Epoch {epoch+1} | Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {correct/total:.4f}")

    print(f"\nProbing network training complete. Model saved to 'probe_model.pt'.")
    t.save(model.state_dict(), 'probe_model.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run a multi-stage pipeline to generate data and train a truth probe."
    )
    # --- Core Arguments ---
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to the dataset CSV file.")
    parser.add_argument('--model_repo_id', type=str, required=True, help="Hugging Face model repository ID.")
    parser.add_argument('--device', type=str, default='cuda' if t.cuda.is_available() else 'cpu')
    
    # --- Pipeline Stage Control ---
    parser.add_argument('--stage', type=str, choices=['generate', 'activate', 'train', 'all'], default='all', 
                        help="Which stage of the probing pipeline to run.")

    # --- Arguments for Parallelization ---
    parser.add_argument('--start-index', type=int, default=0, help="The starting row index of the dataset to process.")
    parser.add_argument('--end-index', type=int, default=None, help="The ending row index of the dataset to process. Processes to the end if not specified.")

    # --- Configuration Arguments ---
    parser.add_argument('--layers', nargs='+', type=int, default=[-1], help="List of layer indices to probe. -1 for all layers.")
    parser.add_argument('--probe_output_dir', type=str, default='probes_data', help="Directory to save generated data and activations.")
    parser.add_argument('--num_generations', type=int, default=30, help="Number of answers to generate per statement for probing.")
    
    args = parser.parse_args()
    hparams.model_name = args.model_repo_id

    # --- Model Loading ---
    print(f"Loading model: {args.model_repo_id}...")
    tokenizer, model, layer_modules = load_model(args.model_repo_id, args.device)
    
    if -1 in args.layers:
        layer_indices = list(range(len(layer_modules)))
    else:
        layer_indices = args.layers

    # --- Dataset Loading and Slicing ---
    print(f"Loading dataset from: {args.dataset_path}")
    df, all_statements, all_correct_answers = load_statements(args.dataset_path)

    start = args.start_index
    end = args.end_index if args.end_index is not None else len(all_statements)
    
    if start >= len(all_statements):
        print(f"Start index {start} is out of bounds for dataset of length {len(all_statements)}. Exiting.")
        exit()
    
    end = min(end, len(all_statements))

    print(f"Processing slice of dataset from index {start} to {end}.")
    statements_to_process = all_statements[start:end]
    answers_to_process = all_correct_answers[start:end]

    # --- Main Pipeline Execution ---
    # STAGE 1: Generate and Label Answers
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
    
    # STAGE 2: Extract Activations
    if args.stage in ['activate', 'all']:
        get_truth_probe_activations(
            statements=statements_to_process,
            tokenizer=tokenizer,
            model=model,
            layers=layer_modules,
            layer_indices=layer_indices,
            device=args.device,
            output_dir=args.probe_output_dir,
            start_index=start
        )

    # STAGE 3: Train Probing Network
    if args.stage == 'train':
        print("\nNote: Training should be run after all activation data has been generated.")
        train_probing_network(args.probe_output_dir, args.device)
