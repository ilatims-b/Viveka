
import argparse
import glob
import os
import torch as t
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Standard utilities
from utils import (
    create_prompts,
    generate_model_answers,
    check_correctness,
    load_model,
    load_statements
)
# Import both pipelines
from hook import get_resid_acts
from probe_module import probe_truth_representation
# Probing network
from probing_model import ProbingNetwork, hparams


def load_activation_dataset(activations_dir):
    """
    Load all .pt files from activations_dir into a single TensorDataset.
    Assumes each file is a dict with 'activations' and 'labels'.
    """
    activations_list, labels_list = [], []
    for fname in glob.glob(os.path.join(activations_dir, 'layer_*_stmt_*.pt')):
        data = t.load(fname)
        activations_list.append(data['activations'])  # [batch, dim]
        labels_list.append(data['labels'])             # [batch]
    X = t.cat(activations_list, dim=0)
    y = t.cat(labels_list, dim=0).float().unsqueeze(1)
    return TensorDataset(X, y)


def train_probing_network(dataset_dir, device):
    """
    Instantiate and train a ProbingNetwork on activations stored in dataset_dir.
    """
    # Load dataset
    activations_dir = os.path.join(dataset_dir, 'activations', hparams.model_name.replace('/', '_'))
    dataset = load_activation_dataset(activations_dir)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = t.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=hparams.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=hparams.batch_size)

    # Initialize model, optimizer, criterion
    model = ProbingNetwork(hparams.model_name).to(device)
    optimizer = t.optim.Adam(model.parameters(), lr=hparams.lr)
    criterion = t.nn.BCELoss()
    scheduler = t.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.0, total_iters=hparams.warmup_steps)

    # Training loop
    for epoch in range(hparams.num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Optional: add validation here
    print(f"Probing network training complete. Model saved to 'probe_model.pt'.")
    t.save(model.state_dict(), 'probe_model.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract activations and optionally probe truth representation"
    )
    # Model & device
    parser.add_argument('--model_repo_id', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda' if t.cuda.is_available() else 'cpu')

    # Layers
    parser.add_argument('--layers', nargs='+', type=int, default=[-1])
    # Datasets
    parser.add_argument('--datasets', nargs='+', required=True)
    # Standard pipeline flags
    parser.add_argument('--enable_llm_extraction', action='store_true')
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--num_generations', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=25)
    # Probe pipeline flags
    parser.add_argument('--probe_truth', action='store_true')
    parser.add_argument('--probe_output_dir', type=str, default='probes_data')
    # Train probing network after extraction
    parser.add_argument('--train_probe', action='store_true',
                        help='Train probing network on saved activations')
    args = parser.parse_args()

    # Load model, tokenizer, and layers
    tokenizer, model, layer_modules = load_model(args.model_repo_id, args.device)
    if args.layers == [-1]:
        layer_indices = list(range(len(layer_modules)))
    else:
        layer_indices = args.layers

    # Resolve datasets list
    datasets = args.datasets
    if datasets == ['all']:
        datasets = [os.path.relpath(fp, 'datasets').replace('.csv', '')
                    for fp in glob.glob('datasets/**/*.csv', recursive=True)]

    for dataset in datasets:
        print(f"\n=== Dataset: {dataset} ===")
        df, statements, correct_answers = load_statements(dataset)

        # Probe truth branch
        if args.probe_truth:
            probe_truth_representation(
                statements,
                correct_answers,
                tokenizer,
                model,
                layer_modules,
                layer_indices,
                args.device,
                num_generations=args.num_generations,
                output_dir=args.probe_output_dir
            )
            # After extracting, optionally train the probing net
            if args.train_probe:
                train_probing_network(args.probe_output_dir, args.device)
            continue

        # Original pipeline
        safe_name = args.model_repo_id.replace('/', '__')
        save_base = os.path.join('acts_output', safe_name, dataset)
        os.makedirs(save_base, exist_ok=True)

        all_corr, all_ans, all_exact = [], [], []
        batch_ctr = 0
        for start in tqdm(range(0, len(statements), args.batch_size),
                          desc=f"Processing {dataset}"):
            batch_stmts = statements[start:start + args.batch_size]
            batch_corr = correct_answers[start:start + args.batch_size]
            acts, corr_list, ans_list, exact_list = get_resid_acts(
                batch_stmts,
                batch_corr,
                tokenizer,
                model,
                layer_modules,
                layer_indices,
                args.device,
                num_generations=args.num_generations,
                enable_llm_extraction=args.enable_llm_extraction
            )
            all_corr.extend(corr_list)
            all_ans.extend(ans_list)
            all_exact.extend(exact_list)
            for li, tensor in acts.items():
                t.save(tensor, os.path.join(save_base, f"layer_{li}_{start}.pt"))
            batch_ctr += 1
            if args.early_stop and batch_ctr >= 2:
                break

        if not args.early_stop and len(all_ans) == len(df):
            df['model_answers'] = all_ans
            df['automatic_correctness'] = all_corr
            df['exact_answers'] = all_exact
            df.to_csv(os.path.join(save_base, f"{dataset}_with_results.csv"), index=False)
            print(f"Saved results to CSV for {dataset}.")

