
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
# Import pipelines
from hook import get_resid_acts
from probe_module import probe_truth_representation
# Probing network and SVD utility
from probing_model import ProbingNetwork, hparams
from svd_utils import load_and_reduce_activations


def build_activation_dataset(activations_dir, reduce_dim=None, device='cpu'):
    """
    Load activations across all layers into a TensorDataset, optionally reducing each batch via SVD.
    """
    # If reduce_dim is specified, use svd_utils to pre-reduce per-batch
    X_parts, y_parts = [], []
    if reduce_dim is not None:
        # group files by start index
        start_indices = set()
        for fname in glob.glob(os.path.join(activations_dir, 'layer_*_stmt_*.pt')):
            # filename format: layer_{layer}_{start}.pt
            parts = fname.split('_')
            start_idx = parts[-1].split('.')[0]
            start_indices.add(int(start_idx))
        # reduce per batch
        for start in sorted(start_indices):
            reduced = load_and_reduce_activations(activations_dir, start, reduce_dim, device)
            # reduced: Dict[layer_idx, Tensor(batch, reduce_dim)]
            # stack across layers
            for acts in reduced.values():
                X_parts.append(acts)
            # load corresponding labels from any one file
            sample = t.load(os.path.join(activations_dir, f"layer_{list(reduced.keys())[0]}_{start}.pt"))
            y_parts.append(sample['labels'].unsqueeze(1).float().to(device))
    else:
        for fname in glob.glob(os.path.join(activations_dir, 'layer_*_stmt_*.pt')):
            data = t.load(fname)
            X_parts.append(data['activations'].to(device))
            y_parts.append(data['labels'].unsqueeze(1).float().to(device))
    # concatenate all
    X = t.cat(X_parts, dim=0)
    y = t.cat(y_parts, dim=0)
    return TensorDataset(X, y)


def train_probing_network(dataset_dir, device, reduce_dim=None):
    """
    Train ProbingNetwork on activations stored in dataset_dir.
    """
    activations_dir = os.path.join(
        dataset_dir,
        'activations',
        hparams.model_name.replace('/', '_')
    )
    dataset = build_activation_dataset(activations_dir, reduce_dim, device)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = t.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=hparams.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=hparams.batch_size)

    model = ProbingNetwork(hparams.model_name).to(device)
    optimizer = t.optim.Adam(model.parameters(), lr=hparams.lr)
    criterion = t.nn.BCELoss()
    scheduler = t.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.0,
        total_iters=hparams.warmup_steps
    )

    for epoch in range(hparams.num_epochs):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f}")

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with t.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                preds = (outputs > 0.5).float()
                correct += (preds == y_batch).sum().item()
                total += y_batch.numel()
        print(f"Epoch {epoch+1}: Val Loss={val_loss/len(val_loader):.4f}, Acc={correct/total:.4f}")

    out_path = os.path.join(dataset_dir, 'probe_model.pt')
    t.save(model.state_dict(), out_path)
    print(f"Probing model saved to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run extraction, probing, and optional network training"
    )
    parser.add_argument('--model_repo_id', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda' if t.cuda.is_available() else 'cpu')
    parser.add_argument('--layers', nargs='+', type=int, default=[-1])
    parser.add_argument('--datasets', nargs='+', required=True)

    # Standard pipeline
    parser.add_argument('--enable_llm_extraction', action='store_true')
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--num_generations', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=25)

    # Probe pipeline
    parser.add_argument('--probe_truth', action='store_true')
    parser.add_argument('--probe_output_dir', type=str, default='probes_data')
    parser.add_argument('--train_probe', action='store_true')
    parser.add_argument('--reduce_dim', type=int,
                        help='Optional SVD target dimension (e.g., 576)')

    args = parser.parse_args()

    tokenizer, model, layer_modules = load_model(args.model_repo_id, args.device)
    layer_indices = list(range(len(layer_modules))) if args.layers == [-1] else args.layers

    datasets = args.datasets
    if datasets == ['all']:
        datasets = [os.path.relpath(fp, 'datasets').replace('.csv','')
                    for fp in glob.glob('datasets/**/*.csv', recursive=True)]

    for dataset in datasets:
        print(f"\n=== Dataset: {dataset} ===")
        df, statements, correct_answers = load_statements(dataset)

        if args.probe_truth:
            probe_truth_representation(
                statements, correct_answers,
                tokenizer, model, layer_modules, layer_indices,
                args.device, num_generations=args.num_generations,
                output_dir=args.probe_output_dir
            )
            if args.train_probe:
                train_probing_network(
                    args.probe_output_dir,
                    args.device,
                    reduce_dim=args.reduce_dim
                )
        else:
            # Standard activation & answer extraction pipeline
            safe_name = args.model_repo_id.replace('/', '__')
            save_base = os.path.join('acts_output', safe_name, dataset)
            os.makedirs(save_base, exist_ok=True)

            all_corr, all_ans, all_exact = [], [], []
            batch_ctr = 0
            for start in tqdm(
                range(0, len(statements), args.batch_size),
                desc=f"Processing {dataset}"
            ):
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
                out_csv = os.path.join(save_base, f"{dataset}_with_results.csv")
                df.to_csv(out_csv, index=False)
                print(f"Saved results to {out_csv}")

