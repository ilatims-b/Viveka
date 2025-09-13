
# Example usage of advanced toy transformer with comprehensive metrics tracking

from toy_model import *
from metrics import AdvancedMetricsTracker
import wandb
import torch
import numpy as np

# Initialize wandb for comprehensive logging
wandb.login()

print("=== Advanced Transformer Metrics Example ===")
print()

# Define transition matrices for Markov process
T0 = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 0.5]
])

T1 = np.array([
    [0, 0, 0],
    [0, 0, 0],
    [0.5, 0, 0]
])

# Create dataset
print("Creating Markov dataset...")
dataset = MarkovData(10000, 30, 3, 2, [T0, T1])

print("\n=== Training with ALL Advanced Metrics ===")

# Train model with ALL metrics enabled
model = train_model(
    dataset=dataset,

    # Architecture - make it more complex for interesting metrics
    n_layers=2,          # Need multiple layers for composition scores
    d_model=32,
    n_heads=2,           # Multiple heads for attention analysis
    d_head=16,
    attn_only=False,     # Include MLPs

    # Training
    n_epochs=20,
    batch_size=64,
    lr=0.05,

    # Logging
    wandb=True,
    wandb_project_name="advanced_transformer_analysis",
    save_dir="advanced_model_checkpoints",

    # ALL ADVANCED METRICS ENABLED
    track_ngrams=True,
    ngram_orders=[1, 2, 3, 4, 5],    # Track many n-gram orders

    track_composition=True,           # Attention head composition scores
    track_previous_token=True,        # Previous token matching
    track_in_context=True,           # In-context learning capability
    track_prefix_matching=True       # Prefix matching behavior
)

print("\n=== Manual Advanced Metrics Testing ===")

# Test individual metrics functions
print("Testing composition scores...")
from metrics_tracker_advanced import compute_composition_scores
comp_scores = compute_composition_scores(model, layer1_idx=0, layer2_idx=1)
print(f"Composition scores: {len(comp_scores)} head pairs analyzed")
for k, v in list(comp_scores.items())[:5]:  # Show first 5
    print(f"  {k}: {v:.4f}")

print("\nTesting previous token matching...")
from metrics_tracker_advanced import compute_previous_token_matching_score
prev_scores = compute_previous_token_matching_score(model, num_samples=100)
print(f"Previous token scores for {len(prev_scores)} attention heads:")
for k, v in list(prev_scores.items())[:5]:  # Show first 5
    print(f"  {k}: {v:.4f}")

print("\nTesting in-context learning...")
from metrics_tracker_advanced import compute_in_context_learning_score
icl_score = compute_in_context_learning_score(model, num_samples=50, k1=5, k2=25)
print(f"In-context learning score: {icl_score:.4f}")
print("  (More negative = better in-context learning)")

print("\nTesting prefix matching...")
from metrics_tracker_advanced import compute_prefix_matching_score
prefix_scores = compute_prefix_matching_score(model, num_samples=100)
print(f"Prefix matching scores for {len(prefix_scores)} attention heads:")
for k, v in list(prefix_scores.items())[:5]:  # Show first 5
    print(f"  {k}: {v:.4f}")

print("\n=== Advanced Metrics Tracker Demo ===")

# Create advanced metrics tracker manually
tracker = AdvancedMetricsTracker(
    ngram_orders=[2, 3, 4],
    track_sets=["train", "val", "complete"],
    track_composition=True,
    track_previous_token=True,
    track_in_context=True,
    track_prefix_matching=True
)

# Prepare test datasets
print("Preparing test datasets for metrics...")
train_data = torch.stack(dataset.data[:200])  # 200 train sequences
val_data = torch.stack(dataset.data[200:300])  # 100 val sequences  
complete_data = torch.cat([train_data, val_data], dim=0)

datasets_dict = {
    "train": train_data,
    "val": val_data,
    "complete": complete_data
}

print("Computing all metrics...")
all_metrics = tracker.compute_all_metrics(model, datasets_dict, step=1000)

print(f"\nComputed {len(all_metrics)} total metrics:")
print("\nN-gram metrics:")
for k, v in all_metrics.items():
    if 'gram' in k:
        print(f"  {k}: {v:.4f}")

print("\nAttention composition metrics:")
for k, v in all_metrics.items():
    if 'comp' in k:
        print(f"  {k}: {v:.4f}")

print("\nBehavioral metrics:")
for k, v in all_metrics.items():
    if k in ['avg_prev_token_matching', 'in_context_learning', 'avg_prefix_matching']:
        print(f"  {k}: {v:.4f}")

print("\n=== Fine-tuning with Metrics ===")

# Finetune with metrics tracking
model_ft = finetune_model(
    model, 
    dataset, 
    n_epochs=10,
    wandb=True,
    wandb_project_name="advanced_transformer_finetuning",

    # Enable all metrics for fine-tuning analysis
    track_ngrams=True,
    ngram_orders=[2, 3],
    track_composition=True,
    track_previous_token=True,
    track_in_context=True,
    track_prefix_matching=True
)

print("\n=== Metrics Interpretation Guide ===")
print("""
1. N-GRAM METRICS:
   - Lower KL divergence = better sequence modeling
   - Track different orders to see what patterns model learns

2. COMPOSITION SCORES:
   - Q/K/V composition between layer heads
   - Higher scores = more composition between heads
   - Reveals how information flows through layers

3. PREVIOUS TOKEN MATCHING:
   - How much heads attend to previous token
   - Important for next-token prediction tasks
   - Values closer to 1 = strong previous token focus

4. IN-CONTEXT LEARNING:
   - Measures performance improvement within sequence
   - More negative = better in-context learning
   - Shows model's ability to adapt during inference

5. PREFIX MATCHING:
   - How well heads track repeated tokens
   - Important for copy/memory tasks
   - Higher values = better token tracking
""")

print("\n=== Training Complete ===")
print("Check your wandb dashboard for comprehensive metrics visualization!")
print("All metrics are logged every 100 training steps.")

# Finish wandb run
wandb.finish()
