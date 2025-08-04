# probing_model.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    f1_score, 
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


class HParams:
    input_dim = 576
    hidden1 = 144
    hidden2 = 12
    output_dim = 1
    batch_size = 32
    lr = 1e-4
    num_epochs = 3
    warmup_steps = 100
    model_name = 'gemma-2-2b-it'


hparams = HParams()


class ProbingNetwork(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.net = nn.Sequential(
            nn.Linear(hparams.input_dim, hparams.hidden1),
            nn.ReLU(),
            nn.Linear(hparams.hidden1, hparams.hidden2),
            nn.ReLU(),
            nn.Linear(hparams.hidden2, hparams.output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        print(f'Using model: {self.model_name}')
        return self.net(x)


model = ProbingNetwork(hparams.model_name).to(device)

# Sample Data 
X = torch.randn(1000, hparams.input_dim).float()
y = torch.randint(0, 2, (1000,)).float().unsqueeze(1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=hparams.batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=hparams.batch_size)


hparams.total_steps = len(train_loader) * hparams.num_epochs


optimizer = optim.Adam(model.parameters(), lr=hparams.lr)
criterion = nn.BCELoss()


def lr_lambda(current_step):
    if current_step < hparams.warmup_steps:
        return float(current_step) / float(max(1, hparams.warmup_steps))
    return 1.0


scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


writer = SummaryWriter(log_dir="runs/probing_run")


def log_confusion_matrix(writer, labels, preds, epoch, class_names=['0', '1']):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    writer.add_figure("ConfusionMatrix/val", fig, global_step=epoch)
    plt.close(fig)


step = 0

for epoch in range(hparams.num_epochs):
    model.train()
    epoch_loss = 0.0
    train_preds = []
    train_labels = []

    start_time = time.time()
    train_bar = tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}/{hparams.num_epochs}", leave=False)

    for x_batch, y_batch in train_bar:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()

        preds = (outputs > 0.5).float()
        train_preds.extend(preds.cpu().numpy())
        train_labels.extend(y_batch.cpu().numpy())

        train_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.6f}"
        })

        step += 1

    train_acc = accuracy_score(train_labels, train_preds)
    train_f1 = f1_score(train_labels, train_preds)
    avg_train_loss = epoch_loss / len(train_loader)
    elapsed = time.time() - start_time
    est_remaining = (hparams.num_epochs - (epoch + 1)) * elapsed

    print(f"\nEpoch {epoch+1} completed in {elapsed:.2f}s | Estimated time remaining: {est_remaining:.2f}s")
    print(f"Train Accuracy: {train_acc:.4f} | F1: {train_f1:.4f}")

    writer.add_scalar("Loss/train", avg_train_loss, epoch)
    writer.add_scalar("Accuracy/train", train_acc, epoch)
    writer.add_scalar("F1/train", train_f1, epoch)
    writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

    model.eval()
    val_preds = []
    val_labels = []
    val_loss = 0.0

    with torch.no_grad():
        val_bar = tqdm(val_loader, desc=f"[Val]   Epoch {epoch+1}", leave=False)
        for x_batch, y_batch in val_bar:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()

            preds = (outputs > 0.5).float()
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(y_batch.cpu().numpy())

            val_bar.set_postfix({
                "val_loss": f"{loss.item():.4f}"
            })

    val_acc = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds)
    avg_val_loss = val_loss / len(val_loader)

    print(f"Val Accuracy: {val_acc:.4f} | F1: {val_f1:.4f}")
    print("Classification Report:\n", classification_report(val_labels, val_preds, digits=4))

    writer.add_scalar("Loss/val", avg_val_loss, epoch)
    writer.add_scalar("Accuracy/val", val_acc, epoch)
    writer.add_scalar("F1/val", val_f1, epoch)
    log_confusion_matrix(writer, val_labels, val_preds, epoch)

writer.close()