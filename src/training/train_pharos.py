import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.data.dataset import PharosDepMapDataset
from src.evaluation.metrics import regression_metrics
from src.models.pharos import PharosModel


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        drug_fp = batch["drug_fp"].to(device)
        cell_expr = batch["cell_expr"].to(device)
        resistance_expr = batch["resistance_expr"].to(device)
        label = batch["label"].to(device)

        optimizer.zero_grad()
        pred = model(drug_fp, cell_expr, resistance_expr)
        loss = loss_fn(pred, label)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * drug_fp.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0

    all_preds = []
    all_labels = []

    for batch in loader:
        drug_fp = batch["drug_fp"].to(device)
        cell_expr = batch["cell_expr"].to(device)
        resistance_expr = batch["resistance_expr"].to(device)
        label = batch["label"].to(device)

        pred = model(drug_fp, cell_expr, resistance_expr)
        loss = loss_fn(pred, label)

        total_loss += loss.item() * drug_fp.size(0)

        all_preds.extend(pred.cpu().numpy().reshape(-1))
        all_labels.extend(label.cpu().numpy().reshape(-1))

    metrics = regression_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def main():
    set_seed(42)

    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")

    print(f"Using device: {device}")

    dataset = PharosDepMapDataset(max_rows=10_000)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

    sample = dataset[0]

    model = PharosModel(
        drug_dim=sample["drug_fp"].shape[0],
        cell_dim=sample["cell_expr"].shape[0],
        resistance_dim=sample["resistance_expr"].shape[0],
        hidden_dim=512,
        dropout=0.2,
    ).to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    epochs = 10
    patience = 3

    best_metrics = None
    best_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0

    history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate(model, val_loader, loss_fn, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics,
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"RMSE: {val_metrics['rmse']:.4f} | "
            f"MAE: {val_metrics['mae']:.4f} | "
            f"R2: {val_metrics['r2']:.4f} | "
            f"Pearson: {val_metrics['pearson']:.4f} | "
            f"Spearman: {val_metrics['spearman']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_metrics = dict(val_metrics)
            best_metrics["train_loss"] = train_loss
            best_metrics["best_epoch"] = best_epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}. Best epoch: {best_epoch}")
            break

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    history_df = pd.DataFrame(history)
    history_path = output_dir / "pharos_training_history.csv"
    history_df.to_csv(history_path, index=False)

    best_path = output_dir / "pharos_best_metrics.csv"
    pd.DataFrame([best_metrics]).to_csv(best_path, index=False)

    print("\nBest PHAROS metrics:")
    print(best_metrics)

    print(f"\nSaved training history to: {history_path}")
    print(f"Saved best metrics to: {best_path}")


if __name__ == "__main__":
    main()