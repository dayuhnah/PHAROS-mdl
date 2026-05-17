import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.data.dataset import PharosDepMapDataset
from src.evaluation.metrics import regression_metrics
from src.models.ablations import (
    DrugExpressionModel,
    DrugExpressionResistanceModel,
    DrugOnlyModel,
)


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


def run_experiment(model_name, model, train_loader, val_loader, device, epochs=5):
    print(f"\n==============================")
    print(f"Running experiment: {model_name}")
    print(f"==============================")

    model = model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    final_metrics = None

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        final_metrics = val_metrics

        print(
            f"{model_name} | Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"RMSE: {val_metrics['rmse']:.4f} | "
            f"MAE: {val_metrics['mae']:.4f} | "
            f"R2: {val_metrics['r2']:.4f} | "
            f"Pearson: {val_metrics['pearson']:.4f} | "
            f"Spearman: {val_metrics['spearman']:.4f}"
        )

    final_metrics = dict(final_metrics)
    final_metrics["model"] = model_name

    return final_metrics


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

    drug_dim = sample["drug_fp"].shape[0]
    cell_dim = sample["cell_expr"].shape[0]
    resistance_dim = sample["resistance_expr"].shape[0]

    experiments = [
        (
            "Drug Only",
            DrugOnlyModel(
                drug_dim=drug_dim,
                hidden_dim=512,
                dropout=0.2,
            ),
        ),
        (
            "Drug + Expression",
            DrugExpressionModel(
                drug_dim=drug_dim,
                cell_dim=cell_dim,
                hidden_dim=512,
                dropout=0.2,
            ),
        ),
        (
            "PHAROS Baseline: Drug + Expression + Resistance",
            DrugExpressionResistanceModel(
                drug_dim=drug_dim,
                cell_dim=cell_dim,
                resistance_dim=resistance_dim,
                hidden_dim=512,
                dropout=0.2,
            ),
        ),
    ]

    results = []

    for model_name, model in experiments:
        metrics = run_experiment(
            model_name=model_name,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=5,
        )
        results.append(metrics)

    results_df = pd.DataFrame(results)

    cols = ["model", "loss", "mse", "rmse", "mae", "r2", "pearson", "spearman"]
    cols = [col for col in cols if col in results_df.columns]
    results_df = results_df[cols]

    print("\n==============================")
    print("Final ablation results")
    print("==============================")
    print(results_df)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "ablation_results.csv"
    results_df.to_csv(output_path, index=False)

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()