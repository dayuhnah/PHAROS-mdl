import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.data.dataset import PharosDepMapDataset
from src.evaluation.metrics import regression_metrics
from src.models.pharos import PharosModel
from src.models.pharos_rx import PharosRXModel


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_unseen_drug_split(
    dataset,
    val_frac=0.2,
    seed=42,
):
    metadata = dataset.get_metadata()

    unique_drugs = np.asarray(
        metadata["broad_id"].dropna().unique(),
        dtype=object,
    ).copy()

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_drugs)

    n_val = int(len(unique_drugs) * val_frac)

    val_drugs = set(unique_drugs[:n_val])
    train_drugs = set(unique_drugs[n_val:])

    train_indices = metadata.index[
        metadata["broad_id"].isin(train_drugs)
    ].tolist()

    val_indices = metadata.index[
        metadata["broad_id"].isin(val_drugs)
    ].tolist()

    print(f"Unique drugs: {len(unique_drugs)}")
    print(f"Train drugs: {len(train_drugs)}")
    print(f"Validation unseen drugs: {len(val_drugs)}")
    print(f"Train rows: {len(train_indices)}")
    print(f"Validation rows: {len(val_indices)}")

    # Sanity check: absolutely no overlapping drugs
    overlap = train_drugs.intersection(val_drugs)

    if len(overlap) != 0:
        raise RuntimeError(
            f"Drug leakage detected: {len(overlap)} overlapping drugs"
        )

    print("Drug overlap between train/validation: 0 ✅")

    return train_indices, val_indices


def make_loaders(
    dataset,
    train_indices,
    val_indices,
    seed,
):
    train_dataset = Subset(
        dataset,
        train_indices,
    )

    val_dataset = Subset(
        dataset,
        val_indices,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader


def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    device,
):
    model.train()

    total_loss = 0.0

    for batch in loader:
        drug_fp = batch["drug_fp"].to(device)
        cell_expr = batch["cell_expr"].to(device)
        resistance_expr = batch["resistance_expr"].to(device)
        label = batch["label"].to(device)

        optimizer.zero_grad()

        pred = model(
            drug_fp,
            cell_expr,
            resistance_expr,
        )

        loss = loss_fn(
            pred,
            label,
        )

        loss.backward()
        optimizer.step()

        total_loss += (
            loss.item() * drug_fp.size(0)
        )

    return (
        total_loss / len(loader.dataset)
    )


@torch.no_grad()
def evaluate(
    model,
    loader,
    loss_fn,
    device,
):
    model.eval()

    total_loss = 0.0

    all_preds = []
    all_labels = []

    for batch in loader:
        drug_fp = batch["drug_fp"].to(device)
        cell_expr = batch["cell_expr"].to(device)
        resistance_expr = batch["resistance_expr"].to(device)
        label = batch["label"].to(device)

        pred = model(
            drug_fp,
            cell_expr,
            resistance_expr,
        )

        loss = loss_fn(
            pred,
            label,
        )

        total_loss += (
            loss.item() * drug_fp.size(0)
        )

        all_preds.extend(
            pred.cpu()
            .numpy()
            .reshape(-1)
        )

        all_labels.extend(
            label.cpu()
            .numpy()
            .reshape(-1)
        )

    metrics = regression_metrics(
        all_labels,
        all_preds,
    )

    metrics["loss"] = (
        total_loss / len(loader.dataset)
    )

    return metrics


def run_experiment(
    model_name,
    model,
    dataset,
    train_indices,
    val_indices,
    device,
    seed,
    checkpoint_name,
    epochs=10,
    patience=3,
):
    print("\n======================================")
    print(
        f"Unseen-drug experiment: {model_name}"
    )
    print(f"Seed: {seed}")
    print("======================================")

    # Reset randomness before each architecture
    set_seed(seed)

    # Fresh loaders ensure same initial batch order
    train_loader, val_loader = make_loaders(
        dataset=dataset,
        train_indices=train_indices,
        val_indices=val_indices,
        seed=seed,
    )

    model = model.to(device)

    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-5,
    )

    best_metrics = None
    best_model_state = None
    best_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
        )

        val_metrics = evaluate(
            model,
            val_loader,
            loss_fn,
            device,
        )

        print(
            f"{model_name} | "
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

            best_metrics = dict(
                val_metrics
            )

            best_metrics["train_loss"] = (
                train_loss
            )

            best_metrics["best_epoch"] = (
                best_epoch
            )

            best_metrics["seed"] = seed

            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }

            patience_counter = 0

        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(
                f"Early stopping for {model_name} "
                f"at epoch {epoch}. "
                f"Best epoch: {best_epoch}"
            )
            break

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    checkpoint_path = (
        output_dir
        / checkpoint_name
    )

    torch.save(
        {
            "model_state_dict": best_model_state,
            "best_epoch": best_epoch,
            "best_metrics": best_metrics,
            "seed": seed,
        },
        checkpoint_path,
    )

    print(
        f"Saved best checkpoint to: "
        f"{checkpoint_path}"
    )

    best_metrics["model"] = model_name

    return best_metrics


def main(seed: int = 42):

    set_seed(seed)

    print(f"Random seed: {seed}")

    device = torch.device("cpu")

    if torch.backends.mps.is_available():
        device = torch.device("mps")

    elif torch.cuda.is_available():
        device = torch.device("cuda")

    print(f"Using device: {device}")

    dataset = PharosDepMapDataset(
        max_rows=10_000
    )

    sample = dataset[0]

    drug_dim = sample["drug_fp"].shape[0]
    cell_dim = sample["cell_expr"].shape[0]
    resistance_dim = (
        sample["resistance_expr"].shape[0]
    )

    print("\nInput dimensions:")
    print(f"Drug: {drug_dim}")
    print(f"Cell: {cell_dim}")
    print(
        f"Resistance: {resistance_dim}"
    )

    # ---------------------------------
    # SAME unseen-drug split for both
    # ---------------------------------

    train_indices, val_indices = (
        make_unseen_drug_split(
            dataset,
            val_frac=0.2,
            seed=seed,
        )
    )

    results = []

    # =================================
    # Baseline PHAROS
    # =================================

    set_seed(seed)

    pharos = PharosModel(
        drug_dim=drug_dim,
        cell_dim=cell_dim,
        resistance_dim=resistance_dim,
        hidden_dim=512,
        dropout=0.2,
    )

    pharos_metrics = run_experiment(
        model_name="PHAROS",
        model=pharos,
        dataset=dataset,
        train_indices=train_indices,
        val_indices=val_indices,
        device=device,
        seed=seed,
        checkpoint_name=(
            f"pharos_unseen_drug_seed_"
            f"{seed}_best_model.pt"
        ),
    )

    results.append(
        pharos_metrics
    )

    # =================================
    # PHAROS-RX
    # =================================

    set_seed(seed)

    pharos_rx = PharosRXModel(
        drug_dim=drug_dim,
        cell_dim=cell_dim,
        resistance_dim=resistance_dim,
        hidden_dim=512,
        dropout=0.2,
    )

    rx_metrics = run_experiment(
        model_name="PHAROS-RX",
        model=pharos_rx,
        dataset=dataset,
        train_indices=train_indices,
        val_indices=val_indices,
        device=device,
        seed=seed,
        checkpoint_name=(
            f"pharos_rx_unseen_drug_seed_"
            f"{seed}_best_model.pt"
        ),
    )

    results.append(
        rx_metrics
    )

    # ---------------------------------
    # Save comparison
    # ---------------------------------

    results_df = pd.DataFrame(
        results
    )

    cols = [
        "model",
        "seed",
        "best_epoch",
        "train_loss",
        "loss",
        "mse",
        "rmse",
        "mae",
        "r2",
        "pearson",
        "spearman",
    ]

    results_df = results_df[
        [
            col
            for col in cols
            if col in results_df.columns
        ]
    ]

    print("\n======================================")
    print("FINAL UNSEEN-DRUG RESULTS")
    print("======================================")

    print(results_df)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    output_path = (
        output_dir
        / f"unseen_drug_seed_{seed}_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nSaved results to: "
        f"{output_path}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    main(seed=args.seed)