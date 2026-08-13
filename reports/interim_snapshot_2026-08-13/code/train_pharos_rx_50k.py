import argparse
import gc
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.data.dataset import PharosDepMapDataset
from src.evaluation.metrics import regression_metrics
from src.models.pharos_rx import PharosRXModel


# =========================================================
# Reproducibility
# =========================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# =========================================================
# Train
# =========================================================

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

        optimizer.zero_grad(
            set_to_none=True
        )

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
            loss.item()
            * label.size(0)
        )

    return (
        total_loss
        / len(loader.dataset)
    )


# =========================================================
# Evaluate
# =========================================================

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
            loss.item()
            * label.size(0)
        )

        all_preds.extend(
            pred.detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )

        all_labels.extend(
            label.detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )

    metrics = regression_metrics(
        all_labels,
        all_preds,
    )

    metrics["loss"] = (
        total_loss
        / len(loader.dataset)
    )

    return metrics


# =========================================================
# Main
# =========================================================

def main(seed: int = 42):

    set_seed(seed)

    print(
        f"Random seed: {seed}"
    )

    # =====================================================
    # Device
    # =====================================================

    device = torch.device("cpu")

    if torch.backends.mps.is_available():
        device = torch.device("mps")

    elif torch.cuda.is_available():
        device = torch.device("cuda")

    print(
        f"Using device: {device}"
    )

    # =====================================================
    # Dataset
    # =====================================================

    # IMPORTANT:
    # Start with the SAME 50k sampled rows as the GNN run.
    dataset = PharosDepMapDataset(
        max_rows=50_000,
        use_drug_graphs=False,
    )

    # =====================================================
    # Match GNN graph coverage
    # =====================================================

    graph_cache_path = Path(
        "data/processed/pharos_drug_graphs.pt"
    )

    graph_cache = torch.load(
        graph_cache_path,
        map_location="cpu",
        weights_only=False,
    )

    graph_drug_ids = {
        str(broad_id)
        for broad_id in graph_cache.keys()
    }

    before = len(dataset.response)

    dataset.response = dataset.response[
        dataset.response["broad_id"]
        .astype(str)
        .isin(graph_drug_ids)
    ].reset_index(
        drop=True
    )

    removed = (
        before - len(dataset.response)
    )

    print(
        f"Removed {removed} rows "
        f"without molecular graphs "
        f"to match GNN dataset."
    )

    print(
        f"Final dataset rows: "
        f"{len(dataset)}"
    )

    # We no longer need the graph cache
    del graph_cache
    gc.collect()

    sample = dataset[0]

    drug_dim = (
        sample["drug_fp"].shape[0]
    )

    cell_dim = (
        sample["cell_expr"].shape[0]
    )

    resistance_dim = (
        sample["resistance_expr"].shape[0]
    )

    print(
        "\nPHAROS-RX 50k input dimensions:"
    )

    print(
        f"Drug: {drug_dim}"
    )

    print(
        f"Cell: {cell_dim}"
    )

    print(
        f"Resistance: {resistance_dim}"
    )

    # =====================================================
    # SAME 80/20 split
    # =====================================================

    train_size = int(
        0.8 * len(dataset)
    )

    val_size = (
        len(dataset)
        - train_size
    )

    train_dataset, val_dataset = (
        random_split(
            dataset,
            [
                train_size,
                val_size,
            ],
            generator=(
                torch.Generator()
                .manual_seed(seed)
            ),
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        generator=(
            torch.Generator()
            .manual_seed(seed)
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    print(
        f"\nTrain rows: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation rows: "
        f"{len(val_dataset)}"
    )

    # =====================================================
    # Model
    # =====================================================

    set_seed(seed)

    model = PharosRXModel(
        drug_dim=drug_dim,
        cell_dim=cell_dim,
        resistance_dim=resistance_dim,
        hidden_dim=512,
        dropout=0.2,
    ).to(device)

    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-5,
    )

    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
        )
    )

    # SAME training budget as GNN
    epochs = 80
    patience = 10

    best_metrics = None
    best_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0

    history = []

    # =====================================================
    # Outputs
    # =====================================================

    output_dir = Path("outputs")

    output_dir.mkdir(
        exist_ok=True
    )

    best_checkpoint_path = (
        output_dir
        / f"pharos_rx_morgan_50k_seed_{seed}_best_model.pt"
    )

    history_path = (
        output_dir
        / f"pharos_rx_morgan_50k_seed_{seed}_training_history.csv"
    )

    best_metrics_path = (
        output_dir
        / f"pharos_rx_morgan_50k_seed_{seed}_best_metrics.csv"
    )

    # =====================================================
    # Training loop
    # =====================================================

    for epoch in range(
        1,
        epochs + 1,
    ):

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

        scheduler.step(
            val_metrics["loss"]
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        row = {
            "epoch": epoch,
            "seed": seed,
            "train_loss": train_loss,
            "lr": current_lr,
            **val_metrics,
        }

        history.append(
            row
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"RMSE: {val_metrics['rmse']:.4f} | "
            f"MAE: {val_metrics['mae']:.4f} | "
            f"R2: {val_metrics['r2']:.4f} | "
            f"Pearson: {val_metrics['pearson']:.4f} | "
            f"Spearman: {val_metrics['spearman']:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # =================================================
        # New best
        # =================================================

        if (
            val_metrics["loss"]
            < best_val_loss
        ):

            best_val_loss = (
                val_metrics["loss"]
            )

            best_epoch = epoch

            best_metrics = dict(
                val_metrics
            )

            best_metrics[
                "train_loss"
            ] = train_loss

            best_metrics[
                "best_epoch"
            ] = best_epoch

            best_metrics[
                "seed"
            ] = seed

            best_metrics[
                "lr"
            ] = current_lr

            patience_counter = 0

            best_model_state = {
                key:
                    value
                    .detach()
                    .cpu()
                    .clone()

                for key, value
                in model
                .state_dict()
                .items()
            }

            torch.save(
                {
                    "epoch":
                        epoch,

                    "model_state_dict":
                        best_model_state,

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "scheduler_state_dict":
                        scheduler.state_dict(),

                    "best_epoch":
                        best_epoch,

                    "best_val_loss":
                        best_val_loss,

                    "best_metrics":
                        best_metrics,

                    "seed":
                        seed,

                    "drug_dim":
                        drug_dim,

                    "cell_dim":
                        cell_dim,

                    "resistance_dim":
                        resistance_dim,
                },
                best_checkpoint_path,
            )

            pd.DataFrame(
                [best_metrics]
            ).to_csv(
                best_metrics_path,
                index=False,
            )

            print(
                "🔥 Saved new best PHAROS-RX "
                f"checkpoint: {best_checkpoint_path}"
            )

        else:

            patience_counter += 1

        # Save history continuously
        pd.DataFrame(
            history
        ).to_csv(
            history_path,
            index=False,
        )

        if device.type == "mps":
            torch.mps.empty_cache()

        gc.collect()

        # =================================================
        # Early stopping
        # =================================================

        if (
            patience_counter
            >= patience
        ):

            print(
                "\nEarly stopping "
                f"triggered at epoch {epoch}."
            )

            print(
                f"Best epoch: "
                f"{best_epoch}"
            )

            break

    # =====================================================
    # Finished
    # =====================================================

    print(
        "\n======================================"
    )

    print(
        "PHAROS-RX MORGAN 50K COMPLETE"
    )

    print(
        "======================================"
    )

    print(
        f"\nBest epoch: "
        f"{best_epoch}"
    )

    print(
        "\nBest metrics:"
    )

    print(
        best_metrics
    )

    print(
        f"\nBest model: "
        f"{best_checkpoint_path}"
    )

    print(
        f"Training history: "
        f"{history_path}"
    )

    print(
        f"Best metrics: "
        f"{best_metrics_path}"
    )


# =========================================================
# CLI
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    main(
        seed=args.seed
    )