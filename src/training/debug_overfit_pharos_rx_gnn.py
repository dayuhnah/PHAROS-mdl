import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.data.dataset import PharosDepMapDataset
from src.data.graph_collate import pharos_graph_collate
from src.evaluation.metrics import regression_metrics
from src.models.pharos_rx_gnn import PharosRXGNNModel


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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

        drug_graph = batch[
            "drug_graph"
        ].to(device)

        cell_expr = batch[
            "cell_expr"
        ].to(device)

        resistance_expr = batch[
            "resistance_expr"
        ].to(device)

        label = batch[
            "label"
        ].to(device)

        optimizer.zero_grad()

        pred = model(
            drug_graph=drug_graph,
            cell_expr=cell_expr,
            resistance_expr=resistance_expr,
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

        drug_graph = batch[
            "drug_graph"
        ].to(device)

        cell_expr = batch[
            "cell_expr"
        ].to(device)

        resistance_expr = batch[
            "resistance_expr"
        ].to(device)

        label = batch[
            "label"
        ].to(device)

        pred = model(
            drug_graph=drug_graph,
            cell_expr=cell_expr,
            resistance_expr=resistance_expr,
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


def main(seed: int = 42):

    set_seed(seed)

    print(
        f"Random seed: {seed}"
    )

    # =================================
    # Device
    # =================================

    device = torch.device("cpu")

    if torch.backends.mps.is_available():
        device = torch.device("mps")

    elif torch.cuda.is_available():
        device = torch.device("cuda")

    print(
        f"Using device: {device}"
    )

    # =================================
    # Dataset
    # =================================

    dataset = PharosDepMapDataset(
        max_rows=256,
        use_drug_graphs=True,
    )

    sample = dataset[0]

    cell_dim = (
        sample["cell_expr"].shape[0]
    )

    resistance_dim = (
        sample["resistance_expr"]
        .shape[0]
    )

    node_dim = (
        sample["drug_graph"]
        .x.shape[1]
    )

    edge_dim = (
        sample["drug_graph"]
        .edge_attr.shape[1]
    )

    print(
        "\nPHAROS-RX-GNN input dimensions:"
    )

    print(
        f"Node features: {node_dim}"
    )

    print(
        f"Edge features: {edge_dim}"
    )

    print(
        f"Cell: {cell_dim}"
    )

    print(
        f"Resistance: {resistance_dim}"
    )

    # =================================
    # Train / validation split
    # =================================

    train_dataset = dataset
    val_dataset = dataset

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        generator=(
            torch.Generator()
            .manual_seed(seed)
        ),
        collate_fn=(
            pharos_graph_collate
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=(
            pharos_graph_collate
        ),
    )

    print(
        f"\nTrain rows: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation rows: "
        f"{len(val_dataset)}"
    )

    # =================================
    # Model
    # =================================

    set_seed(seed)

    model = PharosRXGNNModel(
        cell_dim=cell_dim,
        resistance_dim=resistance_dim,

        node_dim=node_dim,
        edge_dim=edge_dim,

        hidden_dim=512,

        drug_gnn_hidden_dim=128,

        drug_out_dim=256,
        cell_out_dim=256,
        resistance_out_dim=64,

        gnn_heads=4,

        dropout=0.0,
    ).to(device)

    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-5,
    )

    epochs = 100
    patience = 100

    best_metrics = None
    best_epoch = 0

    best_val_loss = float(
        "inf"
    )

    best_model_state = None

    patience_counter = 0

    history = []

    # =================================
    # Training
    # =================================

    for epoch in range(
        1,
        epochs + 1,
    ):

        train_loss = (
            train_one_epoch(
                model,
                train_loader,
                optimizer,
                loss_fn,
                device,
            )
        )

        val_metrics = evaluate(
            model,
            val_loader,
            loss_fn,
            device,
        )

        row = {
            "epoch": epoch,
            "seed": seed,
            "train_loss": train_loss,
            **val_metrics,
        }

        history.append(
            row
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: "
            f"{train_loss:.4f} | "
            f"Val Loss: "
            f"{val_metrics['loss']:.4f} | "
            f"RMSE: "
            f"{val_metrics['rmse']:.4f} | "
            f"MAE: "
            f"{val_metrics['mae']:.4f} | "
            f"R2: "
            f"{val_metrics['r2']:.4f} | "
            f"Pearson: "
            f"{val_metrics['pearson']:.4f} | "
            f"Spearman: "
            f"{val_metrics['spearman']:.4f}"
        )

        # -----------------------------
        # Best checkpoint
        # -----------------------------

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

            patience_counter = 0

        else:

            patience_counter += 1

        # -----------------------------
        # Early stopping
        # -----------------------------

        if (
            patience_counter
            >= patience
        ):

            print(
                "Early stopping "
                f"triggered at epoch "
                f"{epoch}. "
                f"Best epoch: "
                f"{best_epoch}"
            )

            break

    # =================================
    # Save
    # =================================

    output_dir = Path(
        "outputs"
    )

    output_dir.mkdir(
        exist_ok=True
    )

    checkpoint_path = (
        output_dir
        / (
            f"pharos_rx_gnn_seed_"
            f"{seed}_best_model.pt"
        )
    )

    torch.save(
        {
            "model_state_dict":
                best_model_state,

            "best_epoch":
                best_epoch,

            "best_metrics":
                best_metrics,

            "seed":
                seed,

            "cell_dim":
                cell_dim,

            "resistance_dim":
                resistance_dim,

            "node_dim":
                node_dim,

            "edge_dim":
                edge_dim,

            "drug_gnn_hidden_dim":
                128,

            "drug_out_dim":
                256,

            "cell_out_dim":
                256,

            "resistance_out_dim":
                64,

            "gnn_heads":
                4,
        },
        checkpoint_path,
    )

    print(
        f"\nSaved best model to: "
        f"{checkpoint_path}"
    )

    # ---------------------------------
    # History
    # ---------------------------------

    history_df = pd.DataFrame(
        history
    )

    history_path = (
        output_dir
        / (
            f"pharos_rx_gnn_seed_"
            f"{seed}_training_history.csv"
        )
    )

    history_df.to_csv(
        history_path,
        index=False,
    )

    # ---------------------------------
    # Best metrics
    # ---------------------------------

    best_path = (
        output_dir
        / (
            f"pharos_rx_gnn_seed_"
            f"{seed}_best_metrics.csv"
        )
    )

    pd.DataFrame(
        [best_metrics]
    ).to_csv(
        best_path,
        index=False,
    )

    print(
        "\nBest PHAROS-RX-GNN "
        "metrics:"
    )

    print(
        best_metrics
    )

    print(
        f"\nSaved training history "
        f"to: {history_path}"
    )

    print(
        f"Saved best metrics to: "
        f"{best_path}"
    )


if __name__ == "__main__":

    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed for training"
        ),
    )

    args = (
        parser.parse_args()
    )

    main(
        seed=args.seed
    )