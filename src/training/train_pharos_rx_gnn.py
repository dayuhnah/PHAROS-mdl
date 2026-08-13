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
from src.data.graph_collate import pharos_graph_collate
from src.evaluation.metrics import regression_metrics
from src.models.pharos_rx_gnn import PharosRXGNNModel


# =========================================================
# Reproducibility
# =========================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# =========================================================
# Helper: copy model state to CPU
# =========================================================

def get_cpu_state_dict(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


# =========================================================
# Training
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

        drug_graph = batch["drug_graph"].to(device)

        cell_expr = batch[
            "cell_expr"
        ].to(device)

        resistance_expr = batch[
            "resistance_expr"
        ].to(device)

        label = batch[
            "label"
        ].to(device)

        optimizer.zero_grad(
            set_to_none=True
        )

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


# =========================================================
# Validation
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


# =========================================================
# Main
# =========================================================

def main(
    seed: int = 42,
    resume: bool = False,
):

    set_seed(seed)

    print(
        f"Random seed: {seed}"
    )

    # =====================================================
    # Device
    # =====================================================

    device = torch.device(
        "cpu"
    )

    if torch.backends.mps.is_available():
        device = torch.device(
            "mps"
        )

    elif torch.cuda.is_available():
        device = torch.device(
            "cuda"
        )

    print(
        f"Using device: {device}"
    )

    # =====================================================
    # Dataset
    # =====================================================

    dataset = PharosDepMapDataset(
        max_rows=50_000,
        use_drug_graphs=True,
    )

    sample = dataset[0]

    cell_dim = (
        sample["cell_expr"]
        .shape[0]
    )

    resistance_dim = (
        sample["resistance_expr"]
        .shape[0]
    )

    node_dim = (
        sample["drug_graph"]
        .x
        .shape[1]
    )

    edge_dim = (
        sample["drug_graph"]
        .edge_attr
        .shape[1]
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

    # =====================================================
    # Train / validation split
    # =====================================================

    train_size = int(
        0.8 * len(dataset)
    )

    val_size = (
        len(dataset)
        - train_size
    )

    split_generator = (
        torch.Generator()
        .manual_seed(seed)
    )

    train_dataset, val_dataset = (
        random_split(
            dataset,
            [
                train_size,
                val_size,
            ],
            generator=split_generator,
        )
    )

    # Separate generator so we can
    # restore shuffle state after a crash
    train_generator = (
        torch.Generator()
        .manual_seed(seed)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        generator=train_generator,
        collate_fn=pharos_graph_collate,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=pharos_graph_collate,
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

        dropout=0.2,
    ).to(device)

    model_config = {
        "cell_dim": cell_dim,
        "resistance_dim": resistance_dim,
        "node_dim": node_dim,
        "edge_dim": edge_dim,
        "hidden_dim": 512,
        "drug_gnn_hidden_dim": 128,
        "drug_out_dim": 256,
        "cell_out_dim": 256,
        "resistance_out_dim": 64,
        "gnn_heads": 4,
        "dropout": 0.2,
    }

    # =====================================================
    # Optimizer
    # =====================================================

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

    # =====================================================
    # Training configuration
    # =====================================================

    epochs = 80
    patience = 10

    start_epoch = 1

    best_metrics = None
    best_epoch = 0

    best_val_loss = float(
        "inf"
    )

    patience_counter = 0

    history = []

    # =====================================================
    # Output paths
    # =====================================================

    output_dir = Path(
        "outputs"
    )

    output_dir.mkdir(
        exist_ok=True
    )

    best_checkpoint_path = (
        output_dir
        / (
            f"pharos_rx_gnn_50k_"
            f"seed_{seed}_best_model.pt"
        )
    )

    last_checkpoint_path = (
        output_dir
        / (
            f"pharos_rx_gnn_50k_"
            f"seed_{seed}_last_checkpoint.pt"
        )
    )

    history_path = (
        output_dir
        / (
            f"pharos_rx_gnn_50k_"
            f"seed_{seed}_training_history.csv"
        )
    )

    best_metrics_path = (
        output_dir
        / (
            f"pharos_rx_gnn_50k_"
            f"seed_{seed}_best_metrics.csv"
        )
    )

    # =====================================================
    # Resume from interruption
    # =====================================================

    if resume:

        if not last_checkpoint_path.exists():

            raise FileNotFoundError(
                "Resume requested, but no "
                "recovery checkpoint exists:\n"
                f"{last_checkpoint_path}"
            )

        print(
            "\n======================================"
        )

        print(
            "RESUMING TRAINING"
        )

        print(
            "======================================"
        )

        checkpoint = torch.load(
            last_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        model = model.to(
            device
        )

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        scheduler.load_state_dict(
            checkpoint[
                "scheduler_state_dict"
            ]
        )

        start_epoch = (
            checkpoint["epoch"]
            + 1
        )

        best_epoch = checkpoint[
            "best_epoch"
        ]

        best_val_loss = checkpoint[
            "best_val_loss"
        ]

        best_metrics = checkpoint[
            "best_metrics"
        ]

        patience_counter = checkpoint.get(
            "patience_counter",
            0,
        )

        # Restore DataLoader shuffle state
        if (
            "train_generator_state"
            in checkpoint
        ):
            train_generator.set_state(
                checkpoint[
                    "train_generator_state"
                ]
            )

        # Restore history
        if history_path.exists():

            history = (
                pd.read_csv(
                    history_path
                )
                .to_dict(
                    orient="records"
                )
            )

        print(
            f"Resuming from epoch: "
            f"{start_epoch}"
        )

        print(
            f"Previous best epoch: "
            f"{best_epoch}"
        )

        print(
            f"Previous best val loss: "
            f"{best_val_loss:.6f}"
        )

    # =====================================================
    # Training loop
    # =====================================================

    for epoch in range(
        start_epoch,
        epochs + 1,
    ):

        # -------------------------------------------------
        # Train
        # -------------------------------------------------

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
        )

        # -------------------------------------------------
        # Validate
        # -------------------------------------------------

        val_metrics = evaluate(
            model,
            val_loader,
            loss_fn,
            device,
        )

        # -------------------------------------------------
        # LR scheduler
        # -------------------------------------------------

        scheduler.step(
            val_metrics["loss"]
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        # -------------------------------------------------
        # History
        # -------------------------------------------------

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
            f"{val_metrics['spearman']:.4f} | "
            f"LR: "
            f"{current_lr:.2e}"
        )

        # =================================================
        # NEW BEST MODEL
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

            best_model_state = (
                get_cpu_state_dict(
                    model
                )
            )

            # Save best model IMMEDIATELY
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

                    "patience_counter":
                        patience_counter,

                    "train_generator_state":
                        train_generator.get_state(),

                    "seed":
                        seed,

                    "model_config":
                        model_config,
                },
                best_checkpoint_path,
            )

            # Save best metrics immediately
            pd.DataFrame(
                [best_metrics]
            ).to_csv(
                best_metrics_path,
                index=False,
            )

            print(
                "🔥 Saved new best checkpoint: "
                f"{best_checkpoint_path}"
            )

            del best_model_state

        else:

            patience_counter += 1

        # =================================================
        # RECOVERY CHECKPOINT EVERY EPOCH
        # =================================================

        last_model_state = (
            get_cpu_state_dict(
                model
            )
        )

        torch.save(
            {
                "epoch":
                    epoch,

                "model_state_dict":
                    last_model_state,

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

                "patience_counter":
                    patience_counter,

                "train_generator_state":
                    train_generator.get_state(),

                "seed":
                    seed,

                "model_config":
                    model_config,
            },
            last_checkpoint_path,
        )

        del last_model_state

        # -------------------------------------------------
        # Save history every epoch
        # -------------------------------------------------

        pd.DataFrame(
            history
        ).to_csv(
            history_path,
            index=False,
        )

        # =================================================
        # Memory monitoring / cleanup
        # =================================================

        if device.type == "mps":

            try:

                allocated_gb = (
                    torch.mps
                    .current_allocated_memory()
                    / 1024**3
                )

                driver_gb = (
                    torch.mps
                    .driver_allocated_memory()
                    / 1024**3
                )

                print(
                    "MPS memory | "
                    f"allocated: "
                    f"{allocated_gb:.2f} GB | "
                    f"driver: "
                    f"{driver_gb:.2f} GB"
                )

            except Exception as error:

                print(
                    "Could not read MPS "
                    f"memory stats: {error}"
                )

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
                f"triggered at epoch "
                f"{epoch}."
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
        "PHAROS-RX-GNN TRAINING COMPLETE"
    )

    print(
        "======================================"
    )

    print(
        f"\nBest epoch: "
        f"{best_epoch}"
    )

    print(
        "\nBest PHAROS-RX-GNN metrics:"
    )

    print(
        best_metrics
    )

    print(
        "\nFiles:"
    )

    print(
        f"Best model: "
        f"{best_checkpoint_path}"
    )

    print(
        f"Recovery checkpoint: "
        f"{last_checkpoint_path}"
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
# Command line
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume training from the "
            "latest recovery checkpoint."
        ),
    )

    args = parser.parse_args()

    main(
        seed=args.seed,
        resume=args.resume,
    )