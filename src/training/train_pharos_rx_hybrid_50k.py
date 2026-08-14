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
from src.models.pharos_rx_hybrid import PharosRXHybridModel


# =========================================================
# Reproducibility
# =========================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_cpu_state_dict(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


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

        drug_graph = batch[
            "drug_graph"
        ].to(device)

        drug_fp = batch[
            "drug_fp"
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

        optimizer.zero_grad(
            set_to_none=True
        )

        pred = model(
            drug_graph=drug_graph,
            drug_fp=drug_fp,
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

    gate_values = []

    for batch in loader:

        drug_graph = batch[
            "drug_graph"
        ].to(device)

        drug_fp = batch[
            "drug_fp"
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

        outputs = model(
            drug_graph=drug_graph,
            drug_fp=drug_fp,
            cell_expr=cell_expr,
            resistance_expr=resistance_expr,
            return_gate=True,
        )

        pred = outputs[
            "prediction"
        ]

        gate = outputs[
            "gate"
        ]

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

        gate_values.append(
            gate.detach()
            .cpu()
        )

    metrics = regression_metrics(
        all_labels,
        all_preds,
    )

    metrics["loss"] = (
        total_loss
        / len(loader.dataset)
    )

    # =====================================
    # Gate statistics
    # =====================================

    all_gates = torch.cat(
        gate_values,
        dim=0,
    )

    metrics[
        "mean_gnn_gate"
    ] = all_gates.mean().item()

    metrics[
        "mean_morgan_gate"
    ] = (
        1.0
        - all_gates
    ).mean().item()

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

    drug_fp_dim = (
        sample["drug_fp"]
        .shape[0]
    )

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
        "\nPHAROS-RX-HYBRID dimensions:"
    )

    print(
        f"Drug FP: {drug_fp_dim}"
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
    # SAME train / validation split
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

    model = PharosRXHybridModel(
        drug_fp_dim=drug_fp_dim,
        cell_dim=cell_dim,
        resistance_dim=resistance_dim,

        node_dim=node_dim,
        edge_dim=edge_dim,

        hidden_dim=512,

        drug_gnn_hidden_dim=128,

        drug_out_dim=256,
        cell_out_dim=256,
        resistance_out_dim=64,

        dropout=0.2,
    ).to(device)

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
            f"pharos_rx_hybrid_50k_"
            f"seed_{seed}_best_model.pt"
        )
    )

    last_checkpoint_path = (
        output_dir
        / (
            f"pharos_rx_hybrid_50k_"
            f"seed_{seed}_last_checkpoint.pt"
        )
    )

    history_path = (
        output_dir
        / (
            f"pharos_rx_hybrid_50k_"
            f"seed_{seed}_training_history.csv"
        )
    )

    best_metrics_path = (
        output_dir
        / (
            f"pharos_rx_hybrid_50k_"
            f"seed_{seed}_best_metrics.csv"
        )
    )

    # =====================================================
    # Resume
    # =====================================================

    if resume:

        if not last_checkpoint_path.exists():

            raise FileNotFoundError(
                "No recovery checkpoint found:\n"
                f"{last_checkpoint_path}"
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

        if (
            "train_generator_state"
            in checkpoint
        ):

            train_generator.set_state(
                checkpoint[
                    "train_generator_state"
                ]
            )

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
            f"\nResuming from epoch "
            f"{start_epoch}"
        )

    # =====================================================
    # Training
    # =====================================================

    for epoch in range(
        start_epoch,
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
            "epoch":
                epoch,

            "seed":
                seed,

            "train_loss":
                train_loss,

            "lr":
                current_lr,

            **val_metrics,
        }

        history.append(
            row
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_metrics['loss']:.4f} | "
            f"RMSE: {val_metrics['rmse']:.4f} | "
            f"MAE: {val_metrics['mae']:.4f} | "
            f"R2: {val_metrics['r2']:.4f} | "
            f"Pearson: {val_metrics['pearson']:.4f} | "
            f"Spearman: {val_metrics['spearman']:.4f} | "
            f"GNN gate: "
            f"{val_metrics['mean_gnn_gate']:.3f} | "
            f"Morgan gate: "
            f"{val_metrics['mean_morgan_gate']:.3f} | "
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

            best_model_state = (
                get_cpu_state_dict(
                    model
                )
            )

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
                "🔥 Saved new best hybrid checkpoint"
            )

            del best_model_state

        else:

            patience_counter += 1

        # =================================================
        # Recovery checkpoint
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
            },
            last_checkpoint_path,
        )

        del last_model_state

        pd.DataFrame(
            history
        ).to_csv(
            history_path,
            index=False,
        )

        # =================================================
        # MPS cleanup
        # =================================================

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
                f"\nEarly stopping at "
                f"epoch {epoch}."
            )

            print(
                f"Best epoch: "
                f"{best_epoch}"
            )

            break

    # =====================================================
    # Done
    # =====================================================

    print(
        "\n======================================"
    )

    print(
        "PHAROS-RX-HYBRID 50K COMPLETE"
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
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    args = parser.parse_args()

    main(
        seed=args.seed,
        resume=args.resume,
    )