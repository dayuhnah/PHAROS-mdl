import argparse
import gc
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import (
    DataLoader,
    random_split,
)

from src.data.dataset import (
    PharosDepMapDataset,
)

from src.data.graph_collate import (
    pharos_graph_collate,
)

from src.evaluation.metrics import (
    regression_metrics,
)

from src.models.pharos_rx_dual_hybrid import (
    PharosRXDualHybridModel,
)


# =========================================================
# Reproducibility
# =========================================================

def set_seed(seed: int = 42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)


# =========================================================
# Copy model weights safely to CPU
# =========================================================

def get_cpu_state_dict(model):

    return {
        key: value.detach().cpu().clone()
        for key, value
        in model.state_dict().items()
    }


# =========================================================
# Train one epoch
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

        # -----------------------------------------
        # Move inputs to device
        # -----------------------------------------

        drug_graph = batch[
            "drug_graph"
        ].to(device)

        drug_fp = batch[
            "drug_fp"
        ].to(device)

        cell_expr = batch[
            "cell_expr"
        ].to(device)

        ppi_expression = batch[
            "ppi_expression"
        ].to(device)

        resistance_expr = batch[
            "resistance_expr"
        ].to(device)

        labels = batch[
            "label"
        ].to(device)

        # -----------------------------------------
        # Forward
        # -----------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        predictions = model(
            drug_graph=drug_graph,
            drug_fp=drug_fp,
            cell_expr=cell_expr,
            ppi_expression=ppi_expression,
            resistance_expr=resistance_expr,
        )

        # -----------------------------------------
        # Loss
        # -----------------------------------------

        loss = loss_fn(
            predictions,
            labels,
        )

        # -----------------------------------------
        # Backpropagation
        # -----------------------------------------

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
            * labels.size(0)
        )

    return (
        total_loss
        / len(loader.dataset)
    )


# =========================================================
# Evaluation
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

    all_predictions = []

    all_labels = []

    drug_gate_values = []

    cell_gate_values = []

    # =====================================================
    # Evaluate batches
    # =====================================================

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

        ppi_expression = batch[
            "ppi_expression"
        ].to(device)

        resistance_expr = batch[
            "resistance_expr"
        ].to(device)

        labels = batch[
            "label"
        ].to(device)

        # -----------------------------------------
        # Forward with gate diagnostics
        # -----------------------------------------

        output = model(
            drug_graph=drug_graph,
            drug_fp=drug_fp,
            cell_expr=cell_expr,
            ppi_expression=ppi_expression,
            resistance_expr=resistance_expr,
            return_gates=True,
        )

        predictions = output[
            "prediction"
        ]

        drug_gate = output[
            "drug_gate"
        ]

        cell_gate = output[
            "cell_gate"
        ]

        # -----------------------------------------
        # Loss
        # -----------------------------------------

        loss = loss_fn(
            predictions,
            labels,
        )

        total_loss += (
            loss.item()
            * labels.size(0)
        )

        # -----------------------------------------
        # Store predictions
        # -----------------------------------------

        all_predictions.extend(
            predictions
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )

        all_labels.extend(
            labels
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )

        # -----------------------------------------
        # Store gates
        # -----------------------------------------

        drug_gate_values.append(
            drug_gate
            .detach()
            .cpu()
        )

        cell_gate_values.append(
            cell_gate
            .detach()
            .cpu()
        )

    # =====================================================
    # Regression metrics
    # =====================================================

    metrics = regression_metrics(
        all_labels,
        all_predictions,
    )

    metrics[
        "loss"
    ] = (
        total_loss
        / len(loader.dataset)
    )

    # =====================================================
    # Gate statistics
    # =====================================================

    all_drug_gates = torch.cat(
        drug_gate_values,
        dim=0,
    )

    all_cell_gates = torch.cat(
        cell_gate_values,
        dim=0,
    )

    # Drug branch:
    #
    # gate = 1 -> GNN
    # gate = 0 -> Morgan

    metrics[
        "mean_gnn_drug_gate"
    ] = (
        all_drug_gates
        .mean()
        .item()
    )

    metrics[
        "mean_morgan_gate"
    ] = (
        1.0
        - all_drug_gates
    ).mean().item()

    # Cell branch:
    #
    # gate = 1 -> PPI GNN
    # gate = 0 -> Expression MLP

    metrics[
        "mean_ppi_cell_gate"
    ] = (
        all_cell_gates
        .mean()
        .item()
    )

    metrics[
        "mean_expression_gate"
    ] = (
        1.0
        - all_cell_gates
    ).mean().item()

    return metrics


# =========================================================
# Main
# =========================================================

def main(
    seed: int = 42,
    resume: bool = False,
):

    # =====================================================
    # Seed
    # =====================================================

    set_seed(seed)

    print(
        f"Random seed: {seed}"
    )

    # =====================================================
    # Device
    # =====================================================

    if torch.backends.mps.is_available():

        device = torch.device(
            "mps"
        )

    elif torch.cuda.is_available():

        device = torch.device(
            "cuda"
        )

    else:

        device = torch.device(
            "cpu"
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

    # =====================================================
    # Load PPI graph
    # =====================================================

    ppi_graph_path = Path(
        "data/processed/pharos_ppi_graph.pt"
    )

    if not ppi_graph_path.exists():

        raise FileNotFoundError(
            f"PPI graph not found: "
            f"{ppi_graph_path}"
        )

    ppi = torch.load(
        ppi_graph_path,
        map_location="cpu",
        weights_only=False,
    )

    # =====================================================
    # Dimensions
    # =====================================================

    drug_fp_dim = (
        sample[
            "drug_fp"
        ].shape[0]
    )

    cell_dim = (
        sample[
            "cell_expr"
        ].shape[0]
    )

    ppi_expression_dim = (
        sample[
            "ppi_expression"
        ].shape[0]
    )

    resistance_dim = (
        sample[
            "resistance_expr"
        ].shape[0]
    )

    node_dim = (
        sample[
            "drug_graph"
        ].x.shape[1]
    )

    edge_dim = (
        sample[
            "drug_graph"
        ].edge_attr.shape[1]
    )

    num_ppi_genes = len(
        dataset.ppi_gene_cols
    )

    # =====================================================
    # Safety checks
    # =====================================================

    if (
        ppi_expression_dim
        != num_ppi_genes
    ):

        raise ValueError(
            "PPI expression dimension does "
            "not match PPI graph gene count. "
            f"Expression={ppi_expression_dim}, "
            f"Graph={num_ppi_genes}"
        )

    if (
        num_ppi_genes
        != len(ppi["genes"])
    ):

        raise ValueError(
            "Dataset PPI gene count does "
            "not match saved PPI graph."
        )

    print(
        "\n======================================"
    )

    print(
        "PHAROS V2B DUAL-HYBRID"
    )

    print(
        "======================================"
    )

    print(
        f"Dataset rows: "
        f"{len(dataset)}"
    )

    print(
        f"Drug FP: "
        f"{drug_fp_dim}"
    )

    print(
        f"Molecular node features: "
        f"{node_dim}"
    )

    print(
        f"Molecular edge features: "
        f"{edge_dim}"
    )

    print(
        f"Full expression: "
        f"{cell_dim}"
    )

    print(
        f"PPI genes: "
        f"{num_ppi_genes}"
    )

    print(
        f"Resistance genes: "
        f"{resistance_dim}"
    )

    # =====================================================
    # Train / validation split
    # =====================================================

    train_size = int(
        0.8
        * len(dataset)
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

    print(
        f"\nTrain rows: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation rows: "
        f"{len(val_dataset)}"
    )

    # =====================================================
    # Data loaders
    # =====================================================

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

    # =====================================================
    # Model
    # =====================================================

    set_seed(seed)

    model = (
        PharosRXDualHybridModel(

            drug_fp_dim=drug_fp_dim,

            cell_dim=cell_dim,

            resistance_dim=(
                resistance_dim
            ),

            ppi_edge_index=ppi[
                "edge_index"
            ],

            ppi_edge_weight=ppi[
                "edge_weight"
            ],

            num_ppi_genes=(
                num_ppi_genes
            ),

            node_dim=node_dim,

            edge_dim=edge_dim,

            hidden_dim=512,

            drug_gnn_hidden_dim=128,

            ppi_hidden_dim=64,

            drug_out_dim=256,

            cell_out_dim=256,

            resistance_out_dim=64,

            dropout=0.2,
        )
        .to(device)
    )

    # =====================================================
    # Parameter count
    # =====================================================

    total_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"\nTotal parameters: "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    # =====================================================
    # Loss / optimiser
    # =====================================================

    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),

        lr=1e-3,

        weight_decay=1e-5,
    )

    # =====================================================
    # Learning-rate scheduler
    # =====================================================

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
    # Training settings
    # =====================================================

    epochs = 80

    early_stopping_patience = 10

    start_epoch = 1

    best_epoch = 0

    best_val_loss = float(
        "inf"
    )

    best_metrics = None

    patience_counter = 0

    history = []

    # =====================================================
    # Output files
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
            f"pharos_rx_dual_hybrid_50k_"
            f"seed_{seed}_best_model.pt"
        )
    )

    last_checkpoint_path = (
        output_dir
        / (
            f"pharos_rx_dual_hybrid_50k_"
            f"seed_{seed}_last_checkpoint.pt"
        )
    )

    history_path = (
        output_dir
        / (
            f"pharos_rx_dual_hybrid_50k_"
            f"seed_{seed}_training_history.csv"
        )
    )

    best_metrics_path = (
        output_dir
        / (
            f"pharos_rx_dual_hybrid_50k_"
            f"seed_{seed}_best_metrics.csv"
        )
    )

    # =====================================================
    # Resume training
    # =====================================================

    if resume:

        if not (
            last_checkpoint_path.exists()
        ):

            raise FileNotFoundError(
                "No recovery checkpoint "
                "found:\n"
                f"{last_checkpoint_path}"
            )

        print(
            "\nLoading recovery checkpoint..."
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
            checkpoint[
                "epoch"
            ]
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

        patience_counter = (
            checkpoint.get(
                "patience_counter",
                0,
            )
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
            f"Resuming from epoch "
            f"{start_epoch}"
        )

        print(
            f"Current best epoch: "
            f"{best_epoch}"
        )

        print(
            f"Current best val loss: "
            f"{best_val_loss:.6f}"
        )

    # =====================================================
    # Training loop
    # =====================================================

    print(
        "\n======================================"
    )

    print(
        "STARTING V2B TRAINING"
    )

    print(
        "======================================\n"
    )

    for epoch in range(
        start_epoch,
        epochs + 1,
    ):

        # =================================================
        # Train
        # =================================================

        train_loss = train_one_epoch(
            model=model,

            loader=train_loader,

            optimizer=optimizer,

            loss_fn=loss_fn,

            device=device,
        )

        # =================================================
        # Validate
        # =================================================

        val_metrics = evaluate(
            model=model,

            loader=val_loader,

            loss_fn=loss_fn,

            device=device,
        )

        # =================================================
        # Scheduler
        # =================================================

        scheduler.step(
            val_metrics[
                "loss"
            ]
        )

        current_lr = (
            optimizer
            .param_groups[0][
                "lr"
            ]
        )

        # =================================================
        # History
        # =================================================

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

        # =================================================
        # Console output
        # =================================================

        print(
            f"Epoch {epoch:02d} | "

            f"Train: "
            f"{train_loss:.4f} | "

            f"Val: "
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

            f"DrugGNN: "
            f"{val_metrics['mean_gnn_drug_gate']:.3f} | "

            f"Morgan: "
            f"{val_metrics['mean_morgan_gate']:.3f} | "

            f"PPI: "
            f"{val_metrics['mean_ppi_cell_gate']:.3f} | "

            f"Expr: "
            f"{val_metrics['mean_expression_gate']:.3f} | "

            f"LR: "
            f"{current_lr:.2e}"
        )

        # =================================================
        # New best model
        # =================================================

        if (
            val_metrics[
                "loss"
            ]
            < best_val_loss
        ):

            best_val_loss = (
                val_metrics[
                    "loss"
                ]
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

            # ---------------------------------------------
            # Copy weights to CPU
            # ---------------------------------------------

            best_model_state = (
                get_cpu_state_dict(
                    model
                )
            )

            # ---------------------------------------------
            # Save best checkpoint immediately
            # ---------------------------------------------

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
                        train_generator
                        .get_state(),

                    "seed":
                        seed,
                },

                best_checkpoint_path,
            )

            # ---------------------------------------------
            # Save best metrics
            # ---------------------------------------------

            pd.DataFrame(
                [
                    best_metrics
                ]
            ).to_csv(
                best_metrics_path,
                index=False,
            )

            print(
                "🔥 Saved new best "
                "V2B checkpoint"
            )

            del best_model_state

        else:

            patience_counter += 1

        # =================================================
        # Recovery checkpoint every epoch
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
                    train_generator
                    .get_state(),

                "seed":
                    seed,
            },

            last_checkpoint_path,
        )

        del last_model_state

        # =================================================
        # Save history every epoch
        # =================================================

        pd.DataFrame(
            history
        ).to_csv(
            history_path,
            index=False,
        )

        # =================================================
        # Memory cleanup
        # =================================================

        if device.type == "mps":

            torch.mps.empty_cache()

        gc.collect()

        # =================================================
        # Early stopping
        # =================================================

        if (
            patience_counter
            >= early_stopping_patience
        ):

            print(
                "\nEarly stopping triggered."
            )

            print(
                f"Stopped at epoch: "
                f"{epoch}"
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
        "PHAROS V2B DUAL-HYBRID COMPLETE"
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
        "\n======================================"
    )

    print(
        "CURRENT V2A TARGET"
    )

    print(
        "======================================"
    )

    print(
        "RMSE:      0.606124"
    )

    print(
        "MAE:       0.413958"
    )

    print(
        "R2:        0.551296"
    )

    print(
        "Pearson:   0.742749"
    )

    print(
        "Spearman:  0.453181"
    )

    print(
        "\n======================================"
    )

    print(
        "OUTPUT FILES"
    )

    print(
        "======================================"
    )

    print(
        f"Best model:\n"
        f"{best_checkpoint_path}"
    )

    print(
        f"\nRecovery checkpoint:\n"
        f"{last_checkpoint_path}"
    )

    print(
        f"\nTraining history:\n"
        f"{history_path}"
    )

    print(
        f"\nBest metrics:\n"
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