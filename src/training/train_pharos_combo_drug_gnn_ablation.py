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
    Subset,
)

from torch_geometric.data import (
    Batch,
)

from src.data.combo_dataset import (
    PharosComboDataset,
)

from src.evaluation.metrics import (
    regression_metrics,
)

from src.models.pharos_combo_drug_gnn_ablation import (
    PharosComboDrugGNNAblation,
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# Collate
#
# Only data needed for this ablation:
#
# Drug A graph + Morgan
# Drug B graph + Morgan
# Full expression
# ZIP label
# ============================================================

def drug_gnn_collate(batch):

    drug_a_graph = Batch.from_data_list(
        [
            item["drug_a_graph"]
            for item in batch
        ]
    )

    drug_b_graph = Batch.from_data_list(
        [
            item["drug_b_graph"]
            for item in batch
        ]
    )

    return {

        "drug_a_graph":
            drug_a_graph,

        "drug_a_fp":
            torch.stack(
                [
                    item["drug_a_fp"]
                    for item in batch
                ]
            ),

        "drug_b_graph":
            drug_b_graph,

        "drug_b_fp":
            torch.stack(
                [
                    item["drug_b_fp"]
                    for item in batch
                ]
            ),

        "cell_expr":
            torch.stack(
                [
                    item["cell_expr"]
                    for item in batch
                ]
            ),

        "label":
            torch.stack(
                [
                    item["label"]
                    for item in batch
                ]
            ),
    }


# ============================================================
# Load SAME saved split as PHAROS-Combo
# ============================================================

def load_saved_split(
    dataset,
    seed,
):

    output_dir = Path(
        "outputs"
    )

    train_path = (
        output_dir
        / f"pharos_combo_seed_{seed}_train_pairs.csv"
    )

    val_path = (
        output_dir
        / f"pharos_combo_seed_{seed}_val_pairs.csv"
    )

    if not train_path.exists():

        raise FileNotFoundError(
            f"Missing train split: "
            f"{train_path}"
        )

    if not val_path.exists():

        raise FileNotFoundError(
            f"Missing validation split: "
            f"{val_path}"
        )

    train_pairs = pd.read_csv(
        train_path
    )

    val_pairs = pd.read_csv(
        val_path
    )

    train_pair_set = set(
        zip(
            train_pairs[
                "drug_a"
            ].astype(str),

            train_pairs[
                "drug_b"
            ].astype(str),
        )
    )

    val_pair_set = set(
        zip(
            val_pairs[
                "drug_a"
            ].astype(str),

            val_pairs[
                "drug_b"
            ].astype(str),
        )
    )

    overlap = (
        train_pair_set
        &
        val_pair_set
    )

    if overlap:

        raise RuntimeError(
            "Drug-pair leakage detected."
        )

    metadata = (
        dataset
        .get_metadata()
        .reset_index(
            drop=True
        )
    )

    row_pairs = list(
        zip(
            metadata[
                "drug_a"
            ].astype(str),

            metadata[
                "drug_b"
            ].astype(str),
        )
    )

    train_indices = [
        i
        for i, pair
        in enumerate(
            row_pairs
        )
        if pair
        in train_pair_set
    ]

    val_indices = [
        i
        for i, pair
        in enumerate(
            row_pairs
        )
        if pair
        in val_pair_set
    ]

    if (
        len(train_indices)
        +
        len(val_indices)
        != len(dataset)
    ):

        raise RuntimeError(
            "Some rows were not assigned "
            "to train or validation."
        )

    return (
        Subset(
            dataset,
            train_indices,
        ),

        Subset(
            dataset,
            val_indices,
        ),
    )


# ============================================================
# Train one epoch
# ============================================================

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

        drug_a_graph = (
            batch[
                "drug_a_graph"
            ].to(device)
        )

        drug_a_fp = (
            batch[
                "drug_a_fp"
            ].to(device)
        )

        drug_b_graph = (
            batch[
                "drug_b_graph"
            ].to(device)
        )

        drug_b_fp = (
            batch[
                "drug_b_fp"
            ].to(device)
        )

        cell_expr = (
            batch[
                "cell_expr"
            ].to(device)
        )

        labels = (
            batch[
                "label"
            ].to(device)
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        predictions = model(

            drug_a_graph=(
                drug_a_graph
            ),

            drug_a_fp=(
                drug_a_fp
            ),

            drug_b_graph=(
                drug_b_graph
            ),

            drug_b_fp=(
                drug_b_fp
            ),

            cell_expr=(
                cell_expr
            ),
        )

        loss = loss_fn(
            predictions,
            labels,
        )

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


# ============================================================
# Evaluate
# ============================================================

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

    drug_a_gates = []
    drug_b_gates = []

    for batch in loader:

        drug_a_graph = (
            batch[
                "drug_a_graph"
            ].to(device)
        )

        drug_a_fp = (
            batch[
                "drug_a_fp"
            ].to(device)
        )

        drug_b_graph = (
            batch[
                "drug_b_graph"
            ].to(device)
        )

        drug_b_fp = (
            batch[
                "drug_b_fp"
            ].to(device)
        )

        cell_expr = (
            batch[
                "cell_expr"
            ].to(device)
        )

        labels = (
            batch[
                "label"
            ].to(device)
        )

        output = model(

            drug_a_graph=(
                drug_a_graph
            ),

            drug_a_fp=(
                drug_a_fp
            ),

            drug_b_graph=(
                drug_b_graph
            ),

            drug_b_fp=(
                drug_b_fp
            ),

            cell_expr=(
                cell_expr
            ),

            return_details=True,
        )

        predictions = output[
            "prediction"
        ]

        loss = loss_fn(
            predictions,
            labels,
        )

        total_loss += (
            loss.item()
            * labels.size(0)
        )

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

        drug_a_gates.append(
            output[
                "drug_a_gate"
            ]
            .detach()
            .cpu()
        )

        drug_b_gates.append(
            output[
                "drug_b_gate"
            ]
            .detach()
            .cpu()
        )

    # ========================================================
    # Metrics
    # ========================================================

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

    # ========================================================
    # Drug gate diagnostics
    # ========================================================

    drug_a_gates = torch.cat(
        drug_a_gates,
        dim=0,
    )

    drug_b_gates = torch.cat(
        drug_b_gates,
        dim=0,
    )

    combined_gates = torch.cat(
        [
            drug_a_gates,
            drug_b_gates,
        ],
        dim=0,
    )

    metrics[
        "mean_drug_a_gnn_gate"
    ] = (
        drug_a_gates
        .mean()
        .item()
    )

    metrics[
        "mean_drug_b_gnn_gate"
    ] = (
        drug_b_gates
        .mean()
        .item()
    )

    metrics[
        "mean_gnn_drug_gate"
    ] = (
        combined_gates
        .mean()
        .item()
    )

    metrics[
        "mean_morgan_gate"
    ] = (
        1.0
        -
        combined_gates
    ).mean().item()

    return metrics


# ============================================================
# Main
# ============================================================

def main(
    seed=42,
):

    set_seed(
        seed
    )

    # ========================================================
    # Device
    # ========================================================

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
        f"Random seed: {seed}"
    )

    print(
        f"Using device: {device}"
    )

    # ========================================================
    # Dataset
    # ========================================================

    dataset = PharosComboDataset(
        max_rows=None,

        target="zip_score",

        seed=seed,
    )

    sample = dataset[
        0
    ]

    # ========================================================
    # Exact same held-out pair split
    # ========================================================

    (
        train_dataset,
        val_dataset,
    ) = load_saved_split(

        dataset,
        seed,
    )

    print(
        "\n======================================"
    )

    print(
        "PHAROS-COMBO DRUG-GNN ABLATION"
    )

    print(
        "======================================"
    )

    print(
        f"Train rows: "
        f"{len(train_dataset):,}"
    )

    print(
        f"Validation rows: "
        f"{len(val_dataset):,}"
    )

    print(
        "Molecular GNN: ON"
    )

    print(
        "Morgan: ON"
    )

    print(
        "PPI-GNN: OFF"
    )

    print(
        "Resistance FiLM: OFF"
    )

    # ========================================================
    # DataLoaders
    # ========================================================

    train_generator = (
        torch.Generator()
        .manual_seed(seed)
    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=64,

        shuffle=True,

        generator=(
            train_generator
        ),

        num_workers=0,

        collate_fn=(
            drug_gnn_collate
        ),
    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=64,

        shuffle=False,

        num_workers=0,

        collate_fn=(
            drug_gnn_collate
        ),
    )

    # ========================================================
    # Model
    # ========================================================

    set_seed(
        seed
    )

    model = PharosComboDrugGNNAblation(

        drug_fp_dim=(
            sample[
                "drug_a_fp"
            ].shape[0]
        ),

        cell_dim=(
            sample[
                "cell_expr"
            ].shape[0]
        ),

        node_dim=(
            sample[
                "drug_a_graph"
            ].x.shape[1]
        ),

        edge_dim=(
            sample[
                "drug_a_graph"
            ].edge_attr.shape[1]
        ),

        drug_gnn_hidden_dim=128,

        drug_out_dim=256,

        pair_out_dim=256,

        cell_out_dim=256,

        hidden_dim=512,

        dropout=0.2,
    ).to(
        device
    )

    # ========================================================
    # Parameter count
    # ========================================================

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"\nTrainable parameters: "
        f"{total_parameters:,}"
    )

    # ========================================================
    # Optimisation
    # ========================================================

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

    early_stopping_patience = 10

    best_loss = float(
        "inf"
    )

    best_epoch = 0

    best_metrics = None

    patience_counter = 0

    history = []

    # ========================================================
    # Outputs
    # ========================================================

    output_dir = Path(
        "outputs"
    )

    output_dir.mkdir(
        exist_ok=True
    )

    best_model_path = (
        output_dir
        /
        (
            f"pharos_combo_drug_gnn_"
            f"seed_{seed}_best_model.pt"
        )
    )

    history_path = (
        output_dir
        /
        (
            f"pharos_combo_drug_gnn_"
            f"seed_{seed}_training_history.csv"
        )
    )

    metrics_path = (
        output_dir
        /
        (
            f"pharos_combo_drug_gnn_"
            f"seed_{seed}_best_metrics.csv"
        )
    )

    # ========================================================
    # Training
    # ========================================================

    print(
        "\n======================================"
    )

    print(
        "STARTING DRUG-GNN ABLATION"
    )

    print(
        "======================================\n"
    )

    for epoch in range(
        1,
        epochs + 1,
    ):

        train_loss = train_one_epoch(

            model=model,

            loader=train_loader,

            optimizer=optimizer,

            loss_fn=loss_fn,

            device=device,
        )

        metrics = evaluate(

            model=model,

            loader=val_loader,

            loss_fn=loss_fn,

            device=device,
        )

        # ====================================================
        # LR scheduler
        # ====================================================

        scheduler.step(
            metrics[
                "loss"
            ]
        )

        current_lr = (
            optimizer
            .param_groups[0][
                "lr"
            ]
        )

        # ====================================================
        # History
        # ====================================================

        history.append(
            {

                "epoch":
                    epoch,

                "seed":
                    seed,

                "train_loss":
                    train_loss,

                "lr":
                    current_lr,

                **metrics,
            }
        )

        # ====================================================
        # Print
        # ====================================================

        print(

            f"Epoch {epoch:02d} | "

            f"Train: "
            f"{train_loss:.4f} | "

            f"Val: "
            f"{metrics['loss']:.4f} | "

            f"RMSE: "
            f"{metrics['rmse']:.4f} | "

            f"MAE: "
            f"{metrics['mae']:.4f} | "

            f"R2: "
            f"{metrics['r2']:.4f} | "

            f"Pearson: "
            f"{metrics['pearson']:.4f} | "

            f"Spearman: "
            f"{metrics['spearman']:.4f} | "

            f"GNN: "
            f"{metrics['mean_gnn_drug_gate']:.3f} | "

            f"Morgan: "
            f"{metrics['mean_morgan_gate']:.3f} | "

            f"LR: "
            f"{current_lr:.2e}"
        )

        # ====================================================
        # New best
        # ====================================================

        if (
            metrics[
                "loss"
            ]
            < best_loss
        ):

            best_loss = (
                metrics[
                    "loss"
                ]
            )

            best_epoch = (
                epoch
            )

            best_metrics = dict(
                metrics
            )

            best_metrics[
                "train_loss"
            ] = (
                train_loss
            )

            best_metrics[
                "best_epoch"
            ] = (
                best_epoch
            )

            best_metrics[
                "seed"
            ] = (
                seed
            )

            best_metrics[
                "lr"
            ] = (
                current_lr
            )

            best_metrics[
                "model"
            ] = (
                "hybrid_drug_gnn_expression"
            )

            best_metrics[
                "split_type"
            ] = (
                "held_out_drug_pairs"
            )

            patience_counter = 0

            # ------------------------------------------------
            # Save best model
            # ------------------------------------------------

            torch.save(
                {
                    key:
                        value
                        .detach()
                        .cpu()
                    for key, value
                    in model
                    .state_dict()
                    .items()
                },

                best_model_path,
            )

            # ------------------------------------------------
            # Save metrics
            # ------------------------------------------------

            pd.DataFrame(
                [
                    best_metrics
                ]
            ).to_csv(
                metrics_path,
                index=False,
            )

            print(
                "🔥 Saved new best "
                "Drug-GNN ablation"
            )

        else:

            patience_counter += 1

        # ====================================================
        # Save history
        # ====================================================

        pd.DataFrame(
            history
        ).to_csv(
            history_path,
            index=False,
        )

        # ====================================================
        # Cleanup
        # ====================================================

        if (
            device.type
            == "mps"
        ):

            torch.mps.empty_cache()

        gc.collect()

        # ====================================================
        # Early stopping
        # ====================================================

        if (
            patience_counter
            >= early_stopping_patience
        ):

            print(
                "\nEarly stopping."
            )

            break

    # ========================================================
    # Complete
    # ========================================================

    print(
        "\n======================================"
    )

    print(
        "DRUG-GNN ABLATION COMPLETE"
    )

    print(
        "======================================"
    )

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        "\nBest metrics:"
    )

    print(
        best_metrics
    )

    print(
        "\nBest model:"
    )

    print(
        best_model_path
    )

    print(
        "\nBest metrics file:"
    )

    print(
        metrics_path
    )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    main(
        seed=args.seed
    )