import argparse
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

from src.data.combo_dataset import (
    PharosComboDataset,
)

from src.evaluation.metrics import (
    regression_metrics,
)

from src.models.pharos_combo_baseline import (
    PharosComboBaseline,
)


# ============================================================
# Seed
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# Lightweight baseline collate
#
# We deliberately ignore molecular graphs, PPI and resistance.
# ============================================================

def baseline_collate(batch):

    return {

        "drug_a_fp":
            torch.stack(
                [
                    item["drug_a_fp"]
                    for item in batch
                ]
            ),

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
# Load EXACT saved PHAROS-Combo pair split
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
        /
        f"pharos_combo_seed_{seed}_train_pairs.csv"
    )

    val_path = (
        output_dir
        /
        f"pharos_combo_seed_{seed}_val_pairs.csv"
    )

    if not train_path.exists():

        raise FileNotFoundError(
            f"Missing train split: {train_path}"
        )

    if not val_path.exists():

        raise FileNotFoundError(
            f"Missing validation split: {val_path}"
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
            "Train/validation pair leakage detected."
        )

    metadata = (
        dataset
        .get_metadata()
        .reset_index(drop=True)
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
        in enumerate(row_pairs)
        if pair in train_pair_set
    ]

    val_indices = [
        i
        for i, pair
        in enumerate(row_pairs)
        if pair in val_pair_set
    ]

    if (
        len(train_indices)
        +
        len(val_indices)
        != len(dataset)
    ):

        raise RuntimeError(
            "Some rows were not assigned "
            "to the saved pair split."
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
# Train
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

        drug_a_fp = (
            batch[
                "drug_a_fp"
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
            drug_a_fp=drug_a_fp,
            drug_b_fp=drug_b_fp,
            cell_expr=cell_expr,
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
        /
        len(loader.dataset)
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

    predictions_all = []
    labels_all = []

    for batch in loader:

        drug_a_fp = (
            batch[
                "drug_a_fp"
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

        predictions = model(
            drug_a_fp=drug_a_fp,
            drug_b_fp=drug_b_fp,
            cell_expr=cell_expr,
        )

        loss = loss_fn(
            predictions,
            labels,
        )

        total_loss += (
            loss.item()
            * labels.size(0)
        )

        predictions_all.extend(
            predictions
            .cpu()
            .numpy()
            .reshape(-1)
        )

        labels_all.extend(
            labels
            .cpu()
            .numpy()
            .reshape(-1)
        )

    metrics = regression_metrics(
        labels_all,
        predictions_all,
    )

    metrics["loss"] = (
        total_loss
        /
        len(loader.dataset)
    )

    return metrics


# ============================================================
# Mean-prediction baseline
# ============================================================

def calculate_mean_baseline(
    dataset,
    train_dataset,
    val_dataset,
):

    train_labels = np.array(
        [
            dataset.labels[i]
            for i
            in train_dataset.indices
        ],
        dtype=np.float32,
    )

    val_labels = np.array(
        [
            dataset.labels[i]
            for i
            in val_dataset.indices
        ],
        dtype=np.float32,
    )

    train_mean = float(
        train_labels.mean()
    )

    predictions = np.full(
        shape=len(val_labels),
        fill_value=train_mean,
        dtype=np.float32,
    )

    metrics = regression_metrics(
        val_labels,
        predictions,
    )

    return (
        train_mean,
        metrics,
    )


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

    sample = dataset[0]

    # ========================================================
    # SAME pair split as full PHAROS-Combo
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
        "PHAROS-COMBO SIMPLE BASELINE"
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

    # ========================================================
    # Mean baseline
    # ========================================================

    (
        train_mean,
        mean_metrics,
    ) = calculate_mean_baseline(

        dataset,
        train_dataset,
        val_dataset,
    )

    print(
        "\n======================================"
    )

    print(
        "MEAN ZIP BASELINE"
    )

    print(
        "======================================"
    )

    print(
        f"Training ZIP mean: "
        f"{train_mean:.4f}"
    )

    print(
        f"RMSE: "
        f"{mean_metrics['rmse']:.4f}"
    )

    print(
        f"MAE: "
        f"{mean_metrics['mae']:.4f}"
    )

    print(
        f"R2: "
        f"{mean_metrics['r2']:.4f}"
    )

    # ========================================================
    # Loaders
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
            baseline_collate
        ),
    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=64,

        shuffle=False,

        num_workers=0,

        collate_fn=(
            baseline_collate
        ),
    )

    # ========================================================
    # Model
    # ========================================================

    set_seed(
        seed
    )

    model = PharosComboBaseline(

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

        drug_out_dim=256,

        cell_out_dim=256,

        pair_out_dim=256,

        hidden_dim=512,

        dropout=0.2,
    ).to(
        device
    )

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"\nTrainable parameters: "
        f"{total_parameters:,}"
    )

    # ========================================================
    # Training configuration
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
    # Output
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
            f"pharos_combo_baseline_"
            f"seed_{seed}_best_model.pt"
        )
    )

    history_path = (
        output_dir
        /
        (
            f"pharos_combo_baseline_"
            f"seed_{seed}_training_history.csv"
        )
    )

    metrics_path = (
        output_dir
        /
        (
            f"pharos_combo_baseline_"
            f"seed_{seed}_best_metrics.csv"
        )
    )

    # ========================================================
    # Train
    # ========================================================

    print(
        "\n======================================"
    )

    print(
        "STARTING BASELINE TRAINING"
    )

    print(
        "======================================\n"
    )

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

        metrics = evaluate(

            model,
            val_loader,
            loss_fn,
            device,
        )

        scheduler.step(
            metrics[
                "loss"
            ]
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

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

            f"LR: "
            f"{current_lr:.2e}"
        )

        # ====================================================
        # Save best
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

            best_epoch = epoch

            best_metrics = dict(
                metrics
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

            best_metrics[
                "model"
            ] = (
                "simple_morgan_expression"
            )

            best_metrics[
                "split_type"
            ] = (
                "held_out_drug_pairs"
            )

            patience_counter = 0

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

            pd.DataFrame(
                [
                    best_metrics
                ]
            ).to_csv(
                metrics_path,
                index=False,
            )

            print(
                "🔥 Saved new best baseline"
            )

        else:

            patience_counter += 1

        pd.DataFrame(
            history
        ).to_csv(
            history_path,
            index=False,
        )

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
    # Results
    # ========================================================

    print(
        "\n======================================"
    )

    print(
        "BASELINE COMPLETE"
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
        "\n======================================"
    )

    print(
        "FULL PHAROS-COMBO SEED 42"
    )

    print(
        "======================================"
    )

    print(
        "RMSE:      5.4530"
    )

    print(
        "MAE:       3.5949"
    )

    print(
        "R2:        0.2038"
    )

    print(
        "Pearson:   0.4541"
    )

    print(
        "Spearman:  0.4492"
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