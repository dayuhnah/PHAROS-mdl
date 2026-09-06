import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from src.data.v4.mocktail_v4_dataset import (
    PharosMocktailV4Dataset,
)

from src.models.pharos_combo_clamp_resistance_drmref import (
    PharosComboCLAMPResistanceDRMref,
)


EXPERIMENT = "v3_matched"


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
):

    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    )

    error = (
        y_pred
        - y_true
    )

    mse = float(
        np.mean(
            error ** 2
        )
    )

    rmse = float(
        np.sqrt(mse)
    )

    mae = float(
        np.mean(
            np.abs(error)
        )
    )

    ss_res = np.sum(
        (
            y_true
            - y_pred
        ) ** 2
    )

    ss_tot = np.sum(
        (
            y_true
            - y_true.mean()
        ) ** 2
    )

    r2 = float(
        1.0
        - ss_res / ss_tot
    )

    pearson = float(
        np.corrcoef(
            y_true,
            y_pred,
        )[0, 1]
    )

    true_rank = (
        pd.Series(
            y_true
        )
        .rank()
        .to_numpy()
    )

    pred_rank = (
        pd.Series(
            y_pred
        )
        .rank()
        .to_numpy()
    )

    spearman = float(
        np.corrcoef(
            true_rank,
            pred_rank,
        )[0, 1]
    )

    return {
        "mse":
            mse,

        "rmse":
            rmse,

        "mae":
            mae,

        "r2":
            r2,

        "pearson":
            pearson,

        "spearman":
            spearman,
    }


# ============================================================
# FORWARD
# ============================================================

def forward_batch(
    model,
    batch,
    device,
    return_details=False,
):

    def gpu(key):

        return batch[
            key
        ].to(
            device,
            non_blocking=True,
        )

    # --------------------------------------------------------
    # V3 only sees EXPRESSION.
    #
    # CNV / mutation / CRISPR are deliberately NOT supplied.
    # --------------------------------------------------------

    expression = gpu(
        "expression"
    )

    resistance_expr = gpu(
        "resistance_expr"
    )

    # Expression availability is represented by the same
    # availability flag used by Core29.
    expression_available = gpu(
        "core29_available"
    ).float()

    # --------------------------------------------------------
    # Important for matched V4 rows:
    #
    # Original V3 data effectively required expression.
    # V4 contains some cells without expression.
    #
    # We retain those SAME evaluation rows, but ensure V3
    # receives zero biological expression/resistance features
    # for unavailable cells rather than inventing information.
    # --------------------------------------------------------

    resistance_expr = (
        resistance_expr
        * expression_available
    )

    drm_a_available = (
        gpu(
            "drug_a_drmref_available"
        ).float()
        * expression_available
    )

    drm_b_available = (
        gpu(
            "drug_b_drmref_available"
        ).float()
        * expression_available
    )

    return model(

        gpu(
            "drug_a_fp"
        ),

        gpu(
            "drug_a_clamp"
        ),

        gpu(
            "drug_b_fp"
        ),

        gpu(
            "drug_b_clamp"
        ),

        expression,

        resistance_expr,

        gpu(
            "drug_a_drmref_expr"
        ),

        gpu(
            "drug_b_drmref_expr"
        ),

        drm_a_available,

        drm_b_available,

        return_details=
            return_details,
    )


# ============================================================
# TRAIN ONE EPOCH
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
    total_rows = 0

    for batch in loader:

        optimizer.zero_grad(
            set_to_none=True
        )

        prediction = forward_batch(
            model,
            batch,
            device,
        )

        target = (
            batch[
                "label"
            ]
            .to(
                device,
                non_blocking=True,
            )
            .float()
        )

        loss = loss_fn(
            prediction,
            target,
        )

        loss.backward()

        optimizer.step()

        batch_rows = (
            target.numel()
        )

        total_loss += (
            loss.item()
            * batch_rows
        )

        total_rows += (
            batch_rows
        )

    return (
        total_loss
        / total_rows
    )


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    loss_fn,
    device,
):

    model.eval()

    y_true = []
    y_pred = []

    total_loss = 0.0
    total_rows = 0

    clamp_gate_sum = 0.0
    clamp_gate_count = 0

    drm_alpha_sum = 0.0
    drm_alpha_count = 0

    drm_available_rows = 0

    for batch in loader:

        details = forward_batch(
            model,
            batch,
            device,
            return_details=True,
        )

        prediction = details[
            "prediction"
        ]

        target = (
            batch[
                "label"
            ]
            .to(
                device,
                non_blocking=True,
            )
            .float()
        )

        loss = loss_fn(
            prediction,
            target,
        )

        batch_rows = (
            target.numel()
        )

        total_loss += (
            loss.item()
            * batch_rows
        )

        total_rows += (
            batch_rows
        )

        y_true.extend(
            target
            .view(-1)
            .cpu()
            .numpy()
            .tolist()
        )

        y_pred.extend(
            prediction
            .view(-1)
            .cpu()
            .numpy()
            .tolist()
        )

        # ====================================================
        # Morgan / CLAMP gate
        # ====================================================

        gate_a = details[
            "drug_a_clamp_gate"
        ]

        gate_b = details[
            "drug_b_clamp_gate"
        ]

        clamp_gate_sum += (
            gate_a.sum().item()
            +
            gate_b.sum().item()
        )

        clamp_gate_count += (
            gate_a.numel()
            +
            gate_b.numel()
        )

        # ====================================================
        # DRMref alpha only when evidence exists
        # ====================================================

        alpha = details[
            "drmref_alpha"
        ]

        available = (
            details[
                "drmref_pair_available"
            ] > 0
        )

        if available.any():

            selected = alpha[
                available
            ]

            drm_alpha_sum += (
                selected.sum().item()
            )

            drm_alpha_count += (
                selected.numel()
            )

            drm_available_rows += (
                available.sum().item()
            )

    result = calculate_metrics(
        y_true,
        y_pred,
    )

    result[
        "loss"
    ] = float(
        total_loss
        / total_rows
    )

    mean_clamp = (
        clamp_gate_sum
        / max(
            clamp_gate_count,
            1,
        )
    )

    result[
        "mean_clamp_gate"
    ] = float(
        mean_clamp
    )

    result[
        "mean_morgan_gate"
    ] = float(
        1.0
        - mean_clamp
    )

    result[
        "mean_drmref_alpha_when_available"
    ] = float(
        drm_alpha_sum
        / max(
            drm_alpha_count,
            1,
        )
    )

    result[
        "drmref_available_fraction"
    ] = float(
        drm_available_rows
        / max(
            total_rows,
            1,
        )
    )

    return (
        result,
        np.asarray(
            y_true
        ),
        np.asarray(
            y_pred
        ),
    )


# ============================================================
# MAIN
# ============================================================

def main(
    seed,
    device,
    epochs,
):

    set_seed(
        seed
    )

    dev = torch.device(
        device
    )

    # ========================================================
    # EXACT V4 DATA / SPLITS
    # ========================================================

    print(
        "Loading matched V4 datasets...",
        flush=True,
    )

    train_dataset = (
        PharosMocktailV4Dataset(
            split="train",
            seed=seed,
        )
    )

    val_dataset = (
        PharosMocktailV4Dataset(
            split="val",
            seed=seed,
        )
    )

    generator = (
        torch.Generator()
        .manual_seed(
            seed
        )
    )

    # ========================================================
    # ORIGINAL V3 BATCH SIZE = 64
    # ========================================================

    train_loader = DataLoader(

        train_dataset,

        batch_size=
            64,

        shuffle=
            True,

        generator=
            generator,

        num_workers=
            4,

        pin_memory=
            True,

        persistent_workers=
            True,

        drop_last=
            False,
    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=
            64,

        shuffle=
            False,

        num_workers=
            4,

        pin_memory=
            True,

        persistent_workers=
            True,

        drop_last=
            False,
    )

    # ========================================================
    # ORIGINAL V3 ARCHITECTURE
    # ========================================================

    set_seed(
        seed
    )

    model = (
        PharosComboCLAMPResistanceDRMref(

            fingerprint_dim=
                2048,

            clamp_dim=
                768,

            cell_dim=
                train_dataset.expression_dim,

            resistance_dim=
                29,

            drmref_dim=
                6481,

            drug_out_dim=
                256,

            pair_out_dim=
                256,

            cell_out_dim=
                256,

            resistance_out_dim=
                64,

            drmref_out_dim=
                64,

            drmref_hidden_dim=
                256,

            drmref_per_drug_dim=
                64,

            hidden_dim=
                512,

            dropout=
                0.2,
        )
        .to(
            dev
        )
    )

    trainable_parameters = sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad
    )

    # ========================================================
    # ORIGINAL V3 OPTIMISATION
    # ========================================================

    loss_fn = (
        nn.MSELoss()
    )

    optimizer = (
        torch.optim.Adam(

            model.parameters(),

            lr=
                1e-3,

            weight_decay=
                1e-5,
        )
    )

    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(

            optimizer,

            mode=
                "min",

            factor=
                0.5,

            patience=
                5,

            min_lr=
                1e-5,
        )
    )

    early_stopping_patience = (
        10
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    output_dir = (

        Path(
            "outputs/v4/benchmarks/v3_matched"
        )

        / f"seed_{seed}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        output_dir
        / "best_model.pt"
    )

    metrics_path = (
        output_dir
        / "best_metrics.csv"
    )

    history_path = (
        output_dir
        / "training_history.csv"
    )

    prediction_path = (
        output_dir
        / "val_predictions.parquet"
    )

    config_path = (
        output_dir
        / "config.json"
    )

    # ========================================================
    # CONFIG
    # ========================================================

    config = {

        "experiment":
            EXPERIMENT,

        "seed":
            seed,

        "target":
            "ZIP synergy",

        "split":
            "PHAROS V4 exact held-out drug pair",

        "train_rows":
            len(train_dataset),

        "val_rows":
            len(val_dataset),

        "drug_encoder":
            "Morgan2048 + frozen CLAMP768",

        "cell_features":
            "expression only",

        "cell_expression_dim":
            train_dataset.expression_dim,

        "core29":
            True,

        "drmref":
            True,

        "drmref_dim":
            6481,

        "cnv":
            False,

        "mutation":
            False,

        "crispr":
            False,

        "batch_size":
            64,

        "optimizer":
            "Adam",

        "learning_rate":
            1e-3,

        "weight_decay":
            1e-5,

        "scheduler":
            "ReduceLROnPlateau",

        "scheduler_factor":
            0.5,

        "scheduler_patience":
            5,

        "scheduler_min_lr":
            1e-5,

        "early_stopping_patience":
            10,

        "epochs":
            epochs,

        "trainable_parameters":
            trainable_parameters,

        "architecture":
            "original PHAROS Combo DRMref Resistance V3",

        "matched_benchmark":
            True,
    }

    with open(
        config_path,
        "w",
    ) as f:

        json.dump(
            config,
            f,
            indent=2,
        )

    # ========================================================
    # INFO
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "PHAROS V3 — MATCHED V4 BENCHMARK"
    )

    print(
        "=" * 70
    )

    print(
        "Seed:",
        seed
    )

    print(
        "Device:",
        dev
    )

    print(
        "Train rows:",
        f"{len(train_dataset):,}"
    )

    print(
        "Val rows:",
        f"{len(val_dataset):,}"
    )

    print(
        "Expression dim:",
        train_dataset.expression_dim
    )

    print(
        "Trainable params:",
        f"{trainable_parameters:,}"
    )

    print(
        "Batch size: 64"
    )

    print(
        "LR: 1e-3"
    )

    print(
        "Core29: ON"
    )

    print(
        "DRMref: ON"
    )

    print(
        "Mutation/CNV/CRISPR: OFF"
    )

    # ========================================================
    # TRAIN LOOP
    # ========================================================

    best_loss = float(
        "inf"
    )

    best_epoch = 0
    patience_counter = 0

    history = []

    for epoch in range(
        1,
        epochs + 1,
    ):

        start = (
            time.time()
        )

        train_loss = train_one_epoch(

            model,
            train_loader,
            optimizer,
            loss_fn,
            dev,
        )

        (
            metrics,
            _,
            _,
        ) = evaluate(

            model,
            val_loader,
            loss_fn,
            dev,
        )

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

        seconds = (
            time.time()
            - start
        )

        history_row = {

            "experiment":
                EXPERIMENT,

            "seed":
                seed,

            "epoch":
                epoch,

            "train_loss":
                train_loss,

            "lr":
                current_lr,

            "seconds":
                seconds,

            **metrics,
        }

        history.append(
            history_row
        )

        pd.DataFrame(
            history
        ).to_csv(
            history_path,
            index=False,
        )

        print(
            f"Epoch {epoch:03d} "
            f"| Train MSE {train_loss:.4f} "
            f"| Val MSE {metrics['loss']:.4f} "
            f"| RMSE {metrics['rmse']:.4f} "
            f"| R2 {metrics['r2']:.4f} "
            f"| P {metrics['pearson']:.4f} "
            f"| S {metrics['spearman']:.4f} "
            f"| CLAMP {metrics['mean_clamp_gate']:.3f} "
            f"| DRM {metrics['mean_drmref_alpha_when_available']:.3f} "
            f"| {seconds / 60:.1f} min",
            flush=True,
        )

        # ====================================================
        # BEST MODEL
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

            patience_counter = (
                0
            )

            torch.save(
                {

                    "model_state_dict":
                        {
                            key:
                                value.detach().cpu()

                            for key, value

                            in model
                            .state_dict()
                            .items()
                        },

                    "seed":
                        seed,

                    "epoch":
                        epoch,

                    "metrics":
                        metrics,

                    "config":
                        config,
                },

                checkpoint_path,
            )

            best_row = {

                "experiment":
                    EXPERIMENT,

                "seed":
                    seed,

                "best_epoch":
                    epoch,

                **metrics,
            }

            pd.DataFrame(
                [
                    best_row
                ]
            ).to_csv(
                metrics_path,
                index=False,
            )

            print(
                "  ✅ New best",
                flush=True,
            )

        else:

            patience_counter += (
                1
            )

        # ====================================================
        # EARLY STOPPING
        # ====================================================

        if (
            patience_counter
            >= early_stopping_patience
        ):

            print(
                f"Early stopping at "
                f"epoch {epoch}",
                flush=True,
            )

            break

    # ========================================================
    # LOAD BEST
    # ========================================================

    checkpoint = torch.load(

        checkpoint_path,

        map_location=
            dev,

        weights_only=
            False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    (
        final_metrics,
        y_true,
        y_pred,
    ) = evaluate(

        model,
        val_loader,
        loss_fn,
        dev,
    )

    # ========================================================
    # ROW-LEVEL PREDICTIONS
    # ========================================================

    rows = (
        val_dataset
        .df
        .copy()
        .reset_index(
            drop=True
        )
    )

    if (
        len(rows)
        != len(y_pred)
    ):

        raise RuntimeError(

            "Prediction count mismatch: "

            f"{len(rows)} rows vs "
            f"{len(y_pred)} predictions."
        )

    rows[
        "prediction"
    ] = y_pred

    rows[
        "error"
    ] = (
        y_pred
        - y_true
    )

    rows[
        "absolute_error"
    ] = np.abs(
        y_pred
        - y_true
    )

    rows[
        "squared_error"
    ] = (
        y_pred
        - y_true
    ) ** 2

    rows[
        "experiment"
    ] = EXPERIMENT

    rows[
        "seed"
    ] = seed

    rows.to_parquet(
        prediction_path,
        index=False,
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "MATCHED V3 COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "Best epoch:",
        best_epoch
    )

    print(
        "Best RMSE:",
        f"{final_metrics['rmse']:.6f}"
    )

    print(
        "MAE:",
        f"{final_metrics['mae']:.6f}"
    )

    print(
        "R2:",
        f"{final_metrics['r2']:.6f}"
    )

    print(
        "Pearson:",
        f"{final_metrics['pearson']:.6f}"
    )

    print(
        "Spearman:",
        f"{final_metrics['spearman']:.6f}"
    )

    print()
    print(
        "Saved:"
    )

    print(
        checkpoint_path
    )

    print(
        metrics_path
    )

    print(
        history_path
    )

    print(
        prediction_path
    )

    print(
        config_path
    )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
    )

    args = parser.parse_args()

    main(

        seed=
            args.seed,

        device=
            args.device,

        epochs=
            args.epochs,
    )
