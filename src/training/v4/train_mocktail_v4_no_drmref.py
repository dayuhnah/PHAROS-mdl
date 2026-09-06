from pathlib import Path
import argparse
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.v4.mocktail_v4_dataset import (
    PharosMocktailV4Dataset,
)
from src.models.v4.pharos_mocktail_v4 import (
    PharosMocktailV4,
)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    error = y_pred - y_true

    rmse = np.sqrt(np.mean(error ** 2))
    mae = np.mean(np.abs(error))

    ss_res = np.sum(
        (y_true - y_pred) ** 2
    )

    ss_tot = np.sum(
        (y_true - y_true.mean()) ** 2
    )

    r2 = 1.0 - ss_res / ss_tot

    pearson = np.corrcoef(
        y_true,
        y_pred,
    )[0, 1]

    true_rank = pd.Series(
        y_true
    ).rank().to_numpy()

    pred_rank = pd.Series(
        y_pred
    ).rank().to_numpy()

    spearman = np.corrcoef(
        true_rank,
        pred_rank,
    )[0, 1]

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "pearson": float(pearson),
        "spearman": float(spearman),
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
        return batch[key].to(
            device,
            non_blocking=True,
        )

    # --------------------------------------------------------
    # MUTATION-ONLY OMICS MASK
    #
    # Original order:
    # expression, cnv, mutation, crispr
    # --------------------------------------------------------

    original_mask = gpu(
        "omics_mask"
    )

    ablation_mask = original_mask.clone()

    return model(

        gpu("drug_a_fp"),
        gpu("drug_a_clamp"),

        gpu("drug_b_fp"),
        gpu("drug_b_clamp"),

        # We still pass tensors because the model API
        # expects them. The mask prevents them from
        # contributing to cell fusion.
        gpu("expression"),
        gpu("cnv"),
        gpu("mutation"),
        gpu("crispr"),

        ablation_mask,

        gpu("resistance_expr"),
        gpu("core29_available"),

        gpu("drug_a_drmref_expr"),
        gpu("drug_b_drmref_expr"),

        torch.zeros_like(
            gpu("drug_a_drmref_available")
        ),
        torch.zeros_like(
            gpu("drug_b_drmref_available")
        ),

        return_details=return_details,
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
):

    model.eval()

    preds = []
    targets = []
    weights = []

    for batch in loader:

        out = forward_batch(
            model,
            batch,
            device,
            return_details=True,
        )

        preds.extend(
            out["prediction"]
            .squeeze(-1)
            .cpu()
            .numpy()
            .tolist()
        )

        targets.extend(
            batch["label"]
            .squeeze(-1)
            .numpy()
            .tolist()
        )

        weights.append(
            out["modality_weights"]
            .cpu()
            .numpy()
        )

    result = calculate_metrics(
        targets,
        preds,
    )

    w = np.concatenate(
        weights,
        axis=0,
    )

    result.update({
        "expression_weight":
            float(w[:, 0].mean()),

        "cnv_weight":
            float(w[:, 1].mean()),

        "mutation_weight":
            float(w[:, 2].mean()),

        "crispr_weight":
            float(w[:, 3].mean()),

        "drmref_alpha":
            float(
                torch.sigmoid(
                    model.drmref_alpha_logit
                ).item()
            ),
    })

    return (
        result,
        np.asarray(targets),
        np.asarray(preds),
    )


# ============================================================
# TRAINING
# ============================================================

def main(
    seed,
    epochs,
    batch_size,
    lr,
    patience,
):

    experiment = "no_drmref"

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    OUT = (
        Path("outputs/v4/ablations")
        / experiment
        / f"seed_{seed}"
    )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        OUT / "best_model.pt"
    )

    metrics_path = (
        OUT / "best_metrics.csv"
    )

    history_path = (
        OUT / "training_history.csv"
    )

    predictions_path = (
        OUT / "val_predictions.parquet"
    )

    config_path = (
        OUT / "config.json"
    )

    # ========================================================
    # DATA
    # ========================================================

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = PharosMocktailV4(
        expression_dim=
            train_dataset.expression_dim,

        cnv_dim=
            train_dataset.cnv_dim,

        mutation_dim=
            train_dataset.mutation_dim,

        crispr_dim=
            train_dataset.crispr_dim,

        dropout=0.2,
    ).to(device)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    # ========================================================
    # CONFIG
    # ========================================================

    config = {
        "experiment": experiment,
        "seed": seed,

        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "patience": patience,

        "train_rows":
            len(train_dataset),

        "val_rows":
            len(val_dataset),

        "parameters":
            parameter_count,

        "active_omics": [
            "expression",
            "cnv",
            "mutation",
            "crispr"
        ],

        "core29": True,
        "drmref": False,
        "drug_encoder":
            "Morgan + frozen CLAMP",

        "split":
            "held-out exact drug pair",
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

    print("=" * 70)
    print("PHAROS MOCKTAIL V4 — NO DRMref")
    print("=" * 70)

    print("Seed:", seed)
    print("Device:", device)
    print(
        "Parameters:",
        f"{parameter_count:,}"
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    criterion = nn.MSELoss()

    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
        )
    )

    # ========================================================
    # TRAIN LOOP
    # ========================================================

    best_rmse = float("inf")
    best_epoch = 0
    bad_epochs = 0

    history = []

    for epoch in range(
        1,
        epochs + 1,
    ):

        start = time.time()

        model.train()

        sq_error = 0.0
        count = 0

        for batch in train_loader:

            optimizer.zero_grad(
                set_to_none=True
            )

            pred = forward_batch(
                model,
                batch,
                device,
            )

            target = (
                batch["label"]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            loss = criterion(
                pred,
                target,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
            )

            optimizer.step()

            sq_error += (
                (pred.detach() - target)
                .pow(2)
                .sum()
                .item()
            )

            count += target.numel()

        train_rmse = np.sqrt(
            sq_error / count
        )

        val, _, _ = evaluate(
            model,
            val_loader,
            device,
        )

        scheduler.step(
            val["rmse"]
        )

        lr_now = (
            optimizer
            .param_groups[0]["lr"]
        )

        elapsed = (
            time.time() - start
        )

        row = {
            "experiment":
                experiment,

            "seed":
                seed,

            "epoch":
                epoch,

            "train_rmse":
                train_rmse,

            **val,

            "lr":
                lr_now,

            "seconds":
                elapsed,
        }

        history.append(row)

        pd.DataFrame(
            history
        ).to_csv(
            history_path,
            index=False,
        )

        print(
            f"Epoch {epoch:03d} "
            f"| Train {train_rmse:.4f} "
            f"| Val {val['rmse']:.4f} "
            f"| R2 {val['r2']:.4f} "
            f"| P {val['pearson']:.4f} "
            f"| S {val['spearman']:.4f}"
        )

        # ====================================================
        # BEST CHECKPOINT
        # ====================================================

        if val["rmse"] < best_rmse:

            best_rmse = val["rmse"]
            best_epoch = epoch
            bad_epochs = 0

            torch.save(
                {
                    "experiment":
                        experiment,

                    "seed":
                        seed,

                    "epoch":
                        epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "metrics":
                        val,

                    "config":
                        config,
                },
                checkpoint_path,
            )

            best_row = {
                "experiment":
                    experiment,

                "seed":
                    seed,

                "best_epoch":
                    epoch,

                **val,
            }

            pd.DataFrame(
                [best_row]
            ).to_csv(
                metrics_path,
                index=False,
            )

            print(
                "  ✅ New best"
            )

        else:

            bad_epochs += 1

        if bad_epochs >= patience:

            print(
                f"Early stopping "
                f"at epoch {epoch}"
            )

            break

    # ========================================================
    # RELOAD BEST MODEL
    # ========================================================

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # ========================================================
    # FINAL PREDICTIONS
    # ========================================================

    final_metrics, y_true, y_pred = (
        evaluate(
            model,
            val_loader,
            device,
        )
    )

    # Preserve identifiers for paired comparisons later
    pred_df = (
        val_dataset.df[
            [
                "drug_a",
                "drug_b",
                "drug_a_key",
                "drug_b_key",
                "depmap_id",
                "zip_score",
            ]
        ]
        .copy()
        .reset_index(drop=True)
    )

    pred_df[
        "prediction"
    ] = y_pred

    pred_df[
        "error"
    ] = (
        y_pred - y_true
    )

    pred_df[
        "absolute_error"
    ] = np.abs(
        y_pred - y_true
    )

    pred_df[
        "squared_error"
    ] = (
        y_pred - y_true
    ) ** 2

    pred_df[
        "experiment"
    ] = experiment

    pred_df[
        "seed"
    ] = seed

    pred_df.to_parquet(
        predictions_path,
        index=False,
    )

    print()
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)

    print(
        "Best epoch:",
        best_epoch
    )

    print(
        "Best RMSE:",
        f"{best_rmse:.6f}"
    )

    print()
    print("Saved:")
    print(metrics_path)
    print(history_path)
    print(predictions_path)
    print(config_path)
    print(checkpoint_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=12,
    )

    args = parser.parse_args()

    main(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )
