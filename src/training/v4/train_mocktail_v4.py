from pathlib import Path
import argparse
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

def metrics(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    error = y_pred - y_true

    rmse = np.sqrt(
        np.mean(error ** 2)
    )

    mae = np.mean(
        np.abs(error)
    )

    ss_res = np.sum(
        (y_true - y_pred) ** 2
    )

    ss_tot = np.sum(
        (y_true - y_true.mean()) ** 2
    )

    r2 = (
        1.0 - ss_res / ss_tot
        if ss_tot > 0
        else 0.0
    )

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

    def x(key):
        return batch[key].to(
            device,
            non_blocking=True,
        )

    return model(
        x("drug_a_fp"),
        x("drug_a_clamp"),

        x("drug_b_fp"),
        x("drug_b_clamp"),

        x("expression"),
        x("cnv"),
        x("mutation"),
        x("crispr"),

        x("omics_mask"),

        x("resistance_expr"),
        x("core29_available"),

        x("drug_a_drmref_expr"),
        x("drug_b_drmref_expr"),

        x("drug_a_drmref_available"),
        x("drug_b_drmref_available"),

        return_details=return_details,
    )


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
):

    model.eval()

    predictions = []
    targets = []

    modality_weights = []

    for batch in loader:

        out = forward_batch(
            model,
            batch,
            device,
            return_details=True,
        )

        pred = out["prediction"]

        predictions.extend(
            pred.squeeze(-1)
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

        modality_weights.append(
            out["modality_weights"]
            .cpu()
            .numpy()
        )

    result = metrics(
        targets,
        predictions,
    )

    weights = np.concatenate(
        modality_weights,
        axis=0,
    )

    result["expression_weight"] = float(
        weights[:, 0].mean()
    )

    result["cnv_weight"] = float(
        weights[:, 1].mean()
    )

    result["mutation_weight"] = float(
        weights[:, 2].mean()
    )

    result["crispr_weight"] = float(
        weights[:, 3].mean()
    )

    result["drmref_alpha"] = float(
        torch.sigmoid(
            model.drmref_alpha_logit
        ).item()
    )

    return result


# ============================================================
# TRAIN
# ============================================================

def main(
    seed=42,
    epochs=80,
    batch_size=128,
    lr=1e-4,
    patience=12,
):

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("PHAROS MOCKTAIL V4")
    print("=" * 70)

    print("Seed:", seed)
    print("Device:", device)

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    train_dataset = PharosMocktailV4Dataset(
        split="train",
        seed=seed,
    )

    val_dataset = PharosMocktailV4Dataset(
        split="val",
        seed=seed,
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

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = PharosMocktailV4(
        expression_dim=train_dataset.expression_dim,
        cnv_dim=train_dataset.cnv_dim,
        mutation_dim=train_dataset.mutation_dim,
        crispr_dim=train_dataset.crispr_dim,
        dropout=0.2,
    ).to(device)

    print(
        "Parameters:",
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    criterion = nn.MSELoss()

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
        )
    )

    # --------------------------------------------------------
    # OUTPUTS
    # --------------------------------------------------------

    OUT = Path("outputs/v4")
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        OUT /
        f"mocktail_v4_seed{seed}_best.pt"
    )

    metrics_path = (
        OUT /
        f"mocktail_v4_seed{seed}_best_metrics.csv"
    )

    history_path = (
        OUT /
        f"mocktail_v4_seed{seed}_history.csv"
    )

    # --------------------------------------------------------
    # TRAIN LOOP
    # --------------------------------------------------------

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

        total_squared_error = 0.0
        total_count = 0

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
                max_norm=5.0,
            )

            optimizer.step()

            total_squared_error += (
                (pred.detach() - target)
                .pow(2)
                .sum()
                .item()
            )

            total_count += (
                target.numel()
            )

        train_rmse = np.sqrt(
            total_squared_error
            / total_count
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        val = evaluate(
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
            "epoch": epoch,
            "train_rmse": train_rmse,
            **val,
            "lr": lr_now,
            "seconds": elapsed,
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
            f"| Val RMSE {val['rmse']:.4f} "
            f"| R2 {val['r2']:.4f} "
            f"| Pearson {val['pearson']:.4f} "
            f"| Spearman {val['spearman']:.4f} "
            f"| LR {lr_now:.2e} "
            f"| {elapsed:.1f}s"
        )

        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        if val["rmse"] < best_rmse:

            best_rmse = val["rmse"]
            best_epoch = epoch
            bad_epochs = 0

            torch.save(
                {
                    "seed": seed,
                    "epoch": epoch,
                    "model_state_dict":
                        model.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "metrics": val,
                    "dimensions": {
                        "expression":
                            train_dataset.expression_dim,
                        "cnv":
                            train_dataset.cnv_dim,
                        "mutation":
                            train_dataset.mutation_dim,
                        "crispr":
                            train_dataset.crispr_dim,
                    },
                },
                checkpoint_path,
            )

            best_result = {
                "seed": seed,
                "best_epoch": epoch,
                **val,
            }

            pd.DataFrame(
                [best_result]
            ).to_csv(
                metrics_path,
                index=False,
            )

            print(
                "  ✅ New best model"
            )

        else:

            bad_epochs += 1

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if bad_epochs >= patience:

            print()
            print(
                f"Early stopping at epoch {epoch}"
            )

            break

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        "Best epoch:",
        best_epoch
    )

    print(
        "Best validation RMSE:",
        f"{best_rmse:.6f}"
    )

    print(
        "Checkpoint:",
        checkpoint_path
    )


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
