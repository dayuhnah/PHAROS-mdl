from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.v4.mocktail_v4_dataset import (
    PharosMocktailV4Dataset,
)

from src.data.v5.cellresdb_relevance import (
    CellResDBRelevanceMapper,
)

from src.models.v4.pharos_mocktail_v4 import (
    PharosMocktailV4,
)

from src.models.v5.pharos_mocktail_v5 import (
    PharosMocktailV5,
)

from src.models.v5.pharos_mocktail_v5_1 import (
    PharosMocktailV51,
)


SEEDS = [42, 123, 456]


# ============================================================
# METRICS
# ============================================================

def compute_metrics(
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

    error = y_pred - y_true

    rmse = np.sqrt(
        np.mean(
            error ** 2
        )
    )

    mae = np.mean(
        np.abs(error)
    )

    ss_res = np.sum(
        (y_true - y_pred) ** 2
    )

    ss_tot = np.sum(
        (
            y_true
            - y_true.mean()
        ) ** 2
    )

    r2 = (
        1.0
        - ss_res / ss_tot
    )

    pearson = np.corrcoef(
        y_true,
        y_pred,
    )[0, 1]

    true_rank = (
        pd.Series(y_true)
        .rank()
        .to_numpy()
    )

    pred_rank = (
        pd.Series(y_pred)
        .rank()
        .to_numpy()
    )

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
# PATHS
# ============================================================

def checkpoint_path(
    model_name,
    seed,
):
    if model_name == "v4":
        return Path(
            f"outputs/v4/"
            f"mocktail_v4_seed{seed}_best.pt"
        )

    if model_name == "v5":
        return Path(
            f"outputs/v5/"
            f"mocktail_v5_seed{seed}_best.pt"
        )

    if model_name == "v5_1":
        return Path(
            f"outputs/v5/"
            f"mocktail_v5_1_seed{seed}_best.pt"
        )

    raise ValueError(
        f"Unknown model: {model_name}"
    )


def metrics_path(
    model_name,
    seed,
):
    if model_name == "v4":
        return Path(
            f"outputs/v4/"
            f"mocktail_v4_seed{seed}"
            f"_best_metrics.csv"
        )

    if model_name == "v5":
        return Path(
            f"outputs/v5/"
            f"mocktail_v5_seed{seed}"
            f"_best_metrics.csv"
        )

    if model_name == "v5_1":
        return Path(
            f"outputs/v5/"
            f"mocktail_v5_1_seed{seed}"
            f"_best_metrics.csv"
        )

    raise ValueError(
        f"Unknown model: {model_name}"
    )


def output_path(
    model_name,
    seed,
):
    if model_name == "v4":
        root = Path(
            "outputs/v4/predictions/v4"
        )

    elif model_name == "v5":
        root = Path(
            "outputs/v5/predictions/v5"
        )

    elif model_name == "v5_1":
        root = Path(
            "outputs/v5/predictions/v5_1"
        )

    else:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    path = (
        root
        / f"seed_{seed}"
        / "val_predictions.parquet"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


# ============================================================
# MODEL CONSTRUCTION
# ============================================================

def build_model(
    model_name,
    dataset,
    device,
):
    common = {
        "expression_dim":
            dataset.expression_dim,
        "cnv_dim":
            dataset.cnv_dim,
        "mutation_dim":
            dataset.mutation_dim,
        "crispr_dim":
            dataset.crispr_dim,
        "dropout":
            0.2,
    }

    if model_name == "v4":
        model = PharosMocktailV4(
            **common
        )

    elif model_name == "v5":
        model = PharosMocktailV5(
            **common
        )

    elif model_name == "v5_1":
        model = PharosMocktailV51(
            **common
        )

    else:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    return model.to(device)


# ============================================================
# FORWARD
# ============================================================

def forward_model(
    model_name,
    model,
    batch,
    device,
    mapper=None,
):
    def x(key):
        return batch[key].to(
            device,
            non_blocking=True,
        )

    if model_name == "v4":

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
        )

    if model_name == "v5":

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

            x("drug_a_drmref_expr"),
            x("drug_b_drmref_expr"),

            x("drug_a_drmref_available"),
            x("drug_b_drmref_available"),
        )

    if model_name == "v5_1":

        if mapper is None:
            raise RuntimeError(
                "V5.1 requires CellResDB mapper."
            )

        relevance = mapper.build(
            batch["drug_a"],
            batch["drug_b"],
            device=device,
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

            x("drug_a_drmref_expr"),
            x("drug_b_drmref_expr"),

            x("drug_a_drmref_available"),
            x("drug_b_drmref_available"),

            cellresdb_token_relevance=(
                relevance
            ),
        )

    raise ValueError(
        f"Unknown model: {model_name}"
    )


# ============================================================
# EXPORT ONE MODEL / SEED
# ============================================================

def export_one(
    model_name,
    seed,
    batch_size,
    device,
    overwrite=False,
):
    print()
    print("=" * 80)
    print(
        f"EXPORT {model_name.upper()} "
        f"SEED {seed}"
    )
    print("=" * 80)

    out_path = output_path(
        model_name,
        seed,
    )

    if (
        out_path.exists()
        and not overwrite
    ):
        print(
            "Already exists:",
            out_path,
        )
        print(
            "Skipping. Use --overwrite "
            "to regenerate."
        )
        return

    dataset = (
        PharosMocktailV4Dataset(
            split="val",
            seed=seed,
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    mapper = None

    if model_name == "v5_1":

        mapper = CellResDBRelevanceMapper(
            metadata_path=(
                "data/processed/v5/"
                "cellresdb/"
                "resistance_bank_cohort/"
                "model_cache/"
                "cellresdb_token_metadata.json"
            ),
            pathway_tokens_path=(
                "data/processed/v5/"
                "cellresdb/"
                "resistance_bank_cohort/"
                "cellresdb_pathway_tokens.csv"
            ),
        )

    model = build_model(
        model_name,
        dataset,
        device,
    )

    ckpt_path = checkpoint_path(
        model_name,
        seed,
    )

    checkpoint = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    print(
        "Checkpoint:",
        ckpt_path,
    )

    print(
        "Checkpoint epoch:",
        checkpoint["epoch"],
    )

    print(
        "Validation rows:",
        len(dataset),
    )

    predictions = []

    with torch.no_grad():

        for batch_idx, batch in enumerate(
            loader,
            start=1,
        ):

            pred = forward_model(
                model_name,
                model,
                batch,
                device,
                mapper=mapper,
            )

            predictions.append(
                pred.squeeze(-1)
                .detach()
                .cpu()
                .numpy()
            )

            if (
                batch_idx == 1
                or batch_idx % 50 == 0
                or batch_idx == len(loader)
            ):
                print(
                    f"Batch "
                    f"{batch_idx}/"
                    f"{len(loader)}"
                )

    predictions = np.concatenate(
        predictions
    )

    if len(predictions) != len(
        dataset
    ):
        raise RuntimeError(
            "Prediction count does not "
            "match dataset length."
        )

    df = (
        dataset.df
        .copy()
        .reset_index(
            drop=True
        )
    )

    df["prediction"] = (
        predictions
    )

    df["error"] = (
        df["prediction"]
        - df["zip_score"]
    )

    df["absolute_error"] = (
        df["error"].abs()
    )

    df["squared_error"] = (
        df["error"] ** 2
    )

    df["experiment"] = (
        f"pharos_{model_name}"
    )

    df["seed"] = seed

    reproduced = compute_metrics(
        df["zip_score"],
        df["prediction"],
    )

    saved_path = metrics_path(
        model_name,
        seed,
    )

    saved = pd.read_csv(
        saved_path
    ).iloc[0]

    print()
    print("REPRODUCED vs SAVED")
    print("-" * 80)

    for metric in [
        "rmse",
        "mae",
        "r2",
        "pearson",
        "spearman",
    ]:

        reproduced_value = (
            reproduced[metric]
        )

        saved_value = float(
            saved[metric]
        )

        diff = abs(
            reproduced_value
            - saved_value
        )

        print(
            f"{metric:9s} "
            f"reproduced="
            f"{reproduced_value:.12f} "
            f"saved="
            f"{saved_value:.12f} "
            f"diff="
            f"{diff:.3e}"
        )

    rmse_diff = abs(
        reproduced["rmse"]
        - float(saved["rmse"])
    )

    if rmse_diff > 1e-5:

        raise RuntimeError(
            f"RMSE mismatch for "
            f"{model_name} seed {seed}: "
            f"{rmse_diff}"
        )

    df.to_parquet(
        out_path,
        index=False,
    )

    print()
    print(
        "✅ VERIFIED AND SAVED:",
        out_path,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models",
        nargs="+",
        choices=[
            "v4",
            "v5",
            "v5_1",
        ],
        default=[
            "v4",
            "v5",
            "v5_1",
        ],
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEEDS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 80)
    print("PHAROS VALIDATION PREDICTION EXPORT")
    print("=" * 80)

    print("Device:", device)
    print("Models:", args.models)
    print("Seeds :", args.seeds)

    for seed in args.seeds:

        for model_name in args.models:

            export_one(
                model_name=model_name,
                seed=seed,
                batch_size=args.batch_size,
                device=device,
                overwrite=args.overwrite,
            )

    print()
    print("=" * 80)
    print("ALL REQUESTED EXPORTS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
