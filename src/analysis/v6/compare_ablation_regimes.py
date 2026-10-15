from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = [42, 123, 456]

OUT_DIR = Path(
    "outputs/v6/error_autopsy"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# EXPERIMENT PATHS
# ============================================================

def get_path(model, seed):

    if model == "synergyx":
        return Path(
            f"outputs/v4/benchmarks/"
            f"synergyx/seed_{seed}/"
            f"val_predictions.parquet"
        )

    if model == "v5":
        return Path(
            f"outputs/v5/predictions/"
            f"v5/seed_{seed}/"
            f"val_predictions.parquet"
        )

    if model == "v5_1":
        return Path(
            f"outputs/v5/predictions/"
            f"v5_1/seed_{seed}/"
            f"val_predictions.parquet"
        )

    return Path(
        f"outputs/v4/ablations/"
        f"{model}/seed_{seed}/"
        f"val_predictions.parquet"
    )


MODELS = [
    "synergyx",
    "no_resistance",
    "no_drmref",
    "no_core29",
    "mutation_only",
    "expression_only",
    "v5",
    "v5_1",
]


# ============================================================
# LOAD + SUMMARIZE
# ============================================================

rows = []

near_zero_rows = []

overall_rows = []


for model in MODELS:

    for seed in SEEDS:

        path = get_path(
            model,
            seed,
        )

        if not path.exists():
            raise FileNotFoundError(
                path
            )

        df = pd.read_parquet(
            path
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

        df["zip_region"] = pd.cut(
            df["zip_score"],
            bins=[
                -np.inf,
                -10,
                -5,
                5,
                10,
                np.inf,
            ],
            labels=[
                "strong_antagonism",
                "moderate_antagonism",
                "near_zero",
                "moderate_synergy",
                "strong_synergy",
            ],
        )

        # ====================================================
        # OVERALL
        # ====================================================

        overall_rows.append({
            "model":
                model,

            "seed":
                seed,

            "rows":
                len(df),

            "rmse":
                np.sqrt(
                    df[
                        "squared_error"
                    ].mean()
                ),

            "mae":
                df[
                    "absolute_error"
                ].mean(),

            "prediction_std":
                df[
                    "prediction"
                ].std(),

            "bias":
                df[
                    "error"
                ].mean(),
        })

        # ====================================================
        # ZIP REGIONS
        # ====================================================

        for region, d in df.groupby(
            "zip_region",
            observed=True,
        ):

            rows.append({
                "model":
                    model,

                "seed":
                    seed,

                "zip_region":
                    str(region),

                "rows":
                    len(d),

                "true_mean":
                    d[
                        "zip_score"
                    ].mean(),

                "prediction_mean":
                    d[
                        "prediction"
                    ].mean(),

                "prediction_std":
                    d[
                        "prediction"
                    ].std(),

                "bias":
                    d[
                        "error"
                    ].mean(),

                "mae":
                    d[
                        "absolute_error"
                    ].mean(),

                "rmse":
                    np.sqrt(
                        d[
                            "squared_error"
                        ].mean()
                    ),
            })

        # ====================================================
        # NEAR-ZERO SPECIAL ANALYSIS
        # ====================================================

        nz = df[
            df["zip_region"]
            == "near_zero"
        ]

        near_zero_rows.append({
            "model":
                model,

            "seed":
                seed,

            "rows":
                len(nz),

            "prediction_mean":
                nz[
                    "prediction"
                ].mean(),

            "prediction_std":
                nz[
                    "prediction"
                ].std(),

            "bias":
                nz[
                    "error"
                ].mean(),

            "mae":
                nz[
                    "absolute_error"
                ].mean(),

            "rmse":
                np.sqrt(
                    nz[
                        "squared_error"
                    ].mean()
                ),

            "fraction_abs_pred_gt_5":
                (
                    nz[
                        "prediction"
                    ].abs()
                    > 5
                ).mean(),

            "fraction_abs_pred_gt_10":
                (
                    nz[
                        "prediction"
                    ].abs()
                    > 10
                ).mean(),
        })


overall = pd.DataFrame(
    overall_rows
)

region = pd.DataFrame(
    rows
)

near_zero = pd.DataFrame(
    near_zero_rows
)


# ============================================================
# 3-SEED AGGREGATES
# ============================================================

overall_mean = (
    overall
    .groupby(
        "model",
        observed=True,
    )
    .agg(
        rmse_mean=(
            "rmse",
            "mean",
        ),

        rmse_sd=(
            "rmse",
            "std",
        ),

        mae_mean=(
            "mae",
            "mean",
        ),

        prediction_std_mean=(
            "prediction_std",
            "mean",
        ),

        bias_mean=(
            "bias",
            "mean",
        ),
    )
    .reset_index()
    .sort_values(
        "rmse_mean"
    )
)


region_mean = (
    region
    .groupby(
        [
            "model",
            "zip_region",
        ],
        observed=True,
    )
    .agg(
        rows_mean=(
            "rows",
            "mean",
        ),

        prediction_std_mean=(
            "prediction_std",
            "mean",
        ),

        bias_mean=(
            "bias",
            "mean",
        ),

        mae_mean=(
            "mae",
            "mean",
        ),

        rmse_mean=(
            "rmse",
            "mean",
        ),
    )
    .reset_index()
)


near_zero_mean = (
    near_zero
    .groupby(
        "model",
        observed=True,
    )
    .agg(
        prediction_std_mean=(
            "prediction_std",
            "mean",
        ),

        bias_mean=(
            "bias",
            "mean",
        ),

        mae_mean=(
            "mae",
            "mean",
        ),

        rmse_mean=(
            "rmse",
            "mean",
        ),

        fraction_abs_pred_gt_5_mean=(
            "fraction_abs_pred_gt_5",
            "mean",
        ),

        fraction_abs_pred_gt_10_mean=(
            "fraction_abs_pred_gt_10",
            "mean",
        ),
    )
    .reset_index()
    .sort_values(
        "mae_mean"
    )
)


# ============================================================
# SAVE
# ============================================================

overall.to_csv(
    OUT_DIR
    / "ablation_overall_by_seed.csv",
    index=False,
)

overall_mean.to_csv(
    OUT_DIR
    / "ablation_overall_3seed.csv",
    index=False,
)

region.to_csv(
    OUT_DIR
    / "ablation_zip_regions_by_seed.csv",
    index=False,
)

region_mean.to_csv(
    OUT_DIR
    / "ablation_zip_regions_3seed.csv",
    index=False,
)

near_zero.to_csv(
    OUT_DIR
    / "ablation_near_zero_by_seed.csv",
    index=False,
)

near_zero_mean.to_csv(
    OUT_DIR
    / "ablation_near_zero_3seed.csv",
    index=False,
)


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 100)
print("OVERALL — 3 SEED MEAN")
print("=" * 100)

print(
    overall_mean.to_string(
        index=False
    )
)


print()
print("=" * 100)
print("NEAR-ZERO — 3 SEED MEAN")
print("=" * 100)

print(
    near_zero_mean.to_string(
        index=False
    )
)


print()
print("=" * 100)
print("STRONG SYNERGY — 3 SEED MEAN")
print("=" * 100)

strong_syn = (
    region_mean[
        region_mean[
            "zip_region"
        ]
        == "strong_synergy"
    ]
    .sort_values(
        "mae_mean"
    )
)

print(
    strong_syn[
        [
            "model",
            "prediction_std_mean",
            "bias_mean",
            "mae_mean",
            "rmse_mean",
        ]
    ].to_string(
        index=False
    )
)


print()
print("=" * 100)
print("STRONG ANTAGONISM — 3 SEED MEAN")
print("=" * 100)

strong_ant = (
    region_mean[
        region_mean[
            "zip_region"
        ]
        == "strong_antagonism"
    ]
    .sort_values(
        "mae_mean"
    )
)

print(
    strong_ant[
        [
            "model",
            "prediction_std_mean",
            "bias_mean",
            "mae_mean",
            "rmse_mean",
        ]
    ].to_string(
        index=False
    )
)


print()
print("=" * 100)
print("SAVED")
print("=" * 100)

for name in [
    "ablation_overall_by_seed.csv",
    "ablation_overall_3seed.csv",
    "ablation_zip_regions_by_seed.csv",
    "ablation_zip_regions_3seed.csv",
    "ablation_near_zero_by_seed.csv",
    "ablation_near_zero_3seed.csv",
]:
    print(
        OUT_DIR / name
    )
