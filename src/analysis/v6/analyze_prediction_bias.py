from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path(
    "outputs/v6/error_autopsy/"
    "v5_1_vs_synergyx_rows.parquet"
)

OUT = Path(
    "outputs/v6/error_autopsy/"
)

df = pd.read_parquet(INPUT)


# ============================================================
# SIGNED ERRORS
# ============================================================

df["synergyx_error"] = (
    df["synergyx_prediction"]
    - df["zip_score"]
)

df["pharos_error"] = (
    df["pharos_prediction"]
    - df["zip_score"]
)


# ============================================================
# REGION SUMMARY
# ============================================================

region = (
    df
    .groupby(
        [
            "seed",
            "zip_region",
        ],
        observed=True,
    )
    .agg(
        rows=(
            "row_key",
            "size",
        ),

        true_zip_mean=(
            "zip_score",
            "mean",
        ),

        synergyx_prediction_mean=(
            "synergyx_prediction",
            "mean",
        ),

        pharos_prediction_mean=(
            "pharos_prediction",
            "mean",
        ),

        synergyx_bias=(
            "synergyx_error",
            "mean",
        ),

        pharos_bias=(
            "pharos_error",
            "mean",
        ),

        synergyx_prediction_std=(
            "synergyx_prediction",
            "std",
        ),

        pharos_prediction_std=(
            "pharos_prediction",
            "std",
        ),

        synergyx_mae=(
            "synergyx_absolute_error",
            "mean",
        ),

        pharos_mae=(
            "pharos_absolute_error",
            "mean",
        ),
    )
    .reset_index()
)


# ============================================================
# NEAR-ZERO SPECIAL AUDIT
# ============================================================

near = df[
    df["zip_region"]
    == "near_zero"
].copy()

near_summary = (
    near
    .groupby(
        "seed"
    )
    .agg(
        rows=(
            "row_key",
            "size",
        ),

        true_mean=(
            "zip_score",
            "mean",
        ),

        synergyx_pred_mean=(
            "synergyx_prediction",
            "mean",
        ),

        pharos_pred_mean=(
            "pharos_prediction",
            "mean",
        ),

        synergyx_pred_std=(
            "synergyx_prediction",
            "std",
        ),

        pharos_pred_std=(
            "pharos_prediction",
            "std",
        ),

        synergyx_bias=(
            "synergyx_error",
            "mean",
        ),

        pharos_bias=(
            "pharos_error",
            "mean",
        ),

        synergyx_mae=(
            "synergyx_absolute_error",
            "mean",
        ),

        pharos_mae=(
            "pharos_absolute_error",
            "mean",
        ),
    )
    .reset_index()
)


# ============================================================
# EXTREME PREDICTION RATE IN NEAR-ZERO TRUTHS
# ============================================================

rows = []

for seed in sorted(
    near["seed"].unique()
):

    d = near[
        near["seed"] == seed
    ]

    for threshold in [
        5,
        10,
    ]:

        rows.append({
            "seed":
                seed,

            "threshold":
                threshold,

            "rows":
                len(d),

            "synergyx_fraction_abs_prediction_above_threshold":
                (
                    d[
                        "synergyx_prediction"
                    ].abs()
                    > threshold
                ).mean(),

            "pharos_fraction_abs_prediction_above_threshold":
                (
                    d[
                        "pharos_prediction"
                    ].abs()
                    > threshold
                ).mean(),
        })

extreme = pd.DataFrame(
    rows
)


# ============================================================
# SAVE
# ============================================================

region.to_csv(
    OUT
    / "prediction_bias_by_zip_region.csv",
    index=False,
)

near_summary.to_csv(
    OUT
    / "prediction_bias_near_zero.csv",
    index=False,
)

extreme.to_csv(
    OUT
    / "near_zero_extreme_prediction_rate.csv",
    index=False,
)


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 100)
print("PREDICTION BIAS BY ZIP REGION")
print("=" * 100)

print(
    region.to_string(
        index=False
    )
)

print()
print("=" * 100)
print("NEAR-ZERO SPECIAL AUDIT")
print("=" * 100)

print(
    near_summary.to_string(
        index=False
    )
)

print()
print("=" * 100)
print("NEAR-ZERO EXTREME-PREDICTION RATE")
print("=" * 100)

print(
    extreme.to_string(
        index=False
    )
)
