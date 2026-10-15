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
# HELPERS
# ============================================================

def canonical_pair(
    a,
    b,
):
    a = str(a)
    b = str(b)

    return (
        a + "||" + b
        if a <= b
        else b + "||" + a
    )


def load_seed(
    seed,
):
    sx_path = Path(
        f"outputs/v4/benchmarks/"
        f"synergyx/seed_{seed}/"
        f"val_predictions.parquet"
    )

    pharos_path = Path(
        f"outputs/v5/predictions/"
        f"v5_1/seed_{seed}/"
        f"val_predictions.parquet"
    )

    sx = pd.read_parquet(
        sx_path
    )

    ph = pd.read_parquet(
        pharos_path
    )

    print()
    print("=" * 80)
    print(f"SEED {seed}")
    print("=" * 80)

    print(
        "SynergyX rows:",
        len(sx)
    )

    print(
        "PHAROS rows  :",
        len(ph)
    )

    if len(sx) != len(ph):
        raise RuntimeError(
            f"Row count mismatch "
            f"for seed {seed}"
        )

    # --------------------------------------------------------
    # Canonical row identity
    # --------------------------------------------------------

    for df in [sx, ph]:

        df["pair_key"] = [
            canonical_pair(
                a,
                b,
            )
            for a, b in zip(
                df["drug_a_key"],
                df["drug_b_key"],
            )
        ]

        df["row_key"] = (
            df["pair_key"].astype(str)
            + "||"
            + df["depmap_id"].astype(str)
        )

    # --------------------------------------------------------
    # Some canonical pair + cell combinations occur more than
    # once because raw DrugComb records differ only in naming
    # variants / slightly different synergy measurements.
    #
    # Preserve every row by adding a within-group occurrence
    # counter. Do NOT aggregate these records.
    # --------------------------------------------------------

    sx_base_counts = (
        sx["row_key"]
        .value_counts()
        .sort_index()
    )

    ph_base_counts = (
        ph["row_key"]
        .value_counts()
        .sort_index()
    )

    all_base_keys = (
        sx_base_counts.index
        .union(ph_base_counts.index)
    )

    sx_compare = (
        sx_base_counts
        .reindex(
            all_base_keys,
            fill_value=0,
        )
    )

    ph_compare = (
        ph_base_counts
        .reindex(
            all_base_keys,
            fill_value=0,
        )
    )

    mismatched_groups = (
        sx_compare != ph_compare
    ).sum()

    duplicate_groups = (
        sx_base_counts > 1
    ).sum()

    print(
        "Duplicate canonical groups:",
        duplicate_groups
    )

    print(
        "Mismatched group counts:",
        mismatched_groups
    )

    if mismatched_groups:
        raise RuntimeError(
            "SynergyX and PHAROS have "
            "different duplicate-group sizes."
        )

    sx["row_occurrence"] = (
        sx.groupby(
            "row_key",
            sort=False,
        )
        .cumcount()
    )

    ph["row_occurrence"] = (
        ph.groupby(
            "row_key",
            sort=False,
        )
        .cumcount()
    )

    sx["row_key"] = (
        sx["row_key"]
        + "||"
        + sx["row_occurrence"].astype(str)
    )

    ph["row_key"] = (
        ph["row_key"]
        + "||"
        + ph["row_occurrence"].astype(str)
    )

    if sx["row_key"].duplicated().any():
        raise RuntimeError(
            "SynergyX row keys are still "
            "not unique."
        )

    if ph["row_key"].duplicated().any():
        raise RuntimeError(
            "PHAROS row keys are still "
            "not unique."
        )

    # --------------------------------------------------------
    # Join by identity, NOT row order
    # --------------------------------------------------------

    keep_meta = [
        "row_key",
        "pair_key",
        "drug_a",
        "drug_b",
        "drug_a_key",
        "drug_b_key",
        "depmap_id",
        "original_cell_name",
        "zip_score",
        "bliss_score",
        "loewe_score",
        "hsa_score",
        "replicate_count",
        "has_expression",
        "has_cnv",
        "has_mutation",
        "has_crispr",
        "omics_count",
    ]

    sx_small = sx[
        keep_meta
        + [
            "prediction",
            "absolute_error",
            "squared_error",
        ]
    ].copy()

    sx_small = sx_small.rename(
        columns={
            "prediction":
                "synergyx_prediction",
            "absolute_error":
                "synergyx_absolute_error",
            "squared_error":
                "synergyx_squared_error",
        }
    )

    ph_small = ph[
        [
            "row_key",
            "zip_score",
            "prediction",
            "absolute_error",
            "squared_error",
        ]
    ].copy()

    ph_small = ph_small.rename(
        columns={
            "zip_score":
                "pharos_zip_score",
            "prediction":
                "pharos_prediction",
            "absolute_error":
                "pharos_absolute_error",
            "squared_error":
                "pharos_squared_error",
        }
    )

    merged = sx_small.merge(
        ph_small,
        on="row_key",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(sx):
        raise RuntimeError(
            f"Join lost rows for "
            f"seed {seed}: "
            f"{len(merged)} / "
            f"{len(sx)}"
        )

    zip_diff = np.max(
        np.abs(
            merged["zip_score"]
            - merged["pharos_zip_score"]
        )
    )

    print(
        "Maximum ZIP mismatch:",
        zip_diff
    )

    if zip_diff > 1e-5:
        raise RuntimeError(
            "ZIP targets do not match."
        )

    merged = merged.drop(
        columns=[
            "pharos_zip_score"
        ]
    )

    # --------------------------------------------------------
    # Advantage definitions
    #
    # Positive = PHAROS better
    # --------------------------------------------------------

    merged[
        "pharos_abs_advantage"
    ] = (
        merged[
            "synergyx_absolute_error"
        ]
        - merged[
            "pharos_absolute_error"
        ]
    )

    merged[
        "pharos_sq_advantage"
    ] = (
        merged[
            "synergyx_squared_error"
        ]
        - merged[
            "pharos_squared_error"
        ]
    )

    merged[
        "winner"
    ] = np.where(
        merged[
            "pharos_absolute_error"
        ]
        < merged[
            "synergyx_absolute_error"
        ],
        "PHAROS",
        np.where(
            merged[
                "pharos_absolute_error"
            ]
            > merged[
                "synergyx_absolute_error"
            ],
            "SynergyX",
            "Tie",
        ),
    )

    merged["seed"] = seed

    # --------------------------------------------------------
    # ZIP bins
    # --------------------------------------------------------

    merged["zip_region"] = pd.cut(
        merged["zip_score"],
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

    return merged


# ============================================================
# LOAD ALL SEEDS
# ============================================================

all_frames = []

for seed in SEEDS:
    all_frames.append(
        load_seed(seed)
    )

all_df = pd.concat(
    all_frames,
    ignore_index=True,
)

all_path = (
    OUT_DIR
    / "v5_1_vs_synergyx_rows.parquet"
)

all_df.to_parquet(
    all_path,
    index=False,
)


# ============================================================
# GLOBAL SUMMARY
# ============================================================

summary_rows = []

for seed in SEEDS:

    d = all_df[
        all_df["seed"] == seed
    ]

    sx_rmse = np.sqrt(
        d[
            "synergyx_squared_error"
        ].mean()
    )

    ph_rmse = np.sqrt(
        d[
            "pharos_squared_error"
        ].mean()
    )

    ph_win_rate = (
        d["winner"]
        .eq("PHAROS")
        .mean()
    )

    sx_win_rate = (
        d["winner"]
        .eq("SynergyX")
        .mean()
    )

    summary_rows.append({
        "seed":
            seed,
        "rows":
            len(d),
        "synergyx_rmse":
            sx_rmse,
        "pharos_v5_1_rmse":
            ph_rmse,
        "rmse_difference_pharos_minus_synergyx":
            ph_rmse - sx_rmse,
        "pharos_row_win_rate":
            ph_win_rate,
        "synergyx_row_win_rate":
            sx_win_rate,
        "mean_pharos_abs_advantage":
            d[
                "pharos_abs_advantage"
            ].mean(),
        "mean_pharos_sq_advantage":
            d[
                "pharos_sq_advantage"
            ].mean(),
    })

summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    OUT_DIR
    / "v5_1_vs_synergyx_summary.csv",
    index=False,
)


# ============================================================
# ZIP REGION ANALYSIS
# ============================================================

zip_summary = (
    all_df
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
        synergyx_mae=(
            "synergyx_absolute_error",
            "mean",
        ),
        pharos_mae=(
            "pharos_absolute_error",
            "mean",
        ),
        mean_pharos_advantage=(
            "pharos_abs_advantage",
            "mean",
        ),
        pharos_win_rate=(
            "winner",
            lambda x:
                (x == "PHAROS").mean(),
        ),
    )
    .reset_index()
)

zip_summary.to_csv(
    OUT_DIR
    / "v5_1_vs_synergyx_zip_regions.csv",
    index=False,
)


# ============================================================
# OMICS COUNT ANALYSIS
# ============================================================

omics_summary = (
    all_df
    .groupby(
        [
            "seed",
            "omics_count",
        ],
        observed=True,
    )
    .agg(
        rows=(
            "row_key",
            "size",
        ),
        synergyx_mae=(
            "synergyx_absolute_error",
            "mean",
        ),
        pharos_mae=(
            "pharos_absolute_error",
            "mean",
        ),
        mean_pharos_advantage=(
            "pharos_abs_advantage",
            "mean",
        ),
        pharos_win_rate=(
            "winner",
            lambda x:
                (x == "PHAROS").mean(),
        ),
    )
    .reset_index()
)

omics_summary.to_csv(
    OUT_DIR
    / "v5_1_vs_synergyx_omics_count.csv",
    index=False,
)


# ============================================================
# CELL-LINE ANALYSIS
# ============================================================

cell_summary = (
    all_df
    .groupby(
        [
            "seed",
            "depmap_id",
            "original_cell_name",
        ],
        observed=True,
    )
    .agg(
        rows=(
            "row_key",
            "size",
        ),
        synergyx_mae=(
            "synergyx_absolute_error",
            "mean",
        ),
        pharos_mae=(
            "pharos_absolute_error",
            "mean",
        ),
        mean_pharos_advantage=(
            "pharos_abs_advantage",
            "mean",
        ),
        pharos_win_rate=(
            "winner",
            lambda x:
                (x == "PHAROS").mean(),
        ),
    )
    .reset_index()
)

cell_summary.to_csv(
    OUT_DIR
    / "v5_1_vs_synergyx_cells.csv",
    index=False,
)


# ============================================================
# PAIR ANALYSIS
# ============================================================

pair_summary = (
    all_df
    .groupby(
        [
            "seed",
            "pair_key",
        ],
        observed=True,
    )
    .agg(
        rows=(
            "row_key",
            "size",
        ),
        synergyx_mae=(
            "synergyx_absolute_error",
            "mean",
        ),
        pharos_mae=(
            "pharos_absolute_error",
            "mean",
        ),
        mean_pharos_advantage=(
            "pharos_abs_advantage",
            "mean",
        ),
        pharos_win_rate=(
            "winner",
            lambda x:
                (x == "PHAROS").mean(),
        ),
    )
    .reset_index()
)

pair_summary.to_csv(
    OUT_DIR
    / "v5_1_vs_synergyx_pairs.csv",
    index=False,
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("=" * 80)
print("V5.1 VS SYNERGYX")
print("=" * 80)

print(
    summary.to_string(
        index=False
    )
)

print()
print("=" * 80)
print("ZIP REGION SUMMARY")
print("=" * 80)

print(
    zip_summary.to_string(
        index=False
    )
)

print()
print("=" * 80)
print("SAVED")
print("=" * 80)

print(all_path)
print(
    OUT_DIR
    / "v5_1_vs_synergyx_summary.csv"
)
print(
    OUT_DIR
    / "v5_1_vs_synergyx_zip_regions.csv"
)
print(
    OUT_DIR
    / "v5_1_vs_synergyx_omics_count.csv"
)
print(
    OUT_DIR
    / "v5_1_vs_synergyx_cells.csv"
)
print(
    OUT_DIR
    / "v5_1_vs_synergyx_pairs.csv"
)
