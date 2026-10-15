from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.v4.mocktail_v4_dataset import (
    PharosMocktailV4Dataset,
)

from src.data.v5.cellresdb_relevance import (
    CellResDBRelevanceMapper,
)


SEEDS = [42, 123, 456]

MODELS = [
    "synergyx",
    "v5",
    "v5_1",
    "no_resistance",
    "no_drmref",
    "no_core29",
]

OUT_DIR = Path(
    "outputs/v6/error_autopsy"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# PATHS
# ============================================================

def prediction_path(
    model,
    seed,
):
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


# ============================================================
# ROW IDENTITY
# ============================================================

def canonical_pair(
    a,
    b,
):
    a = str(a)
    b = str(b)

    if a <= b:
        return f"{a}||{b}"

    return f"{b}||{a}"


def add_row_identity(
    df,
):
    df = df.copy()

    df["pair_key"] = [
        canonical_pair(a, b)
        for a, b in zip(
            df["drug_a_key"],
            df["drug_b_key"],
        )
    ]

    df["base_row_key"] = (
        df["pair_key"]
        + "||"
        + df["depmap_id"].astype(str)
    )

    df["row_occurrence"] = (
        df.groupby(
            "base_row_key",
            sort=False,
        )
        .cumcount()
    )

    df["row_key"] = (
        df["base_row_key"]
        + "||"
        + df["row_occurrence"].astype(str)
    )

    if df["row_key"].duplicated().any():
        raise RuntimeError(
            "Row identity is not unique."
        )

    return df


# ============================================================
# DRMREF ACTIVE DRUGS
# ============================================================

def get_active_drmref_drugs(
    seed,
):
    ds = PharosMocktailV4Dataset(
        split="val",
        seed=seed,
    )

    active = set()

    for drug, value in (
        ds.drmref_available.items()
    ):

        if torch.is_tensor(value):
            is_active = bool(
                value.detach()
                .cpu()
                .float()
                .max()
                .item()
                > 0
            )

        else:
            try:
                is_active = bool(value)
            except Exception:
                is_active = False

        if is_active:
            active.add(
                str(drug)
                .strip()
                .upper()
            )

    return active


# ============================================================
# CELLRESDB
# ============================================================

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

CELLRES_DRUGS = {
    x.upper()
    for x in mapper.drug_to_cohorts.keys()
}

MM_DRUGS = {
    "CARFILZOMIB",
    "DEXAMETHASONE",
    "POMALIDOMIDE",
}


# ============================================================
# SUBSET ANNOTATION
# ============================================================

def annotate_evidence(
    df,
    drmref_drugs,
):
    df = df.copy()

    a = (
        df["drug_a_key"]
        .astype(str)
        .str.upper()
    )

    b = (
        df["drug_b_key"]
        .astype(str)
        .str.upper()
    )

    df["drug_a_drmref"] = (
        a.isin(drmref_drugs)
    )

    df["drug_b_drmref"] = (
        b.isin(drmref_drugs)
    )

    df["has_drmref_evidence"] = (
        df["drug_a_drmref"]
        | df["drug_b_drmref"]
    )

    df["drug_a_cellresdb"] = (
        a.isin(CELLRES_DRUGS)
    )

    df["drug_b_cellresdb"] = (
        b.isin(CELLRES_DRUGS)
    )

    df["has_cellresdb_evidence"] = (
        df["drug_a_cellresdb"]
        | df["drug_b_cellresdb"]
    )

    df["has_any_resistance_evidence"] = (
        df["has_drmref_evidence"]
        | df["has_cellresdb_evidence"]
    )

    df["has_both_resistance_sources"] = (
        df["has_drmref_evidence"]
        & df["has_cellresdb_evidence"]
    )

    df["drmref_only"] = (
        df["has_drmref_evidence"]
        & ~df["has_cellresdb_evidence"]
    )

    df["cellresdb_only"] = (
        df["has_cellresdb_evidence"]
        & ~df["has_drmref_evidence"]
    )

    df["no_resistance_evidence"] = (
        ~df["has_any_resistance_evidence"]
    )

    # --------------------------------------------------------
    # CellResDB-specific subsets
    # --------------------------------------------------------

    df["paclitaxel_direct"] = (
        (a == "PACLITAXEL")
        | (b == "PACLITAXEL")
    )

    df["vismodegib_bcc_context"] = (
        (a == "VISMODEGIB")
        | (b == "VISMODEGIB")
    )

    df["mm_context"] = (
        a.isin(MM_DRUGS)
        | b.isin(MM_DRUGS)
    )

    return df


# ============================================================
# LOAD REFERENCE + MODELS
# ============================================================

all_seed_tables = []

for seed in SEEDS:

    print()
    print("=" * 100)
    print(f"SEED {seed}")
    print("=" * 100)

    drmref_drugs = (
        get_active_drmref_drugs(seed)
    )

    print(
        "Active DRMref drugs:",
        sorted(drmref_drugs),
    )

    sx = pd.read_parquet(
        prediction_path(
            "synergyx",
            seed,
        )
    )

    sx = add_row_identity(sx)

    sx = annotate_evidence(
        sx,
        drmref_drugs,
    )

    # Keep metadata from SynergyX reference
    meta_cols = [
        "row_key",
        "pair_key",
        "drug_a",
        "drug_b",
        "drug_a_key",
        "drug_b_key",
        "depmap_id",
        "original_cell_name",
        "zip_score",
        "has_drmref_evidence",
        "has_cellresdb_evidence",
        "has_any_resistance_evidence",
        "has_both_resistance_sources",
        "drmref_only",
        "cellresdb_only",
        "no_resistance_evidence",
        "paclitaxel_direct",
        "vismodegib_bcc_context",
        "mm_context",
    ]

    combined = sx[
        meta_cols
    ].copy()

    for model in MODELS:

        path = prediction_path(
            model,
            seed,
        )

        df = pd.read_parquet(path)

        df = add_row_identity(df)

        if len(df) != len(sx):
            raise RuntimeError(
                f"{model} seed {seed}: "
                f"row count mismatch"
            )

        pred_col = (
            f"{model}_prediction"
        )

        temp = df[
            [
                "row_key",
                "zip_score",
                "prediction",
            ]
        ].copy()

        temp = temp.rename(
            columns={
                "zip_score":
                    f"{model}_zip",
                "prediction":
                    pred_col,
            }
        )

        combined = combined.merge(
            temp,
            on="row_key",
            how="left",
            validate="one_to_one",
        )

        max_target_diff = (
            combined["zip_score"]
            - combined[
                f"{model}_zip"
            ]
        ).abs().max()

        if max_target_diff > 1e-5:
            raise RuntimeError(
                f"{model} seed {seed}: "
                f"ZIP mismatch "
                f"{max_target_diff}"
            )

        combined = combined.drop(
            columns=[
                f"{model}_zip"
            ]
        )

    combined["seed"] = seed

    all_seed_tables.append(
        combined
    )


all_df = pd.concat(
    all_seed_tables,
    ignore_index=True,
)


# ============================================================
# SUBSETS
# ============================================================

SUBSETS = {
    "all":
        lambda d:
            np.ones(
                len(d),
                dtype=bool,
            ),

    "no_resistance_evidence":
        lambda d:
            d[
                "no_resistance_evidence"
            ],

    "any_resistance_evidence":
        lambda d:
            d[
                "has_any_resistance_evidence"
            ],

    "drmref_any":
        lambda d:
            d[
                "has_drmref_evidence"
            ],

    "cellresdb_any":
        lambda d:
            d[
                "has_cellresdb_evidence"
            ],

    "drmref_only":
        lambda d:
            d[
                "drmref_only"
            ],

    "cellresdb_only":
        lambda d:
            d[
                "cellresdb_only"
            ],

    "both_sources":
        lambda d:
            d[
                "has_both_resistance_sources"
            ],

    "paclitaxel_direct":
        lambda d:
            d[
                "paclitaxel_direct"
            ],

    "vismodegib_bcc_context":
        lambda d:
            d[
                "vismodegib_bcc_context"
            ],

    "mm_context":
        lambda d:
            d[
                "mm_context"
            ],
}


# ============================================================
# METRICS
# ============================================================

metric_rows = []

for seed in SEEDS:

    seed_df = all_df[
        all_df["seed"] == seed
    ]

    for subset_name, mask_fn in (
        SUBSETS.items()
    ):

        mask = mask_fn(
            seed_df
        )

        subset = seed_df[
            mask
        ]

        if len(subset) == 0:
            continue

        for model in MODELS:

            y = (
                subset[
                    "zip_score"
                ]
                .to_numpy(
                    dtype=np.float64
                )
            )

            p = (
                subset[
                    f"{model}_prediction"
                ]
                .to_numpy(
                    dtype=np.float64
                )
            )

            error = p - y

            metric_rows.append({
                "seed":
                    seed,

                "subset":
                    subset_name,

                "model":
                    model,

                "rows":
                    len(subset),

                "rmse":
                    float(
                        np.sqrt(
                            np.mean(
                                error ** 2
                            )
                        )
                    ),

                "mae":
                    float(
                        np.mean(
                            np.abs(error)
                        )
                    ),

                "bias":
                    float(
                        np.mean(error)
                    ),
            })


metrics = pd.DataFrame(
    metric_rows
)


# ============================================================
# THREE-SEED SUMMARY
# ============================================================

summary = (
    metrics
    .groupby(
        [
            "subset",
            "model",
        ],
        observed=True,
    )
    .agg(
        rows_mean=(
            "rows",
            "mean",
        ),

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

        mae_sd=(
            "mae",
            "std",
        ),

        bias_mean=(
            "bias",
            "mean",
        ),
    )
    .reset_index()
)


# ============================================================
# V5.1 VS SYNERGYX
# ============================================================

pivot = summary.pivot(
    index="subset",
    columns="model",
    values=[
        "rmse_mean",
        "mae_mean",
    ],
)

comparison_rows = []

for subset in pivot.index:

    if (
        ("rmse_mean", "v5_1")
        not in pivot.columns
        or
        ("rmse_mean", "synergyx")
        not in pivot.columns
    ):
        continue

    ph_rmse = pivot.loc[
        subset,
        ("rmse_mean", "v5_1")
    ]

    sx_rmse = pivot.loc[
        subset,
        ("rmse_mean", "synergyx")
    ]

    ph_mae = pivot.loc[
        subset,
        ("mae_mean", "v5_1")
    ]

    sx_mae = pivot.loc[
        subset,
        ("mae_mean", "synergyx")
    ]

    comparison_rows.append({
        "subset":
            subset,

        "pharos_v5_1_rmse":
            ph_rmse,

        "synergyx_rmse":
            sx_rmse,

        "rmse_delta_pharos_minus_synergyx":
            ph_rmse - sx_rmse,

        "pharos_v5_1_mae":
            ph_mae,

        "synergyx_mae":
            sx_mae,

        "mae_delta_pharos_minus_synergyx":
            ph_mae - sx_mae,
    })


comparison = pd.DataFrame(
    comparison_rows
)


# ============================================================
# RESISTANCE COMPONENT COMPARISON
# ============================================================

component_models = [
    "v5_1",
    "v5",
    "no_resistance",
    "no_drmref",
    "no_core29",
]

component_summary = summary[
    summary["model"].isin(
        component_models
    )
].copy()


# ============================================================
# SAVE
# ============================================================

all_df.to_parquet(
    OUT_DIR
    / "resistance_subset_rows.parquet",
    index=False,
)

metrics.to_csv(
    OUT_DIR
    / "resistance_subset_metrics_by_seed.csv",
    index=False,
)

summary.to_csv(
    OUT_DIR
    / "resistance_subset_metrics_3seed.csv",
    index=False,
)

comparison.to_csv(
    OUT_DIR
    / "resistance_v5_1_vs_synergyx.csv",
    index=False,
)

component_summary.to_csv(
    OUT_DIR
    / "resistance_component_comparison.csv",
    index=False,
)


# ============================================================
# PRINT COVERAGE
# ============================================================

print()
print("=" * 110)
print("RESISTANCE EVIDENCE COVERAGE")
print("=" * 110)

for seed in SEEDS:

    d = all_df[
        all_df["seed"] == seed
    ]

    print()
    print(f"SEED {seed}")

    for subset_name in [
        "no_resistance_evidence",
        "any_resistance_evidence",
        "drmref_any",
        "cellresdb_any",
        "drmref_only",
        "cellresdb_only",
        "both_sources",
        "paclitaxel_direct",
        "vismodegib_bcc_context",
        "mm_context",
    ]:

        mask = SUBSETS[
            subset_name
        ](d)

        count = int(
            mask.sum()
        )

        pct = (
            100.0
            * count
            / len(d)
        )

        print(
            f"{subset_name:28s} "
            f"{count:6d} "
            f"({pct:6.2f}%)"
        )


# ============================================================
# PRINT V5.1 VS SYNERGYX
# ============================================================

print()
print("=" * 110)
print("V5.1 VS SYNERGYX — 3-SEED MEAN")
print("=" * 110)

print(
    comparison
    .sort_values(
        "rmse_delta_pharos_minus_synergyx"
    )
    .to_string(
        index=False
    )
)


# ============================================================
# PRINT RESISTANCE MODEL COMPARISON
# ============================================================

important_subsets = [
    "any_resistance_evidence",
    "drmref_any",
    "cellresdb_any",
    "both_sources",
    "paclitaxel_direct",
    "vismodegib_bcc_context",
    "mm_context",
]

for subset in important_subsets:

    print()
    print("=" * 110)
    print(
        f"{subset.upper()} — "
        f"3-SEED MEAN"
    )
    print("=" * 110)

    t = summary[
        summary["subset"]
        == subset
    ].copy()

    t = t.sort_values(
        "rmse_mean"
    )

    print(
        t[
            [
                "model",
                "rows_mean",
                "rmse_mean",
                "rmse_sd",
                "mae_mean",
                "bias_mean",
            ]
        ].to_string(
            index=False
        )
    )


print()
print("=" * 110)
print("SAVED")
print("=" * 110)

for name in [
    "resistance_subset_rows.parquet",
    "resistance_subset_metrics_by_seed.csv",
    "resistance_subset_metrics_3seed.csv",
    "resistance_v5_1_vs_synergyx.csv",
    "resistance_component_comparison.csv",
]:
    print(
        OUT_DIR / name
    )
