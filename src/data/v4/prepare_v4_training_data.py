from pathlib import Path
import argparse
import numpy as np
import pandas as pd


CORE29 = [
    "ABCB1","ABCC1","ABCG2","BCL2","BAX","BAK1",
    "CASP3","CASP8","CASP9","AKT1","TP53",
    "BRCA1","BRCA2","RAD51","ERCC1","MGMT",
    "ATM","ATR","GSTP1","GSTA1","CYP3A4",
    "CYP2D6","HIF1A","VEGFA","IL6","CXCR4",
    "DNMT1","HDAC1","HDAC2",
]

DATA = Path(
    "data/processed/v4/pharos_mocktail_v4_multiomics.parquet"
)

OMICS = Path(
    "data/processed/v4/multiomics"
)

OUT = Path(
    "data/processed/v4/training"
)

OUT.mkdir(parents=True, exist_ok=True)


def preprocess_continuous(
    df,
    train_cells,
    available_cells,
    transform=None,
):
    x = df.astype(np.float32).copy()

    if transform == "cnv":
        x = np.log2(
            1.0 + x.clip(lower=0)
        )

    train_available = [
        c for c in train_cells
        if c in available_cells
    ]

    train = x.loc[train_available]

    median = train.median(axis=0).fillna(0.0)

    mean = train.mean(axis=0).fillna(0.0)
    std = train.std(axis=0).fillna(1.0)

    std = std.mask(std < 1e-8, 1.0)

    x = x.fillna(median)
    x = (x - mean) / std

    # Completely unavailable modality → true zero vector.
    missing_cells = [
        c for c in x.index
        if c not in available_cells
    ]

    if missing_cells:
        x.loc[missing_cells] = 0.0

    return x.astype(np.float32)


def main(seed):

    # ========================================================
    # LOAD DATA
    # ========================================================

    df = pd.read_parquet(DATA).copy()

    print("Rows:", f"{len(df):,}")

    # ========================================================
    # EXACT DRUG-PAIR SPLIT
    # ========================================================

    pairs = (
        df[
            ["drug_a_key", "drug_b_key"]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    rng = np.random.default_rng(seed)

    order = rng.permutation(len(pairs))

    n_val = int(
        round(0.20 * len(pairs))
    )

    val_pairs = {
        tuple(x)
        for x in pairs.iloc[
            order[:n_val]
        ].to_numpy()
    }

    is_val = [
        (a, b) in val_pairs
        for a, b in zip(
            df["drug_a_key"],
            df["drug_b_key"],
        )
    ]

    df["split"] = np.where(
        is_val,
        "val",
        "train",
    )

    split_path = (
        OUT /
        f"mocktail_v4_split_seed{seed}.parquet"
    )

    df.to_parquet(
        split_path,
        index=False,
    )

    print(
        "Train:",
        f"{(df['split']=='train').sum():,}"
    )

    print(
        "Val:",
        f"{(df['split']=='val').sum():,}"
    )

    print(
        "Train pairs:",
        df.loc[
            df["split"] == "train",
            ["drug_a_key","drug_b_key"]
        ].drop_duplicates().shape[0]
    )

    print(
        "Val pairs:",
        df.loc[
            df["split"] == "val",
            ["drug_a_key","drug_b_key"]
        ].drop_duplicates().shape[0]
    )

    train_cells = set(
        df.loc[
            df["split"] == "train",
            "depmap_id"
        ].astype(str)
    )

    # ========================================================
    # MASKS
    # ========================================================

    masks = pd.read_parquet(
        OMICS / "modality_masks.parquet"
    )

    masks.index = masks.index.astype(str)

    # ========================================================
    # PREPROCESS EACH MODALITY
    # ========================================================

    for name in [
        "expression",
        "cnv",
        "mutation",
        "crispr",
    ]:

        x = pd.read_parquet(
            OMICS / f"{name}_aligned.parquet"
        )

        x.index = x.index.astype(str)

        available = set(
            masks.index[
                masks[f"has_{name}"]
            ]
        )

        # -----------------------------------------------
        # Separate Core29 from general biology
        # -----------------------------------------------

        core_cols = [
            g for g in CORE29
            if g in x.columns
        ]

        general_cols = [
            g for g in x.columns
            if g not in CORE29
        ]

        general = x[
            general_cols
        ].copy()

        # -----------------------------------------------
        # General multi-omics branch
        # -----------------------------------------------

        if name == "mutation":

            processed = (
                general
                .fillna(0)
                .astype(np.float32)
            )

            missing = [
                c for c in processed.index
                if c not in available
            ]

            if missing:
                processed.loc[missing] = 0.0

        else:

            processed = preprocess_continuous(
                general,
                train_cells,
                available,
                transform=(
                    "cnv"
                    if name == "cnv"
                    else None
                ),
            )

        processed.to_parquet(
            OUT /
            f"{name}_general_seed{seed}.parquet"
        )

        print(
            name,
            "general:",
            processed.shape,
        )

        # -----------------------------------------------
        # Save Core29 modality separately for later
        # resistance upgrades
        # -----------------------------------------------

        if core_cols:

            core = x.reindex(
                columns=CORE29
            )

            if name == "mutation":

                core_processed = (
                    core
                    .fillna(0)
                    .astype(np.float32)
                )

                missing = [
                    c for c in core_processed.index
                    if c not in available
                ]

                if missing:
                    core_processed.loc[missing] = 0.0

            else:

                core_processed = preprocess_continuous(
                    core,
                    train_cells,
                    available,
                    transform=(
                        "cnv"
                        if name == "cnv"
                        else None
                    ),
                )

            core_processed.to_parquet(
                OUT /
                f"core29_{name}_seed{seed}.parquet"
            )

    print()
    print("Saved:", split_path)
    print("DONE")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    main(args.seed)
