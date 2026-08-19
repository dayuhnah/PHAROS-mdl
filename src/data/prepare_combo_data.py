from pathlib import Path
import re

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

COMBO_PATH = Path(
    "data/raw/combination/drugcombs_scored.csv"
)

MODEL_PATH = Path(
    "data/raw/depmap/Model.csv"
)

EXPRESSION_PATH = Path(
    "data/processed/pharos_depmap_expression.parquet"
)

OUTPUT_DIR = Path(
    "data/processed/combination"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "pharos_combo_clean.parquet"
)

CELL_MAPPING_PATH = (
    OUTPUT_DIR
    / "drugcomb_depmap_cell_mapping.csv"
)

UNMATCHED_CELL_PATH = (
    OUTPUT_DIR
    / "drugcomb_unmatched_cells.csv"
)

DRUG_LIST_PATH = (
    OUTPUT_DIR
    / "pharos_combo_unique_drugs.csv"
)


# ============================================================
# Normalisation
# ============================================================

def normalize_cell_name(value):

    if pd.isna(value):
        return None

    return re.sub(
        r"[^A-Z0-9]",
        "",
        str(value).upper(),
    )


# ============================================================
# Build unambiguous DrugComb -> DepMap mapping
# ============================================================

def build_cell_mapping(
    model,
    expression_ids,
):

    usable_models = model[
        model["ModelID"]
        .astype(str)
        .isin(expression_ids)
    ].copy()

    # key -> set of possible DepMap IDs
    candidate_map = {}

    for _, row in usable_models.iterrows():

        depmap_id = str(
            row["ModelID"]
        )

        candidate_names = [
            row.get(
                "CellLineName"
            ),
            row.get(
                "StrippedCellLineName"
            ),
        ]

        for name in candidate_names:

            key = normalize_cell_name(
                name
            )

            if not key:
                continue

            if key not in candidate_map:
                candidate_map[key] = set()

            candidate_map[key].add(
                depmap_id
            )

    # Keep only unambiguous mappings.
    clean_map = {
        key: next(iter(ids))
        for key, ids
        in candidate_map.items()
        if len(ids) == 1
    }

    ambiguous = {
        key: ids
        for key, ids
        in candidate_map.items()
        if len(ids) > 1
    }

    return (
        clean_map,
        ambiguous,
    )


# ============================================================
# Canonicalise drug pair
# ============================================================

def canonicalize_drug_pair(df):

    drug1 = (
        df["Drug1"]
        .astype(str)
        .str.strip()
    )

    drug2 = (
        df["Drug2"]
        .astype(str)
        .str.strip()
    )

    # Use uppercase only for ordering.
    key1 = drug1.str.upper()
    key2 = drug2.str.upper()

    first_is_a = (
        key1 <= key2
    )

    df["drug_a"] = np.where(
        first_is_a,
        drug1,
        drug2,
    )

    df["drug_b"] = np.where(
        first_is_a,
        drug2,
        drug1,
    )

    return df


# ============================================================
# Main
# ============================================================

def main():

    print(
        "========================================"
    )

    print(
        "PHAROS-COMBO DATA PREPARATION"
    )

    print(
        "========================================"
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    combo = pd.read_csv(
        COMBO_PATH,
        low_memory=False,
    )

    model = pd.read_csv(
        MODEL_PATH,
        low_memory=False,
    )

    expression = pd.read_parquet(
        EXPRESSION_PATH
    )

    print(
        f"\nRaw DrugComb rows: "
        f"{len(combo):,}"
    )

    print(
        f"DrugComb cell lines: "
        f"{combo['Cell line'].nunique():,}"
    )

    # --------------------------------------------------------
    # PHAROS expression IDs
    # --------------------------------------------------------

    expression_ids = set(
        expression[
            "depmap_id"
        ]
        .astype(str)
    )

    print(
        f"PHAROS expression cells: "
        f"{len(expression_ids):,}"
    )

    # --------------------------------------------------------
    # Build mapping
    # --------------------------------------------------------

    (
        cell_map,
        ambiguous,
    ) = build_cell_mapping(
        model,
        expression_ids,
    )

    if ambiguous:

        print(
            "\nWarning:"
            f" {len(ambiguous)} "
            "ambiguous cell-name keys "
            "were ignored."
        )

    combo[
        "_cell_key"
    ] = combo[
        "Cell line"
    ].map(
        normalize_cell_name
    )

    combo[
        "depmap_id"
    ] = combo[
        "_cell_key"
    ].map(
        cell_map
    )

    # --------------------------------------------------------
    # Save matching diagnostics
    # --------------------------------------------------------

    unique_cells = (
        combo[
            [
                "Cell line",
                "_cell_key",
                "depmap_id",
            ]
        ]
        .drop_duplicates()
    )

    matched_cells = (
        unique_cells[
            unique_cells[
                "depmap_id"
            ].notna()
        ]
        .copy()
    )

    unmatched_cells = (
        unique_cells[
            unique_cells[
                "depmap_id"
            ].isna()
        ]
        .copy()
    )

    matched_cells.to_csv(
        CELL_MAPPING_PATH,
        index=False,
    )

    unmatched_cells.to_csv(
        UNMATCHED_CELL_PATH,
        index=False,
    )

    print(
        f"\nMatched DrugComb cells: "
        f"{len(matched_cells)}"
    )

    print(
        f"Unmatched DrugComb cells: "
        f"{len(unmatched_cells)}"
    )

    # --------------------------------------------------------
    # Keep only PHAROS-compatible cells
    # --------------------------------------------------------

    combo = combo[
        combo[
            "depmap_id"
        ].notna()
    ].copy()

    print(
        f"\nRows after cell matching: "
        f"{len(combo):,}"
    )

    # --------------------------------------------------------
    # Remove invalid labels
    # --------------------------------------------------------

    combo = combo[
        np.isfinite(
            combo["ZIP"]
        )
    ].copy()

    # --------------------------------------------------------
    # Remove pathological ZIP outliers
    #
    # We observed only 72 rows above this
    # threshold after DepMap filtering.
    # --------------------------------------------------------

    before_outlier = len(
        combo
    )

    combo = combo[
        combo["ZIP"].abs()
        <= 100
    ].copy()

    removed_outliers = (
        before_outlier
        - len(combo)
    )

    print(
        f"Removed |ZIP| > 100: "
        f"{removed_outliers:,}"
    )

    print(
        f"Rows after ZIP filtering: "
        f"{len(combo):,}"
    )

    # --------------------------------------------------------
    # Canonical Drug A / Drug B ordering
    # --------------------------------------------------------

    combo = canonicalize_drug_pair(
        combo
    )

    # --------------------------------------------------------
    # Aggregate repeated measurements
    #
    # Same:
    # Drug A + Drug B + cancer cell
    #
    # should have ONE regression target.
    #
    # Median is robust to noisy replicate values.
    # --------------------------------------------------------

    grouped = (
        combo
        .groupby(
            [
                "drug_a",
                "drug_b",
                "depmap_id",
            ],
            as_index=False,
        )
        .agg(
            zip_score=(
                "ZIP",
                "median",
            ),

            bliss_score=(
                "Bliss",
                "median",
            ),

            loewe_score=(
                "Loewe",
                "median",
            ),

            hsa_score=(
                "HSA",
                "median",
            ),

            replicate_count=(
                "ID",
                "count",
            ),

            original_cell_name=(
                "Cell line",
                "first",
            ),
        )
    )

    print(
        "\n========================================"
    )

    print(
        "AFTER PAIR-CELL AGGREGATION"
    )

    print(
        "========================================"
    )

    print(
        f"Unique pair-cell rows: "
        f"{len(grouped):,}"
    )

    print(
        f"Unique drugs A: "
        f"{grouped['drug_a'].nunique():,}"
    )

    print(
        f"Unique drugs B: "
        f"{grouped['drug_b'].nunique():,}"
    )

    all_drugs = sorted(
        set(
            grouped[
                "drug_a"
            ].astype(str)
        )
        |
        set(
            grouped[
                "drug_b"
            ].astype(str)
        )
    )

    print(
        f"Unique drugs overall: "
        f"{len(all_drugs):,}"
    )

    print(
        f"Unique DepMap cells: "
        f"{grouped['depmap_id'].nunique():,}"
    )

    print(
        "\nZIP statistics:"
    )

    print(
        grouped[
            "zip_score"
        ].describe()
    )

    print(
        "\nReplicate-count statistics:"
    )

    print(
        grouped[
            "replicate_count"
        ].describe()
    )

    # --------------------------------------------------------
    # Save processed combinations
    # --------------------------------------------------------

    grouped.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Save unique drug list
    # --------------------------------------------------------

    pd.DataFrame(
        {
            "drug_name":
                all_drugs
        }
    ).to_csv(
        DRUG_LIST_PATH,
        index=False,
    )

    print(
        "\n========================================"
    )

    print(
        "SAVED"
    )

    print(
        "========================================"
    )

    print(
        f"Processed combinations:\n"
        f"{OUTPUT_PATH}"
    )

    print(
        f"\nCell mapping:\n"
        f"{CELL_MAPPING_PATH}"
    )

    print(
        f"\nUnmatched cells:\n"
        f"{UNMATCHED_CELL_PATH}"
    )

    print(
        f"\nUnique drugs:\n"
        f"{DRUG_LIST_PATH}"
    )


if __name__ == "__main__":
    main()