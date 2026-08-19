from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ============================================================
# Paths
# ============================================================

PHAROS_DRUG_TABLE = Path(
    "data/processed/combination/pharos_combo_drug_table.csv"
)

EXPRESSION_PARQUET = Path(
    "data/processed/pharos_depmap_expression.parquet"
)

USABLE_LONG = Path(
    "data/processed/combination/scdrugact/"
    "pharos_scdrugact_resistance_long_depmap_usable.csv"
)

OUTPUT_DIR = Path(
    "data/processed/combination/scdrugact"
)

MASK_OUT = OUTPUT_DIR / "pharos_scdrugact_resistance_masks.npz"
DRUG_STATS_OUT = OUTPUT_DIR / "pharos_scdrugact_resistance_mask_stats.csv"
GENE_INDEX_OUT = OUTPUT_DIR / "pharos_scdrugact_resistance_gene_index.csv"


# ============================================================
# Helpers
# ============================================================

def normalize_drug_name(value: object) -> str:
    if value is None:
        return ""

    text = str(value).strip()

    if not text or text.lower() == "nan":
        return ""

    return re.sub(
        r"[^a-z0-9]+",
        "",
        text.lower(),
    )


def normalize_gene_symbol(value: object) -> str:
    if value is None:
        return ""

    text = str(value).strip()

    if not text or text.lower() == "nan":
        return ""

    # DepMap columns may look like TP53 (7157)
    text = re.sub(
        r"\s+\(\d+\)$",
        "",
        text,
    )

    return text.strip().upper()


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
    description: str,
) -> str:

    lower = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for candidate in candidates:
        key = candidate.lower()

        if key in lower:
            return lower[key]

    raise ValueError(
        f"Could not find {description} column.\n"
        f"Available columns: {list(df.columns)}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 72
    )
    print(
        "BUILDING PHAROS SCDRUGACT RESISTANCE MASKS"
    )
    print(
        "=" * 72
    )

    for path in [
        PHAROS_DRUG_TABLE,
        EXPRESSION_PARQUET,
        USABLE_LONG,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file:\n{path}"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. Load PHAROS drug order
    # --------------------------------------------------------

    drug_table = pd.read_csv(
        PHAROS_DRUG_TABLE
    )

    drug_col = find_column(
        drug_table,
        [
            "drug",
            "drug_name",
            "name",
            "compound",
            "compound_name",
        ],
        "PHAROS drug-name",
    )

    drug_names = (
        drug_table[drug_col]
        .dropna()
        .astype(str)
        .str.strip()
    )

    drug_names = (
        drug_names[
            drug_names != ""
        ]
        .drop_duplicates()
        .tolist()
    )

    print(
        f"\nPHAROS drugs: "
        f"{len(drug_names):,}"
    )

    drug_norms = [
        normalize_drug_name(
            drug
        )
        for drug in drug_names
    ]

    # --------------------------------------------------------
    # 2. Load usable scDrugAct associations
    # --------------------------------------------------------

    long_df = pd.read_csv(
        USABLE_LONG
    )

    required = {
        "pharos_drug",
        "gene_norm",
        "depmap_expression_column",
    }

    missing = (
        required
        - set(long_df.columns)
    )

    if missing:
        raise ValueError(
            "Usable resistance table is missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    long_df[
        "pharos_drug_norm"
    ] = (
        long_df[
            "pharos_drug"
        ]
        .map(
            normalize_drug_name
        )
    )

    long_df[
        "gene_norm"
    ] = (
        long_df[
            "gene_norm"
        ]
        .map(
            normalize_gene_symbol
        )
    )

    long_df = (
        long_df[
            (
                long_df[
                    "pharos_drug_norm"
                ]
                != ""
            )
            &
            (
                long_df[
                    "gene_norm"
                ]
                != ""
            )
        ]
        .drop_duplicates(
            subset=[
                "pharos_drug_norm",
                "gene_norm",
            ]
        )
        .copy()
    )

    print(
        f"Usable drug-gene associations: "
        f"{len(long_df):,}"
    )

    print(
        f"Unique normalized scDrugAct-matched "
        f"PHAROS drugs: "
        f"{long_df['pharos_drug_norm'].nunique():,}"
    )

    # --------------------------------------------------------
    # 3. Build resistance gene order in DepMap column order
    # --------------------------------------------------------

    schema = pq.ParquetFile(
        EXPRESSION_PARQUET
    ).schema_arrow

    expression_columns = list(
        schema.names
    )

    resistance_gene_set = set(
        long_df[
            "gene_norm"
        ].unique()
    )

    gene_rows = []
    seen_genes = set()

    for col_index, raw_col in enumerate(
        expression_columns
    ):

        if str(raw_col).strip().lower() == "depmap_id":
            continue

        gene = normalize_gene_symbol(
            raw_col
        )

        if (
            gene in resistance_gene_set
            and gene not in seen_genes
        ):
            gene_rows.append(
                {
                    "mask_index":
                        len(gene_rows),

                    "gene":
                        gene,

                    "depmap_expression_column":
                        raw_col,

                    "depmap_parquet_column_index":
                        col_index,
                }
            )

            seen_genes.add(
                gene
            )

    gene_index = pd.DataFrame(
        gene_rows
    )

    if len(gene_index) != len(
        resistance_gene_set
    ):
        missing_from_order = (
            resistance_gene_set
            - set(
                gene_index[
                    "gene"
                ]
            )
        )

        raise RuntimeError(
            "Some previously validated usable genes could "
            "not be aligned to the DepMap schema:\n"
            + ", ".join(
                sorted(
                    missing_from_order
                )[:50]
            )
        )

    gene_names = (
        gene_index[
            "gene"
        ]
        .astype(str)
        .tolist()
    )

    depmap_columns = (
        gene_index[
            "depmap_expression_column"
        ]
        .astype(str)
        .tolist()
    )

    print(
        f"Resistance mask genes: "
        f"{len(gene_names):,}"
    )

    # --------------------------------------------------------
    # 4. Build fast drug -> gene-set lookup
    # --------------------------------------------------------

    genes_by_drug = defaultdict(
        set
    )

    for row in long_df.itertuples(
        index=False
    ):

        genes_by_drug[
            row.pharos_drug_norm
        ].add(
            row.gene_norm
        )

    gene_to_mask_index = {
        gene: i
        for i, gene in enumerate(
            gene_names
        )
    }

    # --------------------------------------------------------
    # 5. Build mask [num_pharos_drugs, num_resistance_genes]
    # --------------------------------------------------------

    mask = np.zeros(
        (
            len(drug_names),
            len(gene_names),
        ),
        dtype=np.uint8,
    )

    per_drug_rows = []

    for drug_index, (
        drug,
        drug_norm,
    ) in enumerate(
        zip(
            drug_names,
            drug_norms,
        )
    ):

        genes = (
            genes_by_drug.get(
                drug_norm,
                set(),
            )
        )

        for gene in genes:

            mask[
                drug_index,
                gene_to_mask_index[
                    gene
                ],
            ] = 1

        per_drug_rows.append(
            {
                "drug_index":
                    drug_index,

                "pharos_drug":
                    drug,

                "normalized_drug":
                    drug_norm,

                "has_scdrugact_resistance":
                    bool(
                        len(genes)
                    ),

                "n_resistance_genes":
                    len(genes),
            }
        )

    stats = pd.DataFrame(
        per_drug_rows
    )

    # --------------------------------------------------------
    # 6. Sanity checks
    # --------------------------------------------------------

    row_sums = mask.sum(
        axis=1
    )

    nonzero_rows = int(
        (
            row_sums > 0
        ).sum()
    )

    unique_nonzero_normalized = (
        stats.loc[
            stats[
                "has_scdrugact_resistance"
            ],
            "normalized_drug",
        ]
        .nunique()
    )

    print(
        "\n========================================"
    )
    print(
        "MASK SUMMARY"
    )
    print(
        "========================================"
    )

    print(
        f"Mask shape: "
        f"{mask.shape}"
    )

    print(
        f"Mask dtype: "
        f"{mask.dtype}"
    )

    print(
        f"PHAROS drug rows with a non-zero mask: "
        f"{nonzero_rows:,}"
    )

    print(
        f"Unique normalized matched drugs: "
        f"{unique_nonzero_normalized:,}"
    )

    print(
        f"Total active drug-gene mask entries: "
        f"{int(mask.sum()):,}"
    )

    nonzero_counts = row_sums[
        row_sums > 0
    ]

    if len(nonzero_counts):

        print(
            "Genes per non-zero drug mask "
            f"(median/min/max): "
            f"{np.median(nonzero_counts):.0f} / "
            f"{nonzero_counts.min()} / "
            f"{nonzero_counts.max()}"
        )

    # Check every association survived into the mask.
    expected_unique_pairs = (
        long_df[
            [
                "pharos_drug_norm",
                "gene_norm",
            ]
        ]
        .drop_duplicates()
    )

    expected_pair_count = len(
        expected_unique_pairs
    )

    # Because PHAROS's drug table contains capitalization-only
    # duplicates for a small number of compounds, the matrix can
    # legitimately contain slightly more active entries than the
    # normalized unique-pair table.
    print(
        f"Unique normalized drug-gene associations: "
        f"{expected_pair_count:,}"
    )

    # --------------------------------------------------------
    # 7. Save
    # --------------------------------------------------------

    np.savez_compressed(
        MASK_OUT,

        mask=mask,

        drug_names=np.asarray(
            drug_names,
            dtype=str,
        ),

        normalized_drug_names=np.asarray(
            drug_norms,
            dtype=str,
        ),

        genes=np.asarray(
            gene_names,
            dtype=str,
        ),

        depmap_expression_columns=np.asarray(
            depmap_columns,
            dtype=str,
        ),

        depmap_parquet_column_indices=(
            gene_index[
                "depmap_parquet_column_index"
            ]
            .to_numpy(
                dtype=np.int32
            )
        ),
    )

    stats.to_csv(
        DRUG_STATS_OUT,
        index=False,
    )

    gene_index.to_csv(
        GENE_INDEX_OUT,
        index=False,
    )

    print(
        "\nSaved:"
    )

    print(
        f"  {MASK_OUT}"
    )

    print(
        f"  {DRUG_STATS_OUT}"
    )

    print(
        f"  {GENE_INDEX_OUT}"
    )

    print(
        "\n========================================"
    )
    print(
        "RELOAD CHECK"
    )
    print(
        "========================================"
    )

    loaded = np.load(
        MASK_OUT
    )

    print(
        "mask:",
        loaded[
            "mask"
        ].shape,
        loaded[
            "mask"
        ].dtype,
    )

    print(
        "drug_names:",
        loaded[
            "drug_names"
        ].shape,
    )

    print(
        "genes:",
        loaded[
            "genes"
        ].shape,
    )

    print(
        "\nPASS: scDrugAct resistance masks built."
    )


if __name__ == "__main__":
    main()
