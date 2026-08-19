from __future__ import annotations

import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq
except Exception as e:
    raise ImportError(
        "This script needs pyarrow, which is normally already installed "
        "if PHAROS can read parquet files."
    ) from e


# ============================================================
# Paths
# ============================================================

DATA_ROOT = Path("data")

SCDRUGACT_LONG = Path(
    "data/processed/combination/scdrugact/"
    "pharos_scdrugact_resistance_long.csv"
)

SCDRUGACT_MAPPING = Path(
    "data/processed/combination/scdrugact/"
    "pharos_scdrugact_drug_mapping.csv"
)

OUTPUT_DIR = Path(
    "data/processed/combination/scdrugact"
)

GENE_MAPPING_OUT = OUTPUT_DIR / "scdrugact_depmap_gene_mapping.csv"
DRUG_GENE_STATS_OUT = OUTPUT_DIR / "scdrugact_depmap_per_drug_gene_stats.csv"
USABLE_LONG_OUT = OUTPUT_DIR / "pharos_scdrugact_resistance_long_depmap_usable.csv"
SUMMARY_OUT = OUTPUT_DIR / "scdrugact_depmap_gene_coverage_summary.csv"


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
    """
    Normalize gene symbols conservatively.

    Handles common DepMap-style headers such as:
      TP53 (7157) -> TP53
      ABCB1       -> ABCB1
    """
    if value is None:
        return ""

    text = str(value).strip()

    if not text or text.lower() == "nan":
        return ""

    # Remove a trailing numeric Entrez ID in parentheses.
    text = re.sub(
        r"\s+\(\d+\)$",
        "",
        text,
    )

    return text.strip().upper()


def auto_find_expression_parquet() -> Path:
    """
    Find a wide parquet table under data/ that looks like the
    DepMap expression matrix. We inspect parquet schemas only,
    so this does not load every candidate into memory.
    """

    candidates = []

    for path in DATA_ROOT.rglob("*.parquet"):

        try:
            schema = pq.ParquetFile(
                path
            ).schema_arrow

            ncols = len(
                schema.names
            )

        except Exception:
            continue

        # The known PHAROS expression matrix is extremely wide
        # (~19k genes), while response/combination tables are narrow.
        if ncols >= 5000:
            candidates.append(
                (
                    ncols,
                    path,
                )
            )

    if not candidates:
        raise FileNotFoundError(
            "Could not automatically find a wide DepMap expression "
            "parquet under data/. Expected a parquet with >= 5,000 columns."
        )

    candidates.sort(
        reverse=True,
        key=lambda x: x[0],
    )

    print(
        "\nWide parquet candidates:"
    )

    for ncols, path in candidates[:10]:
        print(
            f"  {ncols:,} columns  {path}"
        )

    # Use the widest candidate. For this project this should be
    # the 19,206-column DepMap expression table.
    return candidates[0][1]


def detect_identifier_columns(
    columns: list[str],
) -> set[str]:
    """
    Exclude obvious cell-line/model identifier columns from gene matching.
    """
    identifier_names = {
        "depmap_id",
        "modelid",
        "model_id",
        "cell_line",
        "cellline",
        "cell_line_name",
        "stripped_cell_line_name",
        "index",
    }

    out = set()

    for col in columns:
        key = (
            str(col)
            .strip()
            .lower()
            .replace(" ", "_")
        )

        if key in identifier_names:
            out.add(
                col
            )

    return out


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 72
    )
    print(
        "SCDRUGACT × DEPMAP GENE COVERAGE"
    )
    print(
        "=" * 72
    )

    if not SCDRUGACT_LONG.exists():
        raise FileNotFoundError(
            f"Missing:\n{SCDRUGACT_LONG}"
        )

    if not SCDRUGACT_MAPPING.exists():
        raise FileNotFoundError(
            f"Missing:\n{SCDRUGACT_MAPPING}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load PHAROS-specific scDrugAct resistance associations
    # --------------------------------------------------------

    long_df = pd.read_csv(
        SCDRUGACT_LONG
    )

    required = {
        "pharos_drug",
        "scdrugact_drug",
        "gene",
    }

    missing = (
        required
        - set(long_df.columns)
    )

    if missing:
        raise ValueError(
            "Resistance long table is missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    long_df["gene_norm"] = (
        long_df["gene"]
        .map(
            normalize_gene_symbol
        )
    )

    # Collapse capitalization-only duplicates such as
    # DASATINIB/Dasatinib and TRICIRIBINE/Triciribine.
    long_df["pharos_drug_norm"] = (
        long_df["pharos_drug"]
        .map(
            normalize_drug_name
        )
    )

    long_df = long_df[
        long_df["gene_norm"] != ""
    ].copy()

    scdrugact_vocab = sorted(
        long_df["gene_norm"]
        .unique()
        .tolist()
    )

    print(
        f"\nPHAROS-specific scDrugAct gene vocabulary: "
        f"{len(scdrugact_vocab):,}"
    )

    # --------------------------------------------------------
    # Locate expression matrix
    # --------------------------------------------------------

    expression_path = (
        auto_find_expression_parquet()
    )

    print(
        f"\nSelected expression parquet:\n"
        f"  {expression_path}"
    )

    schema = pq.ParquetFile(
        expression_path
    ).schema_arrow

    columns = list(
        schema.names
    )

    print(
        f"Expression columns: "
        f"{len(columns):,}"
    )

    id_cols = detect_identifier_columns(
        columns
    )

    if id_cols:
        print(
            "Identifier/non-gene columns ignored:"
        )

        for col in sorted(
            id_cols
        ):
            print(
                f"  {col}"
            )

    # Build normalized-gene -> raw-expression-column mapping.
    gene_to_columns = defaultdict(
        list
    )

    for col in columns:

        if col in id_cols:
            continue

        gene = normalize_gene_symbol(
            col
        )

        if gene:
            gene_to_columns[
                gene
            ].append(
                col
            )

    depmap_gene_set = set(
        gene_to_columns
    )

    print(
        f"Normalized DepMap gene symbols: "
        f"{len(depmap_gene_set):,}"
    )

    # --------------------------------------------------------
    # Gene intersection
    # --------------------------------------------------------

    scdrugact_set = set(
        scdrugact_vocab
    )

    usable = (
        scdrugact_set
        & depmap_gene_set
    )

    missing_genes = (
        scdrugact_set
        - depmap_gene_set
    )

    coverage = (
        len(usable)
        / len(scdrugact_set)
        if scdrugact_set
        else 0.0
    )

    print(
        "\n========================================"
    )
    print(
        "GENE COVERAGE"
    )
    print(
        "========================================"
    )

    print(
        f"scDrugAct PHAROS vocabulary: "
        f"{len(scdrugact_set):,}"
    )

    print(
        f"Found in DepMap: "
        f"{len(usable):,}"
    )

    print(
        f"Missing from DepMap: "
        f"{len(missing_genes):,}"
    )

    print(
        f"Gene coverage: "
        f"{100.0 * coverage:.2f}%"
    )

    # --------------------------------------------------------
    # Save one row per scDrugAct gene
    # --------------------------------------------------------

    gene_mapping_rows = []

    for gene in sorted(
        scdrugact_set
    ):

        raw_cols = (
            gene_to_columns.get(
                gene,
                [],
            )
        )

        gene_mapping_rows.append(
            {
                "scdrugact_gene":
                    gene,

                "present_in_depmap":
                    bool(raw_cols),

                "depmap_column_count":
                    len(raw_cols),

                "depmap_columns":
                    "|".join(
                        raw_cols
                    ),
            }
        )

    gene_mapping = pd.DataFrame(
        gene_mapping_rows
    )

    gene_mapping.to_csv(
        GENE_MAPPING_OUT,
        index=False,
    )

    # --------------------------------------------------------
    # Filter long table to usable genes
    # --------------------------------------------------------

    usable_long = long_df[
        long_df["gene_norm"]
        .isin(
            usable
        )
    ].copy()

    # Resolve exact expression column. Normally each normalized
    # symbol maps to one column; if there are duplicates we use
    # the first and record that fact in gene_mapping.
    usable_long[
        "depmap_expression_column"
    ] = usable_long[
        "gene_norm"
    ].map(
        lambda gene:
        gene_to_columns[
            gene
        ][0]
    )

    usable_long.to_csv(
        USABLE_LONG_OUT,
        index=False,
    )

    # --------------------------------------------------------
    # Per-drug usable resistance-gene counts
    # --------------------------------------------------------

    original_counts = (
        long_df
        .groupby(
            "pharos_drug_norm"
        )[
            "gene_norm"
        ]
        .nunique()
        .rename(
            "scdrugact_genes"
        )
    )

    usable_counts = (
        usable_long
        .groupby(
            "pharos_drug_norm"
        )[
            "gene_norm"
        ]
        .nunique()
        .rename(
            "depmap_usable_genes"
        )
    )

    stats = (
        pd.concat(
            [
                original_counts,
                usable_counts,
            ],
            axis=1,
        )
        .fillna(
            0
        )
        .reset_index()
    )

    representative_names = (
        long_df
        .groupby(
            "pharos_drug_norm"
        )[
            "pharos_drug"
        ]
        .agg(
            lambda values: sorted(
                set(
                    map(
                        str,
                        values,
                    )
                ),
                key=lambda x: (
                    x.lower(),
                    x,
                ),
            )[0]
        )
    )

    stats.insert(
        1,
        "pharos_drug",
        stats[
            "pharos_drug_norm"
        ].map(
            representative_names
        ),
    )

    stats[
        "scdrugact_genes"
    ] = stats[
        "scdrugact_genes"
    ].astype(
        int
    )

    stats[
        "depmap_usable_genes"
    ] = stats[
        "depmap_usable_genes"
    ].astype(
        int
    )

    stats[
        "usable_fraction"
    ] = (
        stats[
            "depmap_usable_genes"
        ]
        / stats[
            "scdrugact_genes"
        ].replace(
            0,
            np.nan,
        )
    )

    stats[
        "usable_percent"
    ] = (
        100.0
        * stats[
            "usable_fraction"
        ]
    )

    stats.to_csv(
        DRUG_GENE_STATS_OUT,
        index=False,
    )

    usable_counts_array = (
        stats[
            "depmap_usable_genes"
        ]
        .to_numpy(
            dtype=float
        )
    )

    print(
        "\n========================================"
    )
    print(
        "PER-DRUG USABLE GENE COUNTS"
    )
    print(
        "========================================"
    )

    print(
        f"Matched PHAROS drugs with "
        f"scDrugAct associations: "
        f"{len(stats):,}"
    )

    print(
        "Usable genes/drug "
        f"(median/min/max): "
        f"{np.median(usable_counts_array):.0f} / "
        f"{usable_counts_array.min():.0f} / "
        f"{usable_counts_array.max():.0f}"
    )

    n_zero = int(
        (
            stats[
                "depmap_usable_genes"
            ]
            == 0
        ).sum()
    )

    print(
        f"Matched drugs with zero usable "
        f"DepMap genes: "
        f"{n_zero:,}"
    )

    # --------------------------------------------------------
    # Compact summary file
    # --------------------------------------------------------

    summary = pd.DataFrame(
        [
            {
                "scdrugact_pharos_gene_vocab":
                    len(scdrugact_set),

                "depmap_matched_genes":
                    len(usable),

                "depmap_missing_genes":
                    len(missing_genes),

                "gene_coverage_fraction":
                    coverage,

                "gene_coverage_percent":
                    100.0 * coverage,

                "matched_pharos_drugs":
                    len(stats),

                "median_usable_genes_per_drug":
                    float(
                        np.median(
                            usable_counts_array
                        )
                    ),

                "min_usable_genes_per_drug":
                    int(
                        usable_counts_array.min()
                    ),

                "max_usable_genes_per_drug":
                    int(
                        usable_counts_array.max()
                    ),

                "matched_drugs_zero_usable_genes":
                    n_zero,

                "expression_parquet":
                    str(
                        expression_path
                    ),
            }
        ]
    )

    summary.to_csv(
        SUMMARY_OUT,
        index=False,
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print(
        "\n========================================"
    )
    print(
        "FINAL SUMMARY"
    )
    print(
        "========================================"
    )

    print(
        f"scDrugAct PHAROS genes: "
        f"{len(scdrugact_set):,}"
    )

    print(
        f"Usable in DepMap: "
        f"{len(usable):,}"
    )

    print(
        f"Coverage: "
        f"{100.0 * coverage:.2f}%"
    )

    print(
        f"Median usable genes/drug: "
        f"{np.median(usable_counts_array):.0f}"
    )

    print(
        f"Matched drugs with zero usable genes: "
        f"{n_zero}"
    )

    print(
        "\nSaved:"
    )

    print(
        f"  {GENE_MAPPING_OUT}"
    )

    print(
        f"  {DRUG_GENE_STATS_OUT}"
    )

    print(
        f"  {USABLE_LONG_OUT}"
    )

    print(
        f"  {SUMMARY_OUT}"
    )

    print(
        "\nIf coverage is strong, the next step is "
        "to create fixed-index resistance masks for "
        "each PHAROS drug and align them to the "
        "DepMap expression tensor."
    )


if __name__ == "__main__":
    main()
