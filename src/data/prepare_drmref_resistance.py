from __future__ import annotations

import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset import clean_gene_name


# ============================================================
# Paths
# ============================================================

DRMREF_URL = (
    "https://ccsm.uth.edu/DRMref/"
    "table_summary/gene_summary.txt"
)

RAW_DIR = Path(
    "data/raw/resistance/drmref"
)

OUT_DIR = Path(
    "data/processed/combination/drmref"
)

RAW_GENE_SUMMARY = (
    RAW_DIR
    / "gene_summary.txt"
)

COMBO_PATH = Path(
    "data/processed/combination/"
    "pharos_combo_local_smiles.parquet"
)

EXPRESSION_PATH = Path(
    "data/processed/"
    "pharos_depmap_expression.parquet"
)


# ============================================================
# Helpers
# ============================================================

def normalize_drug_name(
    value: object,
) -> str:
    """
    Normalize names for matching only.

    Examples:
        DASATINIB -> dasatinib
        PLX-4720  -> plx4720
        KDM5-C70  -> kdm5c70
    """
    if value is None:
        return ""

    text = str(
        value
    ).strip()

    if (
        not text
        or text.lower() == "nan"
    ):
        return ""

    return re.sub(
        r"[^a-z0-9]+",
        "",
        text.lower(),
    )


def download_file(
    url: str,
    path: Path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists():
        print(
            f"Using existing DRMref file: "
            f"{path}"
        )
        return

    print(
        "Downloading DRMref resistance DEG table..."
    )
    print(
        f"Source: {url}"
    )

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent":
                "Mozilla/5.0 PHAROS-FYP",
        },
    )

    with (
        urllib.request.urlopen(
            request
        ) as response,
        open(
            path,
            "wb",
        ) as f,
    ):
        while True:
            chunk = response.read(
                1024 * 1024
            )

            if not chunk:
                break

            f.write(
                chunk
            )

    print(
        f"Saved: {path}"
    )


def is_combination_regimen(
    regimen: str,
) -> bool:
    """
    DRMref includes both single-agent and combination-treatment
    resistance signatures.

    For the first PHAROS integration we only use SINGLE-AGENT
    resistance signatures as drug-specific priors.

    Combination signatures are preserved in a separate CSV so they
    can later be tested as pair-specific priors without incorrectly
    assigning a combination-derived signature to one constituent drug.
    """
    text = str(
        regimen
    ).strip()

    return (
        "+"
        in text
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 72
    )
    print(
        "PREPARING DRMref REAL RESISTANT-CELL SIGNATURES FOR PHAROS"
    )
    print(
        "=" * 72
    )

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Download DRMref
    # --------------------------------------------------------

    download_file(
        DRMREF_URL,
        RAW_GENE_SUMMARY,
    )

    print(
        "\nLoading DRMref gene summary..."
    )

    drmref = pd.read_csv(
        RAW_GENE_SUMMARY,
        sep="\t",
        low_memory=False,
    )

    print(
        f"Raw DRMref DEG rows: "
        f"{len(drmref):,}"
    )

    required = {
        "p_val_adj",
        "avg_log2FC",
        "Gene_symbol",
        "Cell_type",
        "Original_Dataset",
        "Cancer_type_level1",
        "Regimen",
        "Source",
        "Tissue",
        "PMID",
        "Description",
    }

    missing = (
        required
        - set(
            drmref.columns
        )
    )

    if missing:
        raise ValueError(
            "DRMref file is missing required columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    # --------------------------------------------------------
    # Keep real cell-line resistance comparisons only
    # --------------------------------------------------------

    cell_line = drmref[
        drmref[
            "Tissue"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq(
            "cell line"
        )
    ].copy()

    # DRMref paper definition of resistance-related DEGs:
    # p_val_adj < 0.05 and |avg_log2FC| > 0.25.
    cell_line = cell_line[
        pd.to_numeric(
            cell_line[
                "p_val_adj"
            ],
            errors="coerce",
        )
        < 0.05
    ].copy()

    cell_line[
        "avg_log2FC"
    ] = pd.to_numeric(
        cell_line[
            "avg_log2FC"
        ],
        errors="coerce",
    )

    cell_line = cell_line[
        cell_line[
            "avg_log2FC"
        ]
        .abs()
        > 0.25
    ].copy()

    cell_line[
        "Gene_symbol"
    ] = [
        clean_gene_name(
            str(gene)
        )
        for gene
        in cell_line[
            "Gene_symbol"
        ]
    ]

    cell_line[
        "regimen_norm"
    ] = (
        cell_line[
            "Regimen"
        ]
        .map(
            normalize_drug_name
        )
    )

    cell_line[
        "is_combination"
    ] = (
        cell_line[
            "Regimen"
        ]
        .astype(str)
        .map(
            is_combination_regimen
        )
    )

    # Save all real resistant-cell DEGs, including combination
    # regimens, for provenance/future work.
    all_cell_line_path = (
        OUT_DIR
        / "drmref_cell_line_resistance_degs.csv"
    )

    cell_line.to_csv(
        all_cell_line_path,
        index=False,
    )

    combination = cell_line[
        cell_line[
            "is_combination"
        ]
    ].copy()

    single = cell_line[
        ~cell_line[
            "is_combination"
        ]
    ].copy()

    combination_path = (
        OUT_DIR
        / "drmref_combination_regimen_degs.csv"
    )

    combination.to_csv(
        combination_path,
        index=False,
    )

    single_path = (
        OUT_DIR
        / "drmref_single_agent_cell_line_degs.csv"
    )

    single.to_csv(
        single_path,
        index=False,
    )

    print(
        "\nDRMref resistant-cell subset:"
    )

    print(
        f"  Cell-line R-DEGs: "
        f"{len(cell_line):,}"
    )

    print(
        f"  Cell-line datasets: "
        f"{cell_line['Original_Dataset'].nunique():,}"
    )

    print(
        f"  Cell-line regimens: "
        f"{cell_line['Regimen'].nunique():,}"
    )

    print(
        f"  Single-agent regimens: "
        f"{single['Regimen'].nunique():,}"
    )

    print(
        f"  Combination regimens held out: "
        f"{combination['Regimen'].nunique():,}"
    )

    print(
        "\nSingle-agent regimens:"
    )

    for regimen in sorted(
        single[
            "Regimen"
        ]
        .dropna()
        .astype(str)
        .unique()
    ):
        print(
            f"  - {regimen}"
        )

    # --------------------------------------------------------
    # Build signed single-agent resistance signatures
    #
    # Positive avg_log2FC:
    #     higher in DRMref resistant cells
    #
    # Negative avg_log2FC:
    #     lower in DRMref resistant cells
    #
    # If a drug/gene occurs more than once, use median log2FC.
    # --------------------------------------------------------

    aggregated = (
        single
        .groupby(
            [
                "regimen_norm",
                "Gene_symbol",
            ],
            as_index=False,
        )
        .agg(
            avg_log2FC=(
                "avg_log2FC",
                "median",
            ),
            evidence_rows=(
                "avg_log2FC",
                "size",
            ),
        )
    )

    aggregated_path = (
        OUT_DIR
        / "drmref_single_agent_signed_signatures.csv"
    )

    aggregated.to_csv(
        aggregated_path,
        index=False,
    )

    # --------------------------------------------------------
    # Load PHAROS drugs
    # --------------------------------------------------------

    if not COMBO_PATH.exists():
        raise FileNotFoundError(
            f"Missing PHAROS combo file: "
            f"{COMBO_PATH}"
        )

    combo = pd.read_parquet(
        COMBO_PATH
    )

    for col in [
        "drug_a",
        "drug_b",
    ]:
        if col not in combo.columns:
            raise ValueError(
                f"Expected '{col}' in "
                f"{COMBO_PATH}"
            )

    pharos_drugs = sorted(
        set(
            combo[
                "drug_a"
            ]
            .astype(str)
        )
        | set(
            combo[
                "drug_b"
            ]
            .astype(str)
        )
    )

    drmref_single_norms = set(
        aggregated[
            "regimen_norm"
        ]
        .astype(str)
    )

    drug_mapping_rows = []

    for drug in pharos_drugs:

        norm = normalize_drug_name(
            drug
        )

        matched = (
            norm
            in drmref_single_norms
        )

        drug_mapping_rows.append(
            {
                "pharos_drug":
                    drug,

                "normalized_name":
                    norm,

                "drmref_matched":
                    matched,
            }
        )

    drug_mapping = pd.DataFrame(
        drug_mapping_rows
    )

    drug_mapping_path = (
        OUT_DIR
        / "pharos_drmref_drug_mapping.csv"
    )

    drug_mapping.to_csv(
        drug_mapping_path,
        index=False,
    )

    matched_drugs = set(
        drug_mapping.loc[
            drug_mapping[
                "drmref_matched"
            ],
            "pharos_drug",
        ]
        .astype(str)
    )

    matched_norms = set(
        drug_mapping.loc[
            drug_mapping[
                "drmref_matched"
            ],
            "normalized_name",
        ]
        .astype(str)
    )

    # --------------------------------------------------------
    # Row-level PHAROS coverage
    # --------------------------------------------------------

    a_match = (
        combo[
            "drug_a"
        ]
        .astype(str)
        .isin(
            matched_drugs
        )
    )

    b_match = (
        combo[
            "drug_b"
        ]
        .astype(str)
        .isin(
            matched_drugs
        )
    )

    any_match = (
        a_match
        | b_match
    )

    both_match = (
        a_match
        & b_match
    )

    exactly_one = (
        a_match
        ^ b_match
    )

    neither = (
        ~any_match
    )

    # --------------------------------------------------------
    # Align DRMref genes to DepMap
    # --------------------------------------------------------

    if not EXPRESSION_PATH.exists():
        raise FileNotFoundError(
            f"Missing DepMap expression file: "
            f"{EXPRESSION_PATH}"
        )

    expression_head = pd.read_parquet(
        EXPRESSION_PATH
    ).head(
        1
    )

    expression_cols = [
        (
            "depmap_id"
            if col == "depmap_id"
            else clean_gene_name(
                str(col)
            )
        )
        for col
        in expression_head.columns
    ]

    depmap_lookup = {}

    ambiguous = set()

    for col in expression_cols:

        if col == "depmap_id":
            continue

        key = (
            str(col)
            .strip()
            .upper()
        )

        if (
            key
            in depmap_lookup
            and depmap_lookup[
                key
            ]
            != col
        ):
            ambiguous.add(
                key
            )

        else:
            depmap_lookup[
                key
            ] = col

    if ambiguous:
        raise ValueError(
            "Ambiguous case-insensitive DepMap genes: "
            f"{sorted(ambiguous)[:20]}"
        )

    # Only genes from DRMref drugs that actually match PHAROS.
    matched_signature = aggregated[
        aggregated[
            "regimen_norm"
        ]
        .isin(
            matched_norms
        )
    ].copy()

    gene_mapping_rows = []

    for gene in sorted(
        matched_signature[
            "Gene_symbol"
        ]
        .dropna()
        .astype(str)
        .unique()
    ):

        key = (
            gene
            .strip()
            .upper()
        )

        resolved = (
            depmap_lookup
            .get(
                key
            )
        )

        gene_mapping_rows.append(
            {
                "drmref_gene":
                    gene,

                "depmap_gene":
                    resolved,

                "usable":
                    (
                        resolved
                        is not None
                    ),
            }
        )

    gene_mapping = pd.DataFrame(
        gene_mapping_rows
    )

    gene_mapping_path = (
        OUT_DIR
        / "drmref_depmap_gene_mapping.csv"
    )

    gene_mapping.to_csv(
        gene_mapping_path,
        index=False,
    )

    usable_gene_mapping = (
        gene_mapping[
            gene_mapping[
                "usable"
            ]
        ]
        .copy()
    )

    usable_genes = (
        usable_gene_mapping[
            "drmref_gene"
        ]
        .astype(str)
        .tolist()
    )

    usable_gene_set = set(
        usable_genes
    )

    # --------------------------------------------------------
    # Build PHAROS drug x DRMref gene signed-weight matrix
    # --------------------------------------------------------

    gene_index = {
        gene:
            i
        for i, gene
        in enumerate(
            usable_genes
        )
    }

    weights = np.zeros(
        (
            len(
                pharos_drugs
            ),
            len(
                usable_genes
            ),
        ),
        dtype=np.float32,
    )

    mask = np.zeros(
        weights.shape,
        dtype=np.uint8,
    )

    grouped = {
        norm:
            frame
        for norm, frame
        in matched_signature.groupby(
            "regimen_norm"
        )
    }

    for drug_i, drug in enumerate(
        pharos_drugs
    ):

        norm = normalize_drug_name(
            drug
        )

        frame = grouped.get(
            norm
        )

        if frame is None:
            continue

        for row in frame.itertuples(
            index=False
        ):

            gene = str(
                row.Gene_symbol
            )

            if (
                gene
                not in usable_gene_set
            ):
                continue

            gene_i = (
                gene_index[
                    gene
                ]
            )

            weights[
                drug_i,
                gene_i
            ] = np.float32(
                row.avg_log2FC
            )

            mask[
                drug_i,
                gene_i
            ] = 1

    weight_cache_path = (
        OUT_DIR
        / "pharos_drmref_resistance_weights.npz"
    )

    np.savez_compressed(
        weight_cache_path,
        weights=weights,
        mask=mask,
        drug_names=np.asarray(
            pharos_drugs,
            dtype=str,
        ),
        genes=np.asarray(
            usable_genes,
            dtype=str,
        ),
    )

    # --------------------------------------------------------
    # Provenance
    # --------------------------------------------------------

    provenance_cols = [
        "Regimen",
        "regimen_norm",
        "Original_Dataset",
        "Cancer_type_level1",
        "Source",
        "PMID",
        "Description",
    ]

    provenance = (
        single[
            provenance_cols
        ]
        .drop_duplicates()
        .sort_values(
            [
                "Regimen",
                "Original_Dataset",
            ]
        )
    )

    provenance_path = (
        OUT_DIR
        / "drmref_resistance_signature_provenance.csv"
    )

    provenance.to_csv(
        provenance_path,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    nonzero_rows = (
        mask.sum(
            axis=1
        )
        > 0
    )

    active_counts = (
        mask[
            nonzero_rows
        ]
        .sum(
            axis=1
        )
    )

    summary = {
        "drmref_raw_deg_rows":
            len(
                drmref
            ),

        "drmref_cell_line_deg_rows":
            len(
                cell_line
            ),

        "drmref_cell_line_datasets":
            cell_line[
                "Original_Dataset"
            ]
            .nunique(),

        "drmref_single_agent_regimens":
            single[
                "Regimen"
            ]
            .nunique(),

        "drmref_combination_regimens_held_out":
            combination[
                "Regimen"
            ]
            .nunique(),

        "pharos_unique_drugs":
            len(
                pharos_drugs
            ),

        "pharos_matched_drug_names":
            len(
                matched_drugs
            ),

        "pharos_matched_unique_normalized_drugs":
            len(
                matched_norms
            ),

        "pharos_rows":
            len(
                combo
            ),

        "rows_drug_a_matched":
            int(
                a_match.sum()
            ),

        "rows_drug_b_matched":
            int(
                b_match.sum()
            ),

        "rows_any_matched":
            int(
                any_match.sum()
            ),

        "rows_both_matched":
            int(
                both_match.sum()
            ),

        "rows_exactly_one_matched":
            int(
                exactly_one.sum()
            ),

        "rows_neither_matched":
            int(
                neither.sum()
            ),

        "rows_any_matched_fraction":
            float(
                any_match.mean()
            ),

        "rows_both_matched_fraction":
            float(
                both_match.mean()
            ),

        "drmref_genes_for_matched_pharos_drugs":
            gene_mapping.shape[
                0
            ],

        "drmref_genes_usable_in_depmap":
            usable_gene_mapping.shape[
                0
            ],

        "drmref_depmap_gene_coverage":
            (
                float(
                    usable_gene_mapping.shape[
                        0
                    ]
                    / gene_mapping.shape[
                        0
                    ]
                )
                if gene_mapping.shape[
                    0
                ]
                else 0.0
            ),

        "weight_matrix_nonzero_drug_rows":
            int(
                nonzero_rows.sum()
            ),

        "weight_matrix_genes":
            len(
                usable_genes
            ),

        "weight_matrix_nonzero_entries":
            int(
                mask.sum()
            ),

        "genes_per_matched_drug_min":
            (
                int(
                    active_counts.min()
                )
                if len(
                    active_counts
                )
                else 0
            ),

        "genes_per_matched_drug_median":
            (
                float(
                    np.median(
                        active_counts
                    )
                )
                if len(
                    active_counts
                )
                else 0.0
            ),

        "genes_per_matched_drug_max":
            (
                int(
                    active_counts.max()
                )
                if len(
                    active_counts
                )
                else 0
            ),
    }

    summary_path = (
        OUT_DIR
        / "drmref_pharos_coverage_summary.csv"
    )

    pd.DataFrame(
        [
            summary
        ]
    ).to_csv(
        summary_path,
        index=False,
    )

    print(
        "\n========================================"
    )
    print(
        "DRMref -> PHAROS COVERAGE"
    )
    print(
        "========================================"
    )

    print(
        f"PHAROS drugs: "
        f"{len(pharos_drugs):,}"
    )

    print(
        f"Matched PHAROS drug names: "
        f"{len(matched_drugs):,}"
    )

    print(
        f"Matched unique normalized drugs: "
        f"{len(matched_norms):,}"
    )

    print(
        f"Rows with Drug A DRMref evidence: "
        f"{a_match.sum():,} "
        f"({100 * a_match.mean():.2f}%)"
    )

    print(
        f"Rows with Drug B DRMref evidence: "
        f"{b_match.sum():,} "
        f"({100 * b_match.mean():.2f}%)"
    )

    print(
        f"Rows with >=1 DRMref-matched drug: "
        f"{any_match.sum():,} "
        f"({100 * any_match.mean():.2f}%)"
    )

    print(
        f"Rows with both DRMref-matched drugs: "
        f"{both_match.sum():,} "
        f"({100 * both_match.mean():.2f}%)"
    )

    print(
        f"Rows with neither: "
        f"{neither.sum():,} "
        f"({100 * neither.mean():.2f}%)"
    )

    print(
        f"DRMref genes for matched drugs: "
        f"{gene_mapping.shape[0]:,}"
    )

    print(
        f"Usable in DepMap: "
        f"{usable_gene_mapping.shape[0]:,}"
    )

    if gene_mapping.shape[
        0
    ]:

        print(
            f"Gene coverage: "
            f"{100 * usable_gene_mapping.shape[0] / gene_mapping.shape[0]:.2f}%"
        )

    print(
        f"Signed weight cache: "
        f"{weights.shape}"
    )

    print(
        f"Non-zero drug rows: "
        f"{nonzero_rows.sum():,}"
    )

    if len(
        active_counts
    ):

        print(
            "Genes per matched drug "
            "(min/median/max): "
            f"{active_counts.min():,}/"
            f"{np.median(active_counts):.1f}/"
            f"{active_counts.max():,}"
        )

    print(
        "\nMatched PHAROS drugs:"
    )

    if matched_drugs:

        for drug in sorted(
            matched_drugs
        ):
            print(
                f"  - {drug}"
            )

    else:

        print(
            "  (none)"
        )

    print(
        "\nSaved:"
    )

    for path in [
        all_cell_line_path,
        combination_path,
        single_path,
        aggregated_path,
        drug_mapping_path,
        gene_mapping_path,
        weight_cache_path,
        provenance_path,
        summary_path,
    ]:

        print(
            f"  {path}"
        )

    print(
        "\nPASS: DRMref real resistant-cell signatures prepared"
    )


if __name__ == "__main__":
    main()
