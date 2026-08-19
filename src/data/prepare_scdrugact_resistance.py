from __future__ import annotations

import ast
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# RDKit is optional here. If available, we use canonical SMILES matching.
try:
    from rdkit import Chem
    HAS_RDKIT = True
except Exception:
    Chem = None
    HAS_RDKIT = False


# ============================================================
# Paths
# ============================================================

SCDRUGACT_DIR = Path(
    "data/raw/resistance/scdrugact/extracted/sens_resi_druginfo"
)

RESISTANCE_PATH = SCDRUGACT_DIR / "Resistance gene.csv"
METADATA_PATH = SCDRUGACT_DIR / "Sens_Resi druginfo.txt"

PHAROS_DRUG_TABLE = Path(
    "data/processed/combination/pharos_combo_drug_table.csv"
)

OUTPUT_DIR = Path(
    "data/processed/combination/scdrugact"
)

MAPPING_OUT = OUTPUT_DIR / "pharos_scdrugact_drug_mapping.csv"
LONG_OUT = OUTPUT_DIR / "pharos_scdrugact_resistance_long.csv"
VOCAB_OUT = OUTPUT_DIR / "scdrugact_resistance_gene_vocabulary.txt"
SUMMARY_OUT = OUTPUT_DIR / "scdrugact_resistance_coverage_summary.csv"


# ============================================================
# Existing PHAROS V1 core panel
# ============================================================

CORE_29 = {
    "ABCB1", "ABCC1", "ABCG2",
    "BCL2", "BAX", "BAK1",
    "CASP3", "CASP8", "CASP9",
    "AKT1", "TP53",
    "BRCA1", "BRCA2", "RAD51", "ERCC1", "MGMT",
    "ATM", "ATR",
    "GSTP1", "GSTA1",
    "CYP3A4", "CYP2D6",
    "HIF1A", "VEGFA", "IL6", "CXCR4",
    "DNMT1", "HDAC1", "HDAC2",
}


# ============================================================
# Helpers
# ============================================================

def normalize_name(value: object) -> str:
    """
    Conservative drug-name normalization for matching.

    Examples:
      "PD-0332991" -> "pd0332991"
      "5-Fluorouracil" -> "5fluorouracil"

    Ambiguous normalized aliases are removed later rather than
    forcing potentially incorrect matches.
    """
    if value is None:
        return ""

    text = str(value).strip()

    if not text or text.lower() == "nan":
        return ""

    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()

    return re.sub(r"[^a-z0-9]+", "", text)


def normalize_gene(value: object) -> str:
    if value is None:
        return ""

    text = str(value).strip()

    if not text or text.lower() == "nan":
        return ""

    return text.upper()


def canonical_smiles(value: object) -> str:
    if value is None:
        return ""

    text = str(value).strip()

    if not text or text.lower() == "nan":
        return ""

    if not HAS_RDKIT:
        # Fallback: exact raw-SMILES matching only.
        return text

    mol = Chem.MolFromSmiles(text)

    if mol is None:
        return ""

    return Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=True,
    )


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
    description: str,
) -> str:

    normalized = {
        normalize_name(col): col
        for col in df.columns
    }

    for candidate in candidates:
        key = normalize_name(candidate)

        if key in normalized:
            return normalized[key]

    raise ValueError(
        f"Could not find {description} column.\n"
        f"Available columns: {list(df.columns)}"
    )


def parse_synonyms(value: object) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [
            str(x).strip()
            for x in value
            if str(x).strip()
        ]

    text = str(value).strip()

    if not text or text.lower() == "nan":
        return []

    # scDrugAct stores this column as Python-list-like text.
    try:
        parsed = ast.literal_eval(text)

        if isinstance(parsed, (list, tuple, set)):
            return [
                str(x).strip()
                for x in parsed
                if str(x).strip()
            ]

    except Exception:
        pass

    # Safe fallback for any oddly formatted record.
    return [
        x.strip().strip("'\"")
        for x in re.split(r"[|;,]", text)
        if x.strip()
    ]


def make_unique_lookup(
    pairs: list[tuple[str, str]],
) -> tuple[dict[str, str], set[str]]:
    """
    Convert (key, target) pairs into a lookup, but only keep keys
    that point to exactly one target. Ambiguous aliases are excluded.
    """

    bucket: dict[str, set[str]] = defaultdict(set)

    for key, target in pairs:
        if key and target:
            bucket[key].add(target)

    lookup = {}
    ambiguous = set()

    for key, targets in bucket.items():
        if len(targets) == 1:
            lookup[key] = next(iter(targets))
        else:
            ambiguous.add(key)

    return lookup, ambiguous


# ============================================================
# Load scDrugAct resistance matrix
# ============================================================

def load_resistance_sets() -> tuple[
    dict[str, set[str]],
    dict[str, str],
]:

    print(
        "\n========================================"
    )
    print(
        "LOADING SCDRUGACT RESISTANCE MATRIX"
    )
    print(
        "========================================"
    )

    if not RESISTANCE_PATH.exists():
        raise FileNotFoundError(
            f"Missing:\n{RESISTANCE_PATH}"
        )

    df = pd.read_csv(
        RESISTANCE_PATH,
        low_memory=False,
    )

    print(
        f"Matrix shape: {df.shape}"
    )
    print(
        f"Drug columns: {len(df.columns):,}"
    )

    resistance_sets: dict[str, set[str]] = {}
    normalized_column_pairs = []

    for drug_col in df.columns:

        genes = {
            normalize_gene(value)
            for value in df[drug_col].dropna().tolist()
        }

        genes.discard("")

        resistance_sets[str(drug_col)] = genes

        norm = normalize_name(drug_col)

        if norm:
            normalized_column_pairs.append(
                (norm, str(drug_col))
            )

    normalized_drug_lookup, ambiguous = make_unique_lookup(
        normalized_column_pairs
    )

    counts = np.array(
        [
            len(v)
            for v in resistance_sets.values()
        ],
        dtype=np.int64,
    )

    print(
        f"Unique normalized drug names: "
        f"{len(normalized_drug_lookup):,}"
    )
    print(
        f"Ambiguous normalized names: "
        f"{len(ambiguous):,}"
    )
    print(
        "Resistance genes per drug "
        f"(median/min/max): "
        f"{np.median(counts):.0f} / "
        f"{counts.min()} / "
        f"{counts.max()}"
    )

    union_genes = set().union(
        *resistance_sets.values()
    )

    print(
        f"Unique resistance genes overall: "
        f"{len(union_genes):,}"
    )

    core_overlap = (
        union_genes
        & CORE_29
    )

    print(
        f"Core-29 genes represented in scDrugAct: "
        f"{len(core_overlap)}/29"
    )

    if core_overlap:
        print(
            "Core overlap:",
            ", ".join(
                sorted(core_overlap)
            )
        )

    return (
        resistance_sets,
        normalized_drug_lookup,
    )


# ============================================================
# Load scDrugAct metadata and aliases
# ============================================================

def build_metadata_lookups(
    normalized_matrix_lookup: dict[str, str],
) -> tuple[
    dict[str, str],
    dict[str, str],
]:

    print(
        "\n========================================"
    )
    print(
        "BUILDING SCDRUGACT ALIAS LOOKUPS"
    )
    print(
        "========================================"
    )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing:\n{METADATA_PATH}"
        )

    meta = pd.read_csv(
        METADATA_PATH,
        sep="\t",
        low_memory=False,
    )

    # If the file is not actually tab-delimited on a specific
    # distribution, fall back to delimiter inference.
    if len(meta.columns) <= 1:
        meta = pd.read_csv(
            METADATA_PATH,
            sep=None,
            engine="python",
            low_memory=False,
        )

    print(
        f"Metadata shape: {meta.shape}"
    )

    drug_col = find_column(
        meta,
        ["Drug"],
        "scDrugAct drug",
    )

    synonyms_col = find_column(
        meta,
        ["synonyms"],
        "scDrugAct synonyms",
    )

    smiles_col = find_column(
        meta,
        ["smiles"],
        "scDrugAct SMILES",
    )

    alias_pairs = []
    smiles_pairs = []

    metadata_rows_resolved_to_matrix = 0

    for _, row in meta.iterrows():

        metadata_drug = str(
            row[drug_col]
        ).strip()

        metadata_norm = normalize_name(
            metadata_drug
        )

        # The metadata table and matrix have different counts.
        # We only use a metadata entry when we can resolve its
        # canonical Drug field back to an actual matrix column.
        matrix_drug = normalized_matrix_lookup.get(
            metadata_norm
        )

        if matrix_drug is None:
            continue

        metadata_rows_resolved_to_matrix += 1

        # Canonical name itself is also an alias.
        alias_candidates = [
            metadata_drug
        ]

        alias_candidates.extend(
            parse_synonyms(
                row[synonyms_col]
            )
        )

        for alias in alias_candidates:
            norm_alias = normalize_name(
                alias
            )

            if norm_alias:
                alias_pairs.append(
                    (
                        norm_alias,
                        matrix_drug,
                    )
                )

        can_smiles = canonical_smiles(
            row[smiles_col]
        )

        if can_smiles:
            smiles_pairs.append(
                (
                    can_smiles,
                    matrix_drug,
                )
            )

    alias_lookup, alias_ambiguous = (
        make_unique_lookup(
            alias_pairs
        )
    )

    smiles_lookup, smiles_ambiguous = (
        make_unique_lookup(
            smiles_pairs
        )
    )

    print(
        "Metadata rows resolved to a resistance "
        f"matrix drug: "
        f"{metadata_rows_resolved_to_matrix:,}"
    )
    print(
        f"Unique normalized aliases: "
        f"{len(alias_lookup):,}"
    )
    print(
        f"Ambiguous aliases ignored: "
        f"{len(alias_ambiguous):,}"
    )
    print(
        f"Unique canonical SMILES mappings: "
        f"{len(smiles_lookup):,}"
    )
    print(
        f"Ambiguous SMILES ignored: "
        f"{len(smiles_ambiguous):,}"
    )
    print(
        f"RDKit canonicalization: "
        f"{'ON' if HAS_RDKIT else 'OFF'}"
    )

    return (
        alias_lookup,
        smiles_lookup,
    )


# ============================================================
# PHAROS drug table
# ============================================================

def load_pharos_drugs() -> tuple[
    pd.DataFrame,
    str,
    str,
]:

    print(
        "\n========================================"
    )
    print(
        "LOADING PHAROS DRUGS"
    )
    print(
        "========================================"
    )

    if not PHAROS_DRUG_TABLE.exists():
        raise FileNotFoundError(
            f"Missing:\n{PHAROS_DRUG_TABLE}"
        )

    df = pd.read_csv(
        PHAROS_DRUG_TABLE
    )

    drug_col = find_column(
        df,
        [
            "drug",
            "drug_name",
            "name",
            "compound",
            "compound_name",
        ],
        "PHAROS drug name",
    )

    smiles_col = find_column(
        df,
        [
            "smiles",
            "canonical_smiles",
            "canonicalsmiles",
        ],
        "PHAROS SMILES",
    )

    df = (
        df[
            [
                drug_col,
                smiles_col,
            ]
        ]
        .dropna(
            subset=[drug_col]
        )
        .copy()
    )

    df[drug_col] = (
        df[drug_col]
        .astype(str)
        .str.strip()
    )

    df = (
        df[
            df[drug_col] != ""
        ]
        .drop_duplicates(
            subset=[drug_col]
        )
        .reset_index(drop=True)
    )

    print(
        f"PHAROS unique drugs: "
        f"{len(df):,}"
    )

    return (
        df,
        drug_col,
        smiles_col,
    )


# ============================================================
# Match PHAROS -> scDrugAct
# ============================================================

def match_pharos_drugs(
    pharos: pd.DataFrame,
    pharos_drug_col: str,
    pharos_smiles_col: str,
    normalized_matrix_lookup: dict[str, str],
    alias_lookup: dict[str, str],
    smiles_lookup: dict[str, str],
    resistance_sets: dict[str, set[str]],
) -> pd.DataFrame:

    print(
        "\n========================================"
    )
    print(
        "MATCHING PHAROS TO SCDRUGACT"
    )
    print(
        "========================================"
    )

    rows = []

    for _, row in pharos.iterrows():

        pharos_drug = str(
            row[pharos_drug_col]
        ).strip()

        pharos_smiles = str(
            row[pharos_smiles_col]
        ).strip()

        normalized = normalize_name(
            pharos_drug
        )

        matched_drug = None
        match_method = None

        # 1. Direct normalized match to resistance matrix column.
        if normalized in normalized_matrix_lookup:
            matched_drug = (
                normalized_matrix_lookup[
                    normalized
                ]
            )
            match_method = (
                "normalized_matrix_name"
            )

        # 2. scDrugAct metadata synonym/alias.
        elif normalized in alias_lookup:
            matched_drug = (
                alias_lookup[
                    normalized
                ]
            )
            match_method = (
                "metadata_alias"
            )

        # 3. Molecular structure through canonical SMILES.
        else:
            can_smiles = canonical_smiles(
                pharos_smiles
            )

            if (
                can_smiles
                and can_smiles
                in smiles_lookup
            ):
                matched_drug = (
                    smiles_lookup[
                        can_smiles
                    ]
                )
                match_method = (
                    "canonical_smiles"
                )

        n_genes = 0

        if matched_drug is not None:
            n_genes = len(
                resistance_sets[
                    matched_drug
                ]
            )

        rows.append(
            {
                "pharos_drug":
                    pharos_drug,

                "pharos_smiles":
                    pharos_smiles,

                "matched_scdrugact_drug":
                    matched_drug,

                "match_method":
                    match_method,

                "matched":
                    matched_drug
                    is not None,

                "n_scdrugact_resistance_genes":
                    n_genes,
            }
        )

    mapping = pd.DataFrame(
        rows
    )

    matched = mapping[
        mapping["matched"]
    ]

    print(
        f"Matched drugs: "
        f"{len(matched):,}/"
        f"{len(mapping):,} "
        f"({100 * len(matched) / len(mapping):.2f}%)"
    )

    if len(matched):

        print(
            "\nMatch methods:"
        )

        print(
            matched[
                "match_method"
            ]
            .value_counts()
            .to_string()
        )

        counts = (
            matched[
                "n_scdrugact_resistance_genes"
            ]
            .to_numpy(
                dtype=float
            )
        )

        print(
            "\nMatched-drug resistance gene count "
            "(median/min/max): "
            f"{np.median(counts):.0f} / "
            f"{counts.min():.0f} / "
            f"{counts.max():.0f}"
        )

    return mapping


# ============================================================
# Build PHAROS-specific resistance long table
# ============================================================

def build_pharos_resistance_long(
    mapping: pd.DataFrame,
    resistance_sets: dict[str, set[str]],
) -> pd.DataFrame:

    rows = []

    matched = mapping[
        mapping["matched"]
    ]

    for _, row in matched.iterrows():

        pharos_drug = (
            row[
                "pharos_drug"
            ]
        )

        scdrugact_drug = (
            row[
                "matched_scdrugact_drug"
            ]
        )

        genes = (
            resistance_sets[
                scdrugact_drug
            ]
        )

        for gene in sorted(
            genes
        ):

            rows.append(
                {
                    "pharos_drug":
                        pharos_drug,

                    "scdrugact_drug":
                        scdrugact_drug,

                    "gene":
                        gene,

                    "source":
                        "scDrugAct",

                    "match_method":
                        row[
                            "match_method"
                        ],
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "========================================"
    )
    print(
        "PHAROS × SCDRUGACT RESISTANCE PREP"
    )
    print(
        "========================================"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        resistance_sets,
        normalized_matrix_lookup,
    ) = load_resistance_sets()

    (
        alias_lookup,
        smiles_lookup,
    ) = build_metadata_lookups(
        normalized_matrix_lookup
    )

    (
        pharos,
        pharos_drug_col,
        pharos_smiles_col,
    ) = load_pharos_drugs()

    mapping = match_pharos_drugs(
        pharos=pharos,

        pharos_drug_col=(
            pharos_drug_col
        ),

        pharos_smiles_col=(
            pharos_smiles_col
        ),

        normalized_matrix_lookup=(
            normalized_matrix_lookup
        ),

        alias_lookup=(
            alias_lookup
        ),

        smiles_lookup=(
            smiles_lookup
        ),

        resistance_sets=(
            resistance_sets
        ),
    )

    resistance_long = (
        build_pharos_resistance_long(
            mapping,
            resistance_sets,
        )
    )

    # ========================================================
    # Global matched-gene vocabulary
    # ========================================================

    if len(resistance_long):
        vocabulary = sorted(
            resistance_long[
                "gene"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
    else:
        vocabulary = []

    matched_drugs = int(
        mapping[
            "matched"
        ].sum()
    )

    total_drugs = len(
        mapping
    )

    coverage = (
        matched_drugs
        / total_drugs
        if total_drugs
        else 0.0
    )

    matched_core = sorted(
        set(vocabulary)
        & CORE_29
    )

    summary = pd.DataFrame(
        [
            {
                "pharos_drugs":
                    total_drugs,

                "matched_pharos_drugs":
                    matched_drugs,

                "drug_coverage_fraction":
                    coverage,

                "drug_coverage_percent":
                    100.0 * coverage,

                "matched_resistance_gene_vocabulary":
                    len(vocabulary),

                "core29_genes_present":
                    len(matched_core),

                "rdkit_smiles_matching":
                    HAS_RDKIT,
            }
        ]
    )

    # ========================================================
    # Save
    # ========================================================

    mapping.to_csv(
        MAPPING_OUT,
        index=False,
    )

    resistance_long.to_csv(
        LONG_OUT,
        index=False,
    )

    VOCAB_OUT.write_text(
        "\n".join(
            vocabulary
        )
        + (
            "\n"
            if vocabulary
            else ""
        )
    )

    summary.to_csv(
        SUMMARY_OUT,
        index=False,
    )

    # ========================================================
    # Final summary
    # ========================================================

    print(
        "\n========================================"
    )
    print(
        "FINAL COVERAGE SUMMARY"
    )
    print(
        "========================================"
    )

    print(
        f"PHAROS drugs: "
        f"{total_drugs:,}"
    )

    print(
        f"Matched to scDrugAct resistance data: "
        f"{matched_drugs:,}"
    )

    print(
        f"Drug coverage: "
        f"{100.0 * coverage:.2f}%"
    )

    print(
        f"Resistance-gene vocabulary among "
        f"matched PHAROS drugs: "
        f"{len(vocabulary):,}"
    )

    print(
        f"Core-29 overlap within matched "
        f"vocabulary: "
        f"{len(matched_core)}/29"
    )

    if matched_core:
        print(
            "Core-29 overlap:",
            ", ".join(
                matched_core
            )
        )

    print(
        "\nSaved:"
    )

    print(
        f"  {MAPPING_OUT}"
    )

    print(
        f"  {LONG_OUT}"
    )

    print(
        f"  {VOCAB_OUT}"
    )

    print(
        f"  {SUMMARY_OUT}"
    )

    print(
        "\nNext decision:"
    )

    print(
        "If drug coverage is strong enough, "
        "build Resistance V2 masks from this "
        "mapping and intersect the gene "
        "vocabulary with DepMap expression."
    )


if __name__ == "__main__":
    main()
