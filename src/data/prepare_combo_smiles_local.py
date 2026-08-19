from pathlib import Path
import re

import pandas as pd

from rdkit import Chem
from rdkit import RDLogger


# ============================================================
# Quiet RDKit parser warnings
# ============================================================

RDLogger.DisableLog("rdApp.*")


# ============================================================
# Paths
# ============================================================

PHAROS_RESPONSE_PATH = Path(
    "data/processed/pharos_depmap_response_pairs.parquet"
)

COMBO_DRUG_PATH = Path(
    "data/processed/combination/"
    "pharos_combo_unique_drugs.csv"
)

COMBO_DATA_PATH = Path(
    "data/processed/combination/"
    "pharos_combo_clean.parquet"
)

OUTPUT_DIR = Path(
    "data/processed/combination"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DRUG_MAP_PATH = (
    OUTPUT_DIR
    / "combo_drug_smiles_local_clean.csv"
)

UNRESOLVED_PATH = (
    OUTPUT_DIR
    / "combo_drugs_still_unresolved.csv"
)

AMBIGUOUS_PATH = (
    OUTPUT_DIR
    / "combo_drugs_ambiguous_smiles.csv"
)

USABLE_COMBO_PATH = (
    OUTPUT_DIR
    / "pharos_combo_local_smiles.parquet"
)


# ============================================================
# Drug-name normalization
# ============================================================

def normalize_drug_name(value):

    if pd.isna(value):
        return None

    return re.sub(
        r"[^A-Z0-9]",
        "",
        str(value).upper(),
    )


# ============================================================
# Some PHAROS fields contain:
#
# SMILES, SMILES, SMILES
#
# instead of one clean SMILES.
# ============================================================

def split_smiles_field(value):

    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    parts = re.split(
        r"\s*,\s*",
        value,
    )

    return [
        part.strip()
        for part in parts
        if part.strip()
    ]


# ============================================================
# Canonicalize with RDKit
# ============================================================

def canonicalize_smiles(smiles):

    try:

        mol = Chem.MolFromSmiles(
            smiles
        )

        if mol is None:
            return None

        return Chem.MolToSmiles(
            mol,
            canonical=True,
            isomericSmiles=True,
        )

    except Exception:

        return None


# ============================================================
# Main
# ============================================================

def main():

    print(
        "========================================"
    )
    print(
        "PHAROS-COMBO LOCAL SMILES CLEANUP"
    )
    print(
        "========================================"
    )

    # --------------------------------------------------------
    # Load source PHAROS chemistry
    # --------------------------------------------------------

    pharos = pd.read_parquet(
        PHAROS_RESPONSE_PATH,
        columns=[
            "name",
            "smiles",
        ],
    )

    pharos = (
        pharos
        .dropna(
            subset=[
                "name",
                "smiles",
            ]
        )
        .drop_duplicates()
        .copy()
    )

    print(
        f"\nUnique PHAROS "
        f"name/SMILES records: "
        f"{len(pharos):,}"
    )

    # --------------------------------------------------------
    # Build normalized name -> canonical SMILES candidates
    # --------------------------------------------------------

    smiles_candidates = {}

    invalid_smiles_count = 0

    total_smiles_candidates = 0

    for _, row in pharos.iterrows():

        key = normalize_drug_name(
            row["name"]
        )

        if not key:
            continue

        raw_candidates = (
            split_smiles_field(
                row["smiles"]
            )
        )

        for raw_smiles in raw_candidates:

            total_smiles_candidates += 1

            canonical = (
                canonicalize_smiles(
                    raw_smiles
                )
            )

            if canonical is None:

                invalid_smiles_count += 1
                continue

            if key not in smiles_candidates:

                smiles_candidates[
                    key
                ] = set()

            smiles_candidates[
                key
            ].add(
                canonical
            )

    # --------------------------------------------------------
    # Unambiguous vs ambiguous mappings
    # --------------------------------------------------------

    resolved_map = {}

    ambiguous_map = {}

    for key, candidates in (
        smiles_candidates.items()
    ):

        if len(candidates) == 1:

            resolved_map[
                key
            ] = next(
                iter(candidates)
            )

        elif len(candidates) > 1:

            ambiguous_map[
                key
            ] = sorted(
                candidates
            )

    print(
        f"\nTotal parsed SMILES candidates: "
        f"{total_smiles_candidates:,}"
    )

    print(
        f"Invalid SMILES candidates: "
        f"{invalid_smiles_count:,}"
    )

    print(
        f"Unambiguous canonical "
        f"name mappings: "
        f"{len(resolved_map):,}"
    )

    print(
        f"Ambiguous canonical "
        f"name mappings: "
        f"{len(ambiguous_map):,}"
    )

    # --------------------------------------------------------
    # Load DrugComb unique drugs
    # --------------------------------------------------------

    combo_drugs = pd.read_csv(
        COMBO_DRUG_PATH
    )

    combo_drugs[
        "drug_key"
    ] = combo_drugs[
        "drug_name"
    ].map(
        normalize_drug_name
    )

    combo_drugs[
        "smiles"
    ] = combo_drugs[
        "drug_key"
    ].map(
        resolved_map
    )

    combo_drugs[
        "status"
    ] = "unresolved"

    combo_drugs.loc[
        combo_drugs[
            "smiles"
        ].notna(),
        "status",
    ] = "resolved_local"

    combo_drugs.loc[
        combo_drugs[
            "drug_key"
        ].isin(
            ambiguous_map.keys()
        ),
        "status",
    ] = "ambiguous"

    # --------------------------------------------------------
    # Save drug mapping
    # --------------------------------------------------------

    resolved_drugs = combo_drugs[
        combo_drugs[
            "status"
        ]
        == "resolved_local"
    ].copy()

    unresolved_drugs = combo_drugs[
        combo_drugs[
            "status"
        ]
        == "unresolved"
    ].copy()

    ambiguous_drugs = combo_drugs[
        combo_drugs[
            "status"
        ]
        == "ambiguous"
    ].copy()

    resolved_drugs.to_csv(
        DRUG_MAP_PATH,
        index=False,
    )

    unresolved_drugs.to_csv(
        UNRESOLVED_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Save ambiguous details
    # --------------------------------------------------------

    ambiguous_rows = []

    for _, row in (
        ambiguous_drugs.iterrows()
    ):

        key = row[
            "drug_key"
        ]

        candidates = (
            ambiguous_map.get(
                key,
                [],
            )
        )

        ambiguous_rows.append(
            {
                "drug_name":
                    row["drug_name"],

                "drug_key":
                    key,

                "candidate_smiles":
                    " || ".join(
                        candidates
                    ),
            }
        )

    pd.DataFrame(
        ambiguous_rows
    ).to_csv(
        AMBIGUOUS_PATH,
        index=False,
    )

    total_drugs = len(
        combo_drugs
    )

    print(
        "\n========================================"
    )
    print(
        "DRUG-LEVEL COVERAGE"
    )
    print(
        "========================================"
    )

    print(
        f"Combination drugs: "
        f"{total_drugs:,}"
    )

    print(
        f"Resolved locally: "
        f"{len(resolved_drugs):,}"
    )

    print(
        f"Unresolved: "
        f"{len(unresolved_drugs):,}"
    )

    print(
        f"Ambiguous: "
        f"{len(ambiguous_drugs):,}"
    )

    print(
        f"Usable local coverage: "
        f"{100 * len(resolved_drugs) / total_drugs:.2f}%"
    )

    # ========================================================
    # Now determine PAIR-LEVEL coverage
    # ========================================================

    combo = pd.read_parquet(
        COMBO_DATA_PATH
    )

    combo[
        "drug_a_key"
    ] = combo[
        "drug_a"
    ].map(
        normalize_drug_name
    )

    combo[
        "drug_b_key"
    ] = combo[
        "drug_b"
    ].map(
        normalize_drug_name
    )

    combo[
        "smiles_a"
    ] = combo[
        "drug_a_key"
    ].map(
        resolved_map
    )

    combo[
        "smiles_b"
    ] = combo[
        "drug_b_key"
    ].map(
        resolved_map
    )

    combo[
        "drug_a_resolved"
    ] = combo[
        "smiles_a"
    ].notna()

    combo[
        "drug_b_resolved"
    ] = combo[
        "smiles_b"
    ].notna()

    both_resolved = (
        combo[
            "drug_a_resolved"
        ]
        &
        combo[
            "drug_b_resolved"
        ]
    )

    usable = combo[
        both_resolved
    ].copy()

    # --------------------------------------------------------
    # Save immediately usable two-drug data
    # --------------------------------------------------------

    usable.to_parquet(
        USABLE_COMBO_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print(
        "\n========================================"
    )
    print(
        "PAIR-LEVEL COVERAGE"
    )
    print(
        "========================================"
    )

    print(
        f"Total clean combination rows: "
        f"{len(combo):,}"
    )

    print(
        f"Rows with Drug A resolved: "
        f"{combo['drug_a_resolved'].sum():,}"
    )

    print(
        f"Rows with Drug B resolved: "
        f"{combo['drug_b_resolved'].sum():,}"
    )

    print(
        f"Rows with BOTH drugs resolved: "
        f"{len(usable):,}"
    )

    print(
        f"Row coverage: "
        f"{100 * len(usable) / len(combo):.2f}%"
    )

    unique_pairs = (
        usable[
            [
                "drug_a",
                "drug_b",
            ]
        ]
        .drop_duplicates()
    )

    print(
        f"\nUsable unique drug pairs: "
        f"{len(unique_pairs):,}"
    )

    print(
        f"Usable unique drugs: "
        f"{len(
            set(usable['drug_a'])
            | set(usable['drug_b'])
        ):,}"
    )

    print(
        f"Usable DepMap cells: "
        f"{usable['depmap_id'].nunique():,}"
    )

    print(
        "\nZIP statistics on usable rows:"
    )

    print(
        usable[
            "zip_score"
        ].describe()
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
        f"Clean local drug map:\n"
        f"{DRUG_MAP_PATH}"
    )

    print(
        f"\nUnresolved drugs:\n"
        f"{UNRESOLVED_PATH}"
    )

    print(
        f"\nAmbiguous drugs:\n"
        f"{AMBIGUOUS_PATH}"
    )

    print(
        f"\nImmediately usable "
        f"combination dataset:\n"
        f"{USABLE_COMBO_PATH}"
    )


if __name__ == "__main__":
    main()