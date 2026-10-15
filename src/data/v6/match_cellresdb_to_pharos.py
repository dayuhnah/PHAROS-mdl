from pathlib import Path
import re
import pandas as pd


CELLRES_PATH = Path(
    "data/external/cellresdb_full/catalog/"
    "cellresdb_full_drugs_clean.csv"
)

PHAROS_PATH = Path(
    "data/processed/v4/drug_registry.parquet"
)

OUT_DIR = Path(
    "outputs/v6/cellresdb_expansion"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# NORMALISATION
# ============================================================

def norm(value):

    if pd.isna(value):
        return ""

    value = str(
        value
    ).lower().strip()

    # Remove punctuation / spaces:
    #
    # "5-Fluorouracil"
    # "5 fluorouracil"
    # -> same normalized identity
    value = re.sub(
        r"[^a-z0-9]+",
        "",
        value,
    )

    return value


# Terms that require manual biological review even if a
# name happens to appear in the PHAROS registry.
#
# These are treatment classes rather than precise compounds.
NON_SPECIFIC = {
    "platinum",
}


# ============================================================
# LOAD
# ============================================================

cell = pd.read_csv(
    CELLRES_PATH
)

pharos = pd.read_parquet(
    PHAROS_PATH
)


print("=" * 100)
print("CELLRESDB × PHAROS DRUG OVERLAP AUDIT")
print("=" * 100)

print(
    "PHAROS registry rows:",
    len(pharos)
)

print(
    "CellResDB treatment rows:",
    len(cell)
)


# ============================================================
# CELLRESDB UNIQUE DRUG TABLE
# ============================================================

cell_unique = (
    cell[
        cell[
            "drug_name_canonical"
        ].notna()
    ]
    .groupby(
        [
            "drug_name_canonical",
            "drugbank_id",
        ],
        dropna=False,
    )
    .agg(
        dataset_count=(
            "dataset_id",
            "nunique",
        ),
        datasets=(
            "dataset_id",
            lambda x:
                ";".join(
                    sorted(
                        set(
                            x.astype(str)
                        )
                    )
                ),
        ),
        cancer_types=(
            "cancer_type",
            lambda x:
                ";".join(
                    sorted(
                        set(
                            x.dropna()
                            .astype(str)
                        )
                    )
                ),
        ),
        therapy_types=(
            "drug_type",
            lambda x:
                ";".join(
                    sorted(
                        set(
                            x.dropna()
                            .astype(str)
                        )
                    )
                ),
        ),
    )
    .reset_index()
)


cell_unique[
    "cellres_norm"
] = cell_unique[
    "drug_name_canonical"
].map(
    norm
)


# ============================================================
# PHAROS NORMALISATION
# ============================================================

pharos = pharos.copy()

pharos[
    "name_norm"
] = pharos[
    "drug_name"
].map(
    norm
)

pharos[
    "key_norm"
] = pharos[
    "drug_key"
].map(
    norm
)


# ============================================================
# MATCH
# ============================================================

results = []


for _, drug in cell_unique.iterrows():

    cell_name = drug[
        "drug_name_canonical"
    ]

    cell_norm = drug[
        "cellres_norm"
    ]


    exact = pharos[
        (
            pharos[
                "name_norm"
            ]
            == cell_norm
        )
        |
        (
            pharos[
                "key_norm"
            ]
            == cell_norm
        )
    ].copy()


    # --------------------------------------------------------
    # NO DIRECT MATCH
    # --------------------------------------------------------

    if exact.empty:

        results.append({
            "cellres_drug":
                cell_name,

            "drugbank_id":
                drug[
                    "drugbank_id"
                ],

            "dataset_count":
                drug[
                    "dataset_count"
                ],

            "datasets":
                drug[
                    "datasets"
                ],

            "cancer_types":
                drug[
                    "cancer_types"
                ],

            "therapy_types":
                drug[
                    "therapy_types"
                ],

            "match_status":
                "unmatched",

            "exact_pharos_keys":
                "",

            "matched_pharos_keys":
                "",

            "matched_pharos_names":
                "",

            "matched_smiles_count":
                0,

            "pharos_key_count":
                0,
        })

        continue


    # --------------------------------------------------------
    # EXACT MATCH EXISTS
    # --------------------------------------------------------

    exact_keys = sorted(
        exact[
            "drug_key"
        ]
        .astype(str)
        .unique()
    )


    exact_smiles = sorted(
        exact[
            "canonical_smiles"
        ]
        .dropna()
        .astype(str)
        .unique()
    )


    # --------------------------------------------------------
    # EXPAND THROUGH IDENTICAL CANONICAL SMILES
    #
    # Example:
    #
    # Trametinib
    # TRAMETINIB (GSK1120212)
    #
    # If both have exactly the same canonical molecular
    # structure, resistance evidence should be able to map
    # to both PHAROS aliases.
    # --------------------------------------------------------

    equivalent = pharos[
        pharos[
            "canonical_smiles"
        ].isin(
            exact_smiles
        )
    ].copy()


    matched_keys = sorted(
        equivalent[
            "drug_key"
        ]
        .astype(str)
        .unique()
    )

    matched_names = sorted(
        equivalent[
            "drug_name"
        ]
        .astype(str)
        .unique()
    )


    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    if (
        norm(cell_name)
        in NON_SPECIFIC
    ):
        status = (
            "manual_review_non_specific"
        )

    elif (
        len(exact_smiles)
        > 1
    ):
        status = (
            "manual_review_multiple_structures"
        )

    elif (
        len(matched_keys)
        > len(exact_keys)
    ):
        status = (
            "exact_plus_same_smiles_aliases"
        )

    else:
        status = (
            "exact"
        )


    results.append({
        "cellres_drug":
            cell_name,

        "drugbank_id":
            drug[
                "drugbank_id"
            ],

        "dataset_count":
            drug[
                "dataset_count"
            ],

        "datasets":
            drug[
                "datasets"
            ],

        "cancer_types":
            drug[
                "cancer_types"
            ],

        "therapy_types":
            drug[
                "therapy_types"
            ],

        "match_status":
            status,

        "exact_pharos_keys":
            ";".join(
                exact_keys
            ),

        "matched_pharos_keys":
            ";".join(
                matched_keys
            ),

        "matched_pharos_names":
            ";".join(
                matched_names
            ),

        "matched_smiles_count":
            len(
                exact_smiles
            ),

        "pharos_key_count":
            len(
                matched_keys
            ),
    })


result = pd.DataFrame(
    results
)


# ============================================================
# SAVE
# ============================================================

out_path = (
    OUT_DIR
    / "cellresdb_pharos_drug_overlap.csv"
)

result.to_csv(
    out_path,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 100)
print("MATCH SUMMARY")
print("=" * 100)

print(
    "Unique CellResDB drugs:",
    len(result)
)


matched_mask = (
    result[
        "match_status"
    ]
    != "unmatched"
)


approved_candidate_mask = result[
    "match_status"
].isin(
    [
        "exact",
        "exact_plus_same_smiles_aliases",
    ]
)


print(
    "Any PHAROS name match:",
    int(
        matched_mask.sum()
    )
)

print(
    "Clean candidate matches:",
    int(
        approved_candidate_mask.sum()
    )
)

print(
    "Unmatched:",
    int(
        (
            result[
                "match_status"
            ]
            == "unmatched"
        ).sum()
    )
)

print(
    "Manual review:",
    int(
        result[
            "match_status"
        ]
        .str.startswith(
            "manual_review"
        )
        .sum()
    )
)


print()
print("=" * 100)
print("MATCH STATUS COUNTS")
print("=" * 100)

print(
    result[
        "match_status"
    ]
    .value_counts()
    .to_string()
)


print()
print("=" * 100)
print("MATCHED DRUGS")
print("=" * 100)

matched = result[
    result[
        "match_status"
    ]
    != "unmatched"
][
    [
        "cellres_drug",
        "drugbank_id",
        "dataset_count",
        "match_status",
        "matched_pharos_keys",
        "matched_pharos_names",
    ]
]

print(
    matched.to_string(
        index=False
    )
)


print()
print("=" * 100)
print("UNMATCHED DRUGS")
print("=" * 100)

unmatched = result[
    result[
        "match_status"
    ]
    == "unmatched"
][
    [
        "cellres_drug",
        "drugbank_id",
        "dataset_count",
        "cancer_types",
        "therapy_types",
    ]
]

print(
    unmatched.to_string(
        index=False
    )
)


print()
print("=" * 100)
print("SAME-SMILES ALIAS EXPANSIONS")
print("=" * 100)

aliases = result[
    result[
        "match_status"
    ]
    == "exact_plus_same_smiles_aliases"
][
    [
        "cellres_drug",
        "exact_pharos_keys",
        "matched_pharos_keys",
        "matched_pharos_names",
    ]
]

if len(
    aliases
):
    print(
        aliases.to_string(
            index=False
        )
    )
else:
    print(
        "None"
    )


print()
print("=" * 100)
print("SAVED")
print("=" * 100)

print(
    out_path
)
