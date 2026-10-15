from pathlib import Path
import pandas as pd


DRUGS_PATH = Path(
    "data/external/cellresdb_full/catalog/"
    "cellresdb_full_drugs_clean.csv"
)

UNPARSED_PATH = Path(
    "data/external/cellresdb_full/catalog/"
    "cellresdb_unparsed_regimen_components.csv"
)

MAP_PATH = Path(
    "outputs/v6/cellresdb_expansion/"
    "cellresdb_pharos_drug_overlap_audited.csv"
)

RESPONSE_PATH = Path(
    "data/processed/v4/"
    "pharos_mocktail_v4_multiomics.parquet"
)

OUT_DIR = Path(
    "outputs/v6/cellresdb_expansion"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# HELPERS
# ============================================================

def norm_key(value):

    if pd.isna(value):
        return ""

    return str(
        value
    ).strip().upper()


def parse_bool(value):

    return str(
        value
    ).strip().lower() in {
        "true",
        "1",
        "yes",
    }


# ============================================================
# LOAD
# ============================================================

drugs = pd.read_csv(
    DRUGS_PATH
)

mapping = pd.read_csv(
    MAP_PATH
)

response = pd.read_parquet(
    RESPONSE_PATH,
    columns=[
        "drug_a_key",
        "drug_b_key",
    ],
)


if UNPARSED_PATH.exists():

    unparsed = pd.read_csv(
        UNPARSED_PATH
    )

else:

    unparsed = pd.DataFrame(
        columns=[
            "dataset_id",
            "unparsed_component",
        ]
    )


mapping[
    "approved"
] = mapping[
    "approved"
].map(
    parse_bool
)


approved_map = mapping[
    mapping[
        "approved"
    ]
].copy()


approved_drugs = set(
    approved_map[
        "cellres_drug"
    ]
    .astype(str)
)


# ============================================================
# NUMBER OF CONCRETE DRUGS PER DATASET
# ============================================================
#
# Important:
#
# Count ALL DrugBank-tagged concrete drugs,
# not only PHAROS-matched drugs.
#
# Example:
#
# Pembrolizumab + Vismodegib
#
# Vismodegib is PHAROS-compatible but this is still
# combination-context evidence.
# ============================================================

valid_drugs = drugs[
    drugs[
        "drug_name_canonical"
    ].notna()
].copy()


dataset_concrete_count = (
    valid_drugs
    .groupby(
        "dataset_id"
    )[
        "drug_name_canonical"
    ]
    .nunique()
    .to_dict()
)


dataset_concrete_names = (
    valid_drugs
    .groupby(
        "dataset_id"
    )[
        "drug_name_canonical"
    ]
    .agg(
        lambda x:
            ";".join(
                sorted(
                    set(
                        x.astype(str)
                    )
                )
            )
    )
    .to_dict()
)


# ============================================================
# UNPARSED TREATMENT COMPONENTS PER DATASET
# ============================================================

if len(
    unparsed
):

    dataset_unparsed_count = (
        unparsed
        .groupby(
            "dataset_id"
        )[
            "unparsed_component"
        ]
        .nunique()
        .to_dict()
    )

    dataset_unparsed_names = (
        unparsed
        .groupby(
            "dataset_id"
        )[
            "unparsed_component"
        ]
        .agg(
            lambda x:
                ";".join(
                    sorted(
                        set(
                            x.astype(str)
                        )
                    )
                )
        )
        .to_dict()
    )

else:

    dataset_unparsed_count = {}

    dataset_unparsed_names = {}


# ============================================================
# CLASSIFY DATASET × DRUG EVIDENCE
# ============================================================

matched_rows = valid_drugs[
    valid_drugs[
        "drug_name_canonical"
    ].astype(str).isin(
        approved_drugs
    )
].copy()


records = []


for _, row in matched_rows.iterrows():

    dataset_id = str(
        row[
            "dataset_id"
        ]
    )

    drug = str(
        row[
            "drug_name_canonical"
        ]
    )

    concrete_count = int(
        dataset_concrete_count.get(
            dataset_id,
            0,
        )
    )

    unparsed_count = int(
        dataset_unparsed_count.get(
            dataset_id,
            0,
        )
    )


    # --------------------------------------------------------
    # Evidence classification
    # --------------------------------------------------------
    #
    # Direct:
    # exactly one concrete drug
    # AND no additional unparsed treatment component.
    #
    # Contextual:
    # multiple concrete drugs
    # OR another treatment component is present.
    # --------------------------------------------------------

    if (
        concrete_count == 1
        and
        unparsed_count == 0
    ):

        evidence_context = (
            "direct_monotherapy"
        )

    else:

        evidence_context = (
            "combination_context"
        )


    records.append({
        "dataset_id":
            dataset_id,

        "cellres_drug":
            drug,

        "drugbank_id":
            row.get(
                "drugbank_id"
            ),

        "evidence_context":
            evidence_context,

        "concrete_drug_count":
            concrete_count,

        "concrete_drugs":
            dataset_concrete_names.get(
                dataset_id,
                "",
            ),

        "unparsed_component_count":
            unparsed_count,

        "unparsed_components":
            dataset_unparsed_names.get(
                dataset_id,
                "",
            ),

        "therapeutic_regimen":
            row.get(
                "therapeutic_regimen"
            ),

        "drug_type":
            row.get(
                "drug_type"
            ),

        "cancer_type":
            row.get(
                "cancer_type"
            ),

        "source_dataset":
            row.get(
                "source_dataset"
            ),

        "pmid":
            row.get(
                "pmid"
            ),
    })


evidence = pd.DataFrame(
    records
).drop_duplicates(
    subset=[
        "dataset_id",
        "cellres_drug",
    ]
)


# ============================================================
# PER-DRUG SUMMARY
# ============================================================

summary_rows = []


for drug, group in evidence.groupby(
    "cellres_drug"
):

    direct = group[
        group[
            "evidence_context"
        ]
        ==
        "direct_monotherapy"
    ]

    contextual = group[
        group[
            "evidence_context"
        ]
        ==
        "combination_context"
    ]


    map_row = approved_map[
        approved_map[
            "cellres_drug"
        ].astype(str)
        ==
        str(
            drug
        )
    ]


    keys = ""

    if len(
        map_row
    ):

        keys = str(
            map_row.iloc[0][
                "approved_pharos_keys"
            ]
        )


    summary_rows.append({
        "cellres_drug":
            drug,

        "approved_pharos_keys":
            keys,

        "total_datasets":
            group[
                "dataset_id"
            ].nunique(),

        "direct_monotherapy_datasets":
            direct[
                "dataset_id"
            ].nunique(),

        "combination_context_datasets":
            contextual[
                "dataset_id"
            ].nunique(),

        "has_direct_monotherapy":
            len(
                direct
            )
            > 0,

        "has_combination_context":
            len(
                contextual
            )
            > 0,

        "direct_cancers":
            ";".join(
                sorted(
                    set(
                        direct[
                            "cancer_type"
                        ]
                        .dropna()
                        .astype(str)
                    )
                )
            ),

        "contextual_cancers":
            ";".join(
                sorted(
                    set(
                        contextual[
                            "cancer_type"
                        ]
                        .dropna()
                        .astype(str)
                    )
                )
            ),
    })


summary = pd.DataFrame(
    summary_rows
).sort_values(
    [
        "has_direct_monotherapy",
        "total_datasets",
    ],
    ascending=[
        False,
        False,
    ],
)


# ============================================================
# PHAROS KEY SETS
# ============================================================

all_keys = set()

direct_keys = set()

context_only_keys = set()


for _, row in summary.iterrows():

    keys = {
        norm_key(
            x
        )
        for x in str(
            row[
                "approved_pharos_keys"
            ]
        ).split(";")
        if str(
            x
        ).strip()
    }


    all_keys.update(
        keys
    )


    if bool(
        row[
            "has_direct_monotherapy"
        ]
    ):

        direct_keys.update(
            keys
        )

    else:

        context_only_keys.update(
            keys
        )


# ============================================================
# DRUGCOMB COVERAGE
# ============================================================

a = response[
    "drug_a_key"
].map(
    norm_key
)

b = response[
    "drug_b_key"
].map(
    norm_key
)


all_mask = (
    a.isin(
        all_keys
    )
    |
    b.isin(
        all_keys
    )
)


direct_mask = (
    a.isin(
        direct_keys
    )
    |
    b.isin(
        direct_keys
    )
)


context_only_mask = (
    all_mask
    &
    ~direct_mask
)


def pct(mask):

    return (
        100.0
        * int(
            mask.sum()
        )
        / len(
            response
        )
    )


# ============================================================
# SAVE
# ============================================================

evidence_path = (
    OUT_DIR
    / "cellresdb_evidence_context_dataset_drug.csv"
)

summary_path = (
    OUT_DIR
    / "cellresdb_evidence_context_per_drug.csv"
)

coverage_path = (
    OUT_DIR
    / "cellresdb_evidence_context_coverage.csv"
)


evidence.to_csv(
    evidence_path,
    index=False,
)

summary.to_csv(
    summary_path,
    index=False,
)


coverage = pd.DataFrame([
    {
        "evidence_scheme":
            "all_matched",

        "pharos_key_count":
            len(
                all_keys
            ),

        "drugcomb_rows":
            int(
                all_mask.sum()
            ),

        "drugcomb_pct":
            pct(
                all_mask
            ),
    },

    {
        "evidence_scheme":
            "has_direct_monotherapy",

        "pharos_key_count":
            len(
                direct_keys
            ),

        "drugcomb_rows":
            int(
                direct_mask.sum()
            ),

        "drugcomb_pct":
            pct(
                direct_mask
            ),
    },

    {
        "evidence_scheme":
            "context_only_increment",

        "pharos_key_count":
            len(
                context_only_keys
            ),

        "drugcomb_rows":
            int(
                context_only_mask.sum()
            ),

        "drugcomb_pct":
            pct(
                context_only_mask
            ),
    },
])


coverage.to_csv(
    coverage_path,
    index=False,
)


# ============================================================
# REPORT
# ============================================================

print("=" * 100)
print("CELLRESDB EVIDENCE CONTEXT AUDIT")
print("=" * 100)

print(
    "Matched CellResDB drugs:",
    summary[
        "cellres_drug"
    ].nunique()
)

print(
    "Dataset × matched-drug evidence rows:",
    len(
        evidence
    )
)

print()

print(
    "Drugs with >=1 direct monotherapy dataset:",
    int(
        summary[
            "has_direct_monotherapy"
        ].sum()
    )
)

print(
    "Drugs with combination/context evidence:",
    int(
        summary[
            "has_combination_context"
        ].sum()
    )
)

print(
    "Drugs with ONLY contextual evidence:",
    int(
        (
            ~summary[
                "has_direct_monotherapy"
            ]
            &
            summary[
                "has_combination_context"
            ]
        ).sum()
    )
)


print()
print("=" * 100)
print("PER-DRUG EVIDENCE CONTEXT")
print("=" * 100)

print(
    summary[
        [
            "cellres_drug",
            "total_datasets",
            "direct_monotherapy_datasets",
            "combination_context_datasets",
            "direct_cancers",
            "contextual_cancers",
        ]
    ].to_string(
        index=False
    )
)


print()
print("=" * 100)
print("DRUGCOMB COVERAGE BY EVIDENCE QUALITY")
print("=" * 100)

print(
    coverage.to_string(
        index=False
    )
)


print()
print("=" * 100)
print("DIRECT MONOTHERAPY DATASETS")
print("=" * 100)

direct_rows = evidence[
    evidence[
        "evidence_context"
    ]
    ==
    "direct_monotherapy"
]

if len(
    direct_rows
):

    print(
        direct_rows[
            [
                "dataset_id",
                "cellres_drug",
                "therapeutic_regimen",
                "cancer_type",
                "source_dataset",
                "pmid",
            ]
        ]
        .sort_values(
            [
                "cellres_drug",
                "dataset_id",
            ]
        )
        .to_string(
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
    evidence_path
)

print(
    summary_path
)

print(
    coverage_path
)
