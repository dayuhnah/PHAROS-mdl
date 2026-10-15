from pathlib import Path
import re
import pandas as pd


CATALOG_PATH = Path(
    "data/external/cellresdb_full/catalog/"
    "cellresdb_full_catalog.csv"
)

DRUG_PATH = Path(
    "data/external/cellresdb_full/catalog/"
    "cellresdb_full_drugs.csv"
)

OUT_DIR = Path(
    "data/external/cellresdb_full/catalog"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CANONICAL DRUGBANK-ID OVERRIDES
# ============================================================
#
# These correct source naming/formulation inconsistencies
# without changing the underlying DrugBank identity.
#

DRUGBANK_CANONICAL = {
    "DB08910": "Pomalidomide",
    "DB01229": "Paclitaxel",
}


def clean(value):
    if pd.isna(value):
        return None

    value = re.sub(
        r"\s+",
        " ",
        str(value),
    ).strip()

    return value or None


def split_regimen(regimen):
    """
    Split regimen into top-level '+' components.
    """

    if not regimen:
        return []

    return [
        x.strip()
        for x in regimen.split("+")
        if x.strip()
    ]


def strip_drugbank_annotation(component):
    """
    Example:
        Pembrolizumab(DB09037)
        ->
        Pembrolizumab
    """

    return re.sub(
        r"\s*\(DB\d+\)\s*",
        "",
        component,
        flags=re.I,
    ).strip()


catalog = pd.read_csv(
    CATALOG_PATH
)

drugs = pd.read_csv(
    DRUG_PATH
)


# ============================================================
# CANONICALISE PARSED DRUGS
# ============================================================

drugs[
    "drug_name_original"
] = drugs[
    "drug_name"
]


def canonical_name(row):

    dbid = clean(
        row.get(
            "drugbank_id"
        )
    )

    original = clean(
        row.get(
            "drug_name"
        )
    )

    if dbid in DRUGBANK_CANONICAL:
        return DRUGBANK_CANONICAL[
            dbid
        ]

    return original


drugs[
    "drug_name_canonical"
] = drugs.apply(
    canonical_name,
    axis=1,
)


drugs[
    "canonicalization_changed"
] = (
    drugs[
        "drug_name_original"
    ].fillna("")
    !=
    drugs[
        "drug_name_canonical"
    ].fillna("")
)


# ============================================================
# FIND UNPARSED REGIMEN COMPONENTS
# ============================================================

parsed_by_dataset = {}

for dataset_id, group in drugs.groupby(
    "dataset_id"
):

    parsed = set()

    for _, row in group.iterrows():

        dbid = clean(
            row.get(
                "drugbank_id"
            )
        )

        original = clean(
            row.get(
                "drug_name_original"
            )
        )

        canonical = clean(
            row.get(
                "drug_name_canonical"
            )
        )

        if dbid:
            parsed.add(
                dbid.lower()
            )

        if original:
            parsed.add(
                original.lower()
            )

        if canonical:
            parsed.add(
                canonical.lower()
            )

    parsed_by_dataset[
        str(dataset_id)
    ] = parsed


unparsed_rows = []

for _, row in catalog.iterrows():

    dataset_id = str(
        row["Database_ID"]
    )

    regimen = clean(
        row.get(
            "Therapeutic_regimen"
        )
    )

    known = parsed_by_dataset.get(
        dataset_id,
        set(),
    )

    for component in split_regimen(
        regimen
    ):

        dbids = re.findall(
            r"DB\d+",
            component,
            flags=re.I,
        )

        component_name = (
            strip_drugbank_annotation(
                component
            )
        )

        component_is_parsed = False

        if any(
            dbid.lower() in known
            for dbid in dbids
        ):
            component_is_parsed = True

        if (
            component_name.lower()
            in known
        ):
            component_is_parsed = True

        if not component_is_parsed:

            unparsed_rows.append({
                "dataset_id":
                    dataset_id,

                "standard_disease":
                    row.get(
                        "standardDisease"
                    ),

                "drug_type":
                    row.get(
                        "Drug_type"
                    ),

                "therapeutic_regimen":
                    regimen,

                "unparsed_component":
                    component_name,
            })


unparsed = pd.DataFrame(
    unparsed_rows
)


# ============================================================
# SAVE
# ============================================================

clean_path = (
    OUT_DIR
    / "cellresdb_full_drugs_clean.csv"
)

unparsed_path = (
    OUT_DIR
    / "cellresdb_unparsed_regimen_components.csv"
)


drugs.to_csv(
    clean_path,
    index=False,
)

unparsed.to_csv(
    unparsed_path,
    index=False,
)


# ============================================================
# REPORT
# ============================================================

print("=" * 100)
print("CELLRESDB DRUG CATALOG CLEANING")
print("=" * 100)

print(
    "Drug rows:",
    len(drugs)
)

print(
    "Unique original names:",
    drugs[
        "drug_name_original"
    ].dropna().nunique()
)

print(
    "Unique canonical names:",
    drugs[
        "drug_name_canonical"
    ].dropna().nunique()
)

print(
    "Unique DrugBank IDs:",
    drugs[
        "drugbank_id"
    ].dropna().nunique()
)


print()
print("=" * 100)
print("CANONICALIZATION CHANGES")
print("=" * 100)

changed = drugs[
    drugs[
        "canonicalization_changed"
    ]
]

if len(changed):

    print(
        changed[
            [
                "dataset_id",
                "drug_name_original",
                "drug_name_canonical",
                "drugbank_id",
            ]
        ]
        .drop_duplicates()
        .to_string(
            index=False
        )
    )

else:
    print("None")


print()
print("=" * 100)
print("UNPARSED REGIMEN COMPONENTS")
print("=" * 100)

if len(unparsed):

    summary = (
        unparsed[
            "unparsed_component"
        ]
        .value_counts()
        .rename_axis(
            "component"
        )
        .reset_index(
            name="dataset_count"
        )
    )

    print(
        summary.to_string(
            index=False
        )
    )

else:
    print("None ✅")


print()
print("=" * 100)
print("OUTPUTS")
print("=" * 100)

print(clean_path)
print(unparsed_path)
