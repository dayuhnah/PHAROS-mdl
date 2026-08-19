from pathlib import Path
import re
import unicodedata

import pandas as pd


COMBO_PATH = Path(
    "data/processed/combination/pharos_combo_local_smiles.parquet"
)

MAPPING_PATH = Path(
    "data/processed/combination/scdrugact/"
    "pharos_scdrugact_drug_mapping.csv"
)


def normalize_name(value):
    if pd.isna(value):
        return ""

    text = str(value).strip()

    text = unicodedata.normalize(
        "NFKD",
        text,
    )

    text = (
        text
        .encode(
            "ascii",
            "ignore",
        )
        .decode(
            "ascii"
        )
    )

    text = text.lower()

    return re.sub(
        r"[^a-z0-9]+",
        "",
        text,
    )


def find_drug_columns(df):

    candidates_a = [
        "drug1",
        "drug_1",
        "drug_a",
        "druga",
        "Drug1",
        "Drug A",
    ]

    candidates_b = [
        "drug2",
        "drug_2",
        "drug_b",
        "drugb",
        "Drug2",
        "Drug B",
    ]

    lower_map = {
        str(col).lower(): col
        for col in df.columns
    }

    drug_a = None
    drug_b = None

    for candidate in candidates_a:

        if candidate.lower() in lower_map:

            drug_a = lower_map[
                candidate.lower()
            ]

            break

    for candidate in candidates_b:

        if candidate.lower() in lower_map:

            drug_b = lower_map[
                candidate.lower()
            ]

            break

    if drug_a is None or drug_b is None:

        raise ValueError(
            "Could not automatically detect "
            "drug columns.\n"
            f"Columns available:\n{list(df.columns)}"
        )

    return (
        drug_a,
        drug_b,
    )


print(
    "=" * 60
)

print(
    "PHAROS × SCDRUGACT COMBINATION COVERAGE"
)

print(
    "=" * 60
)


# ============================================================
# Load PHAROS combinations
# ============================================================

combo = pd.read_parquet(
    COMBO_PATH
)

drug_a_col, drug_b_col = (
    find_drug_columns(
        combo
    )
)


print(
    "\nCombo rows:",
    f"{len(combo):,}"
)

print(
    "Drug A column:",
    drug_a_col
)

print(
    "Drug B column:",
    drug_b_col
)


# ============================================================
# Load scDrugAct mapping
# ============================================================

mapping = pd.read_csv(
    MAPPING_PATH
)

matched = mapping[
    mapping["matched"] == True
].copy()


matched_names = {
    normalize_name(drug)
    for drug in matched[
        "pharos_drug"
    ]
}


print(
    "\nMatched PHAROS drugs:",
    f"{len(matched_names):,}"
)


# ============================================================
# Row matching
# ============================================================

combo["_drug_a_norm"] = (
    combo[drug_a_col]
    .map(normalize_name)
)

combo["_drug_b_norm"] = (
    combo[drug_b_col]
    .map(normalize_name)
)


combo["_a_scdrugact"] = (
    combo["_drug_a_norm"]
    .isin(
        matched_names
    )
)

combo["_b_scdrugact"] = (
    combo["_drug_b_norm"]
    .isin(
        matched_names
    )
)


both = (
    combo["_a_scdrugact"]
    & combo["_b_scdrugact"]
)

at_least_one = (
    combo["_a_scdrugact"]
    | combo["_b_scdrugact"]
)

a_only = (
    combo["_a_scdrugact"]
    & ~combo["_b_scdrugact"]
)

b_only = (
    ~combo["_a_scdrugact"]
    & combo["_b_scdrugact"]
)

neither = (
    ~combo["_a_scdrugact"]
    & ~combo["_b_scdrugact"]
)


def report(
    name,
    mask,
):

    count = int(
        mask.sum()
    )

    percent = (
        100
        * count
        / len(combo)
    )

    print(
        f"{name:<25}"
        f"{count:>8,}"
        f"   ({percent:6.2f}%)"
    )


print(
    "\n======================================"
)

print(
    "ROW COVERAGE"
)

print(
    "======================================"
)


report(
    "Drug A matched:",
    combo["_a_scdrugact"],
)

report(
    "Drug B matched:",
    combo["_b_scdrugact"],
)

report(
    "At least one matched:",
    at_least_one,
)

report(
    "Both drugs matched:",
    both,
)

report(
    "Only Drug A matched:",
    a_only,
)

report(
    "Only Drug B matched:",
    b_only,
)

report(
    "Neither matched:",
    neither,
)


# ============================================================
# Pair-level coverage
# ============================================================

pair_df = (
    combo[
        [
            "_drug_a_norm",
            "_drug_b_norm",
            "_a_scdrugact",
            "_b_scdrugact",
        ]
    ]
    .drop_duplicates(
        subset=[
            "_drug_a_norm",
            "_drug_b_norm",
        ]
    )
)


pair_at_least_one = (
    pair_df["_a_scdrugact"]
    | pair_df["_b_scdrugact"]
)

pair_both = (
    pair_df["_a_scdrugact"]
    & pair_df["_b_scdrugact"]
)


print(
    "\n======================================"
)

print(
    "UNIQUE PAIR COVERAGE"
)

print(
    "======================================"
)


print(
    "Unique pairs:",
    f"{len(pair_df):,}"
)


print(
    "Pairs with at least one "
    "scDrugAct drug:",
    f"{pair_at_least_one.sum():,}",
    f"({100 * pair_at_least_one.mean():.2f}%)"
)


print(
    "Pairs with BOTH "
    "scDrugAct drugs:",
    f"{pair_both.sum():,}",
    f"({100 * pair_both.mean():.2f}%)"
)


# ============================================================
# Most frequent matched drugs
# ============================================================

all_drugs = pd.concat(
    [
        combo[
            [
                drug_a_col
            ]
        ]
        .rename(
            columns={
                drug_a_col:
                "drug"
            }
        ),

        combo[
            [
                drug_b_col
            ]
        ]
        .rename(
            columns={
                drug_b_col:
                "drug"
            }
        ),
    ],
    ignore_index=True,
)


all_drugs[
    "normalized"
] = (
    all_drugs[
        "drug"
    ]
    .map(
        normalize_name
    )
)


matched_frequency = (
    all_drugs[
        all_drugs[
            "normalized"
        ]
        .isin(
            matched_names
        )
    ]
    ["drug"]
    .value_counts()
    .head(
        20
    )
)


print(
    "\n======================================"
)

print(
    "TOP MATCHED DRUGS BY ROW FREQUENCY"
)

print(
    "======================================"
)


print(
    matched_frequency
    .to_string()
)