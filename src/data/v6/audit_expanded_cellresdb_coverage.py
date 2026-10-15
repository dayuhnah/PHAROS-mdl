from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq


MAP_PATH = Path(
    "outputs/v6/cellresdb_expansion/"
    "cellresdb_pharos_drug_overlap_audited.csv"
)

DATA_DIR = Path(
    "data/processed/v4"
)

OUT_DIR = Path(
    "outputs/v6/cellresdb_expansion"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def norm_key(x):

    if pd.isna(x):
        return ""

    return str(
        x
    ).strip().upper()


# ============================================================
# USE THE FULL UNSPLIT V4 MODEL POPULATION
# ============================================================

response_path = Path(
    "data/processed/v4/"
    "pharos_mocktail_v4_multiomics.parquet"
)

if not response_path.exists():
    raise FileNotFoundError(
        f"Missing response dataset: {response_path}"
    )

schema = pq.read_schema(
    response_path
)

required_cols = {
    "drug_a_key",
    "drug_b_key",
}

missing_cols = (
    required_cols
    - set(schema.names)
)

if missing_cols:
    raise RuntimeError(
        "Response dataset is missing required columns: "
        f"{sorted(missing_cols)}"
    )

row_count = (
    pq.ParquetFile(
        response_path
    )
    .metadata
    .num_rows
)

print("=" * 100)
print("PHAROS V4 RESPONSE DATASET")
print("=" * 100)

print(
    "Selected response file:",
    response_path
)

print(
    "Rows:",
    row_count
)

if row_count != 280584:
    raise RuntimeError(
        "Unexpected V4 population size: "
        f"{row_count:,}; expected 280,584."
    )

print(
    "✅ Full unsplit V4 population confirmed"
)

print()


# ============================================================
# LOAD AUDITED CELLRESDB MAPPING
# ============================================================

mapping = pd.read_csv(
    MAP_PATH
)


approved = mapping[
    mapping[
        "approved"
    ].astype(bool)
].copy()


expanded_keys = set()

drug_to_keys = {}


for _, row in approved.iterrows():

    drug = row[
        "cellres_drug"
    ]

    keys = [
        norm_key(x)
        for x in str(
            row[
                "approved_pharos_keys"
            ]
        ).split(";")
        if str(x).strip()
    ]

    drug_to_keys[
        drug
    ] = keys

    expanded_keys.update(
        keys
    )


# ============================================================
# OLD 5-DRUG CELLRESDB BANK
# ============================================================

old_drugs = {
    "Vismodegib",
    "Carfilzomib",
    "Dexamethasone",
    "Pomalidomide",
    "Paclitaxel",
}


old_keys = set()

for drug in old_drugs:

    old_keys.update(
        drug_to_keys.get(
            drug,
            []
        )
    )


print()
print("=" * 100)
print("EVIDENCE KEY SETS")
print("=" * 100)

print(
    "Old CellResDB drugs:",
    len(old_drugs)
)

print(
    "Old PHAROS keys:",
    len(old_keys)
)

print(
    "Expanded CellResDB drugs:",
    len(drug_to_keys)
)

print(
    "Expanded PHAROS keys:",
    len(expanded_keys)
)


# ============================================================
# LOAD ONLY REQUIRED RESPONSE COLUMNS
# ============================================================

response = pd.read_parquet(
    response_path
)


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


# ============================================================
# COVERAGE
# ============================================================

old_a = a.isin(
    old_keys
)

old_b = b.isin(
    old_keys
)

new_a = a.isin(
    expanded_keys
)

new_b = b.isin(
    expanded_keys
)


old_any = (
    old_a
    |
    old_b
)

old_both = (
    old_a
    &
    old_b
)


new_any = (
    new_a
    |
    new_b
)

new_both = (
    new_a
    &
    new_b
)


new_only = (
    new_any
    &
    ~old_any
)


def pct(mask):

    return (
        100.0
        * mask.sum()
        / len(response)
    )


print()
print("=" * 100)
print("FULL DRUGCOMB ROW COVERAGE")
print("=" * 100)

print(
    "Rows:",
    len(response)
)

print()

print(
    f"OLD any evidence      : "
    f"{old_any.sum():,} "
    f"({pct(old_any):.4f}%)"
)

print(
    f"OLD both drugs        : "
    f"{old_both.sum():,} "
    f"({pct(old_both):.4f}%)"
)

print()

print(
    f"EXPANDED any evidence : "
    f"{new_any.sum():,} "
    f"({pct(new_any):.4f}%)"
)

print(
    f"EXPANDED both drugs   : "
    f"{new_both.sum():,} "
    f"({pct(new_both):.4f}%)"
)

print()

print(
    f"NEWLY COVERED rows    : "
    f"{new_only.sum():,} "
    f"({pct(new_only):.4f}%)"
)


if old_any.sum() > 0:

    fold = (
        new_any.sum()
        /
        old_any.sum()
    )

    print(
        f"Coverage multiplier   : "
        f"{fold:.3f}x"
    )


# ============================================================
# UNIQUE DRUG-PAIR COVERAGE
# ============================================================

pairs = pd.DataFrame({
    "a":
        a,
    "b":
        b,
})


pairs[
    "pair"
] = [
    "||".join(
        sorted(
            (
                x,
                y,
            )
        )
    )
    for x, y
    in zip(
        pairs["a"],
        pairs["b"],
    )
]


print()
print("=" * 100)
print("UNIQUE DRUG-PAIR COVERAGE")
print("=" * 100)

print(
    "Total unique pairs:",
    pairs[
        "pair"
    ].nunique()
)

print(
    "Old covered unique pairs:",
    pairs.loc[
        old_any,
        "pair",
    ].nunique()
)

print(
    "Expanded covered unique pairs:",
    pairs.loc[
        new_any,
        "pair",
    ].nunique()
)


# ============================================================
# PER-CELLRESDB-DRUG COVERAGE
# ============================================================

coverage_rows = []


for drug, keys in drug_to_keys.items():

    key_set = set(
        keys
    )

    mask = (
        a.isin(
            key_set
        )
        |
        b.isin(
            key_set
        )
    )

    coverage_rows.append({
        "cellres_drug":
            drug,

        "pharos_keys":
            ";".join(
                keys
            ),

        "drugcomb_rows":
            int(
                mask.sum()
            ),

        "drugcomb_row_pct":
            pct(
                mask
            ),

        "unique_pairs":
            int(
                pairs.loc[
                    mask,
                    "pair",
                ].nunique()
            ),
    })


coverage_df = (
    pd.DataFrame(
        coverage_rows
    )
    .sort_values(
        "drugcomb_rows",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


print()
print("=" * 100)
print("PER-DRUG COVERAGE")
print("=" * 100)

print(
    coverage_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

coverage_path = (
    OUT_DIR
    / "expanded_cellresdb_drugcomb_coverage.csv"
)

coverage_df.to_csv(
    coverage_path,
    index=False,
)


summary = pd.DataFrame([
    {
        "scheme":
            "old_5_drug",

        "drug_count":
            len(old_drugs),

        "key_count":
            len(old_keys),

        "any_rows":
            int(
                old_any.sum()
            ),

        "any_pct":
            pct(
                old_any
            ),

        "both_rows":
            int(
                old_both.sum()
            ),

        "both_pct":
            pct(
                old_both
            ),
    },

    {
        "scheme":
            "expanded",

        "drug_count":
            len(drug_to_keys),

        "key_count":
            len(expanded_keys),

        "any_rows":
            int(
                new_any.sum()
            ),

        "any_pct":
            pct(
                new_any
            ),

        "both_rows":
            int(
                new_both.sum()
            ),

        "both_pct":
            pct(
                new_both
            ),
    },
])


summary_path = (
    OUT_DIR
    / "expanded_cellresdb_coverage_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False,
)


print()
print("=" * 100)
print("SAVED")
print("=" * 100)

print(
    coverage_path
)

print(
    summary_path
)
