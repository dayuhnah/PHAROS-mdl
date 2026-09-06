from pathlib import Path
import re

import pandas as pd


# ============================================================
# Paths
# ============================================================

CLEAN_PATH = Path(
    "data/processed/combination/pharos_combo_clean.parquet"
)

RESOLVED_PATH = Path(
    "data/processed/combination/combo_drug_smiles_local_clean.csv"
)

UNRESOLVED_PATH = Path(
    "data/processed/combination/combo_drugs_still_unresolved.csv"
)

AMBIGUOUS_PATH = Path(
    "data/processed/combination/combo_drugs_ambiguous_smiles.csv"
)

CELL_MAP_PATH = Path(
    "data/processed/combination/drugcomb_depmap_cell_mapping.csv"
)

UNMATCHED_CELL_PATH = Path(
    "data/processed/combination/drugcomb_unmatched_cells.csv"
)

RAW_DRUGCOMB_PATH = Path(
    "data/raw/combination/drugcombs_scored.csv"
)

OUTPUT_DIR = Path(
    "outputs/v4_audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Helpers
# ============================================================

def normalize_drug_name(value):
    """
    Match the existing PHAROS drug-key convention as closely
    as possible.

    Examples:
        (+)-JQ1       -> JQ1
        BMS-754807    -> BMS754807
    """
    if pd.isna(value):
        return ""

    value = str(value).upper()

    return re.sub(
        r"[^A-Z0-9]",
        "",
        value,
    )


def normalize_cell_name(value):
    if pd.isna(value):
        return ""

    value = str(value).upper()

    return re.sub(
        r"[^A-Z0-9]",
        "",
        value,
    )


def find_column(columns, candidates):
    """
    Find a column while being tolerant to spaces,
    underscores and capitalization.
    """

    normalized = {
        re.sub(
            r"[^a-z0-9]",
            "",
            str(c).lower(),
        ): c
        for c in columns
    }

    for candidate in candidates:
        key = re.sub(
            r"[^a-z0-9]",
            "",
            candidate.lower(),
        )

        if key in normalized:
            return normalized[key]

    return None


# ============================================================
# Load current V3 information
# ============================================================

print("=" * 80)
print("PHAROS MOCKTAIL V4 COVERAGE AUDIT")
print("=" * 80)

clean = pd.read_parquet(
    CLEAN_PATH
)

resolved = pd.read_csv(
    RESOLVED_PATH
)

unresolved = pd.read_csv(
    UNRESOLVED_PATH
)

ambiguous = pd.read_csv(
    AMBIGUOUS_PATH
)

cell_map = pd.read_csv(
    CELL_MAP_PATH
)

unmatched_cells = pd.read_csv(
    UNMATCHED_CELL_PATH
)


# ============================================================
# Determine resolved drug keys
# ============================================================

resolved_key_col = find_column(
    resolved.columns,
    [
        "drug_key",
        "drug key",
        "key",
    ],
)

resolved_name_col = find_column(
    resolved.columns,
    [
        "drug_name",
        "drug name",
        "name",
    ],
)

if resolved_key_col is not None:
    resolved_keys = set(
        resolved[
            resolved_key_col
        ]
        .astype(str)
        .map(normalize_drug_name)
    )

elif resolved_name_col is not None:
    resolved_keys = set(
        resolved[
            resolved_name_col
        ]
        .astype(str)
        .map(normalize_drug_name)
    )

else:
    raise RuntimeError(
        "Could not identify drug key/name column "
        "in resolved structure table."
    )


ambiguous_keys = set(
    ambiguous[
        "drug_key"
    ]
    .astype(str)
    .map(normalize_drug_name)
)

unresolved_keys = set(
    unresolved[
        "drug_key"
    ]
    .astype(str)
    .map(normalize_drug_name)
)


# ============================================================
# Drug pair resolution audit
# ============================================================

clean = clean.copy()

clean["row_id"] = range(
    len(clean)
)

clean["drug_a_key_v4"] = (
    clean["drug_a"]
    .map(normalize_drug_name)
)

clean["drug_b_key_v4"] = (
    clean["drug_b"]
    .map(normalize_drug_name)
)

clean["a_resolved"] = (
    clean["drug_a_key_v4"]
    .isin(resolved_keys)
)

clean["b_resolved"] = (
    clean["drug_b_key_v4"]
    .isin(resolved_keys)
)

both_resolved = (
    clean["a_resolved"]
    &
    clean["b_resolved"]
)

exactly_one = (
    clean["a_resolved"]
    ^
    clean["b_resolved"]
)

neither_resolved = (
    ~clean["a_resolved"]
    &
    ~clean["b_resolved"]
)


print()
print("-" * 80)
print("DRUG STRUCTURE COVERAGE")
print("-" * 80)

print(
    f"Clean pair-cell rows: "
    f"{len(clean):,}"
)

print(
    f"Both drugs resolved: "
    f"{both_resolved.sum():,} "
    f"({both_resolved.mean() * 100:.2f}%)"
)

print(
    f"Exactly one drug resolved: "
    f"{exactly_one.sum():,} "
    f"({exactly_one.mean() * 100:.2f}%)"
)

print(
    f"Neither drug resolved: "
    f"{neither_resolved.sum():,} "
    f"({neither_resolved.mean() * 100:.2f}%)"
)


# ============================================================
# Build missing-drug impact table
# ============================================================

missing_records = []


for side, partner in [
    ("a", "b"),
    ("b", "a"),
]:

    drug_col = f"drug_{side}"
    key_col = f"drug_{side}_key_v4"

    resolved_col = (
        f"{side}_resolved"
    )

    partner_resolved_col = (
        f"{partner}_resolved"
    )

    subset = clean[
        ~clean[resolved_col]
    ]

    for row in subset.itertuples(
        index=False
    ):
        row_dict = row._asdict()

        key = row_dict[key_col]

        if key in ambiguous_keys:
            status = "ambiguous"

        elif key in unresolved_keys:
            status = "unresolved"

        else:
            status = "unknown"

        missing_records.append(
            {
                "row_id": row_dict["row_id"],
                "drug_name": row_dict[drug_col],
                "drug_key": key,
                "status": status,
                "partner_resolved": bool(
                    row_dict[
                        partner_resolved_col
                    ]
                ),
            }
        )


missing = pd.DataFrame(
    missing_records
)


impact = (
    missing
    .groupby(
        [
            "drug_key",
            "status",
        ],
        as_index=False,
    )
    .agg(
        representative_name=(
            "drug_name",
            "first",
        ),
        affected_rows=(
            "row_id",
            "nunique",
        ),
        rows_immediately_recoverable=(
            "partner_resolved",
            "sum",
        ),
    )
)


impact = impact.sort_values(
    [
        "rows_immediately_recoverable",
        "affected_rows",
    ],
    ascending=False,
)


impact.to_csv(
    OUTPUT_DIR
    / "unresolved_drug_row_impact.csv",
    index=False,
)


print()
print(
    "Top 30 unresolved/ambiguous drugs "
    "by immediately recoverable rows:"
)

print(
    impact.head(30)
    .to_string(
        index=False
    )
)


# ============================================================
# Current cell matching audit
# ============================================================

mapped_cell_keys = set(
    cell_map[
        "Cell line"
    ]
    .map(normalize_cell_name)
)

unmatched_cell_keys = set(
    unmatched_cells[
        "Cell line"
    ]
    .map(normalize_cell_name)
)


print()
print("-" * 80)
print("CELL-LINE COVERAGE")
print("-" * 80)

print(
    f"Mapped DrugComb cell names: "
    f"{len(mapped_cell_keys):,}"
)

print(
    f"Unmatched DrugComb cell names: "
    f"{len(unmatched_cell_keys):,}"
)


# ============================================================
# Count raw DrugComb rows by cell using chunks
# ============================================================

if RAW_DRUGCOMB_PATH.exists():

    header = pd.read_csv(
        RAW_DRUGCOMB_PATH,
        nrows=0,
    )

    cell_col = find_column(
        header.columns,
        [
            "Cell line",
            "cell_line",
            "cell line name",
            "cell",
        ],
    )

    if cell_col is None:

        print()
        print(
            "Could not automatically identify "
            "cell-line column in raw DrugComb."
        )

        print(
            "Raw columns:"
        )

        for c in header.columns:
            print(
                " ",
                c,
            )

    else:

        print()
        print(
            f"Raw DrugComb cell column: "
            f"{cell_col}"
        )

        raw_counts = {}

        total_raw_rows = 0

        for chunk in pd.read_csv(
            RAW_DRUGCOMB_PATH,
            usecols=[cell_col],
            chunksize=200_000,
        ):

            total_raw_rows += len(
                chunk
            )

            keys = (
                chunk[cell_col]
                .map(normalize_cell_name)
            )

            counts = (
                keys
                .value_counts()
            )

            for key, count in counts.items():

                raw_counts[key] = (
                    raw_counts.get(
                        key,
                        0,
                    )
                    + int(count)
                )

        cell_rows = []

        for _, row in unmatched_cells.iterrows():

            raw_name = row[
                "Cell line"
            ]

            key = normalize_cell_name(
                raw_name
            )

            cell_rows.append(
                {
                    "cell_line": raw_name,
                    "cell_key": key,
                    "raw_rows": raw_counts.get(
                        key,
                        0,
                    ),
                }
            )

        unmatched_impact = pd.DataFrame(
            cell_rows
        )

        unmatched_impact = (
            unmatched_impact
            .sort_values(
                "raw_rows",
                ascending=False,
            )
        )

        unmatched_impact.to_csv(
            OUTPUT_DIR
            / "unmatched_cell_row_impact.csv",
            index=False,
        )


        matched_raw_rows = sum(
            raw_counts.get(
                key,
                0,
            )
            for key
            in mapped_cell_keys
        )

        unmatched_raw_rows = sum(
            raw_counts.get(
                key,
                0,
            )
            for key
            in unmatched_cell_keys
        )


        print()
        print(
            f"Raw DrugComb rows: "
            f"{total_raw_rows:,}"
        )

        print(
            f"Rows belonging to currently "
            f"mapped cells: "
            f"{matched_raw_rows:,}"
        )

        print(
            f"Rows belonging to currently "
            f"unmatched cells: "
            f"{unmatched_raw_rows:,}"
        )

        print()
        print(
            "Top 30 unmatched cell lines "
            "by raw DrugComb row count:"
        )

        print(
            unmatched_impact.head(30)
            .to_string(
                index=False
            )
        )


# ============================================================
# Save complete row classification
# ============================================================

clean[
    [
        "drug_a",
        "drug_b",
        "depmap_id",
        "zip_score",
        "a_resolved",
        "b_resolved",
    ]
].to_parquet(
    OUTPUT_DIR
    / "clean_row_resolution_status.parquet",
    index=False,
)


print()
print("=" * 80)
print("AUDIT COMPLETE")
print("=" * 80)

print(
    "Saved:"
)

print(
    OUTPUT_DIR
    / "unresolved_drug_row_impact.csv"
)

print(
    OUTPUT_DIR
    / "unmatched_cell_row_impact.csv"
)

print(
    OUTPUT_DIR
    / "clean_row_resolution_status.parquet"
)
