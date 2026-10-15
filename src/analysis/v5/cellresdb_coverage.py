from pathlib import Path
import re
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]

CELLRES_DRUGS = ROOT / "data/raw/cellresdb/Supplementary_Data_2.xlsx"
REGISTRY = ROOT / "data/processed/v4/drug_registry.parquet"
COMBO = ROOT / "data/processed/v4/pharos_combo_clean_v4.parquet"

OUTDIR = ROOT / "data/processed/v5/cellresdb"
OUTDIR.mkdir(parents=True, exist_ok=True)


def normalize_name(x):
    if pd.isna(x):
        return None

    x = str(x).strip().lower()

    # Normalize punctuation / spacing
    x = re.sub(r"[^a-z0-9]+", "", x)

    return x or None


def find_column(columns, candidates):
    lower_map = {str(c).lower(): c for c in columns}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    return None


# ============================================================
# 1. Load CellResDB drugs
# ============================================================

cellres = pd.read_excel(
    CELLRES_DRUGS,
    sheet_name="Sheet1",
    header=1,
)

cellres.columns = [
    "drug",
    "drugbank_id",
    "frequency",
    "treatment_type",
]

cellres["drug"] = cellres["drug"].astype(str).str.strip()
cellres["normalized_drug"] = cellres["drug"].map(normalize_name)

print("\n" + "=" * 80)
print("CELLRESDB")
print("=" * 80)

print("CellResDB drugs:", len(cellres))
print("\nTreatment types:")
print(cellres["treatment_type"].value_counts(dropna=False))

print("\nExample drugs:")
print(cellres.head(10).to_string(index=False))


# ============================================================
# 2. Load PHAROS drug registry
# ============================================================

registry = pd.read_parquet(REGISTRY)

print("\n" + "=" * 80)
print("PHAROS DRUG REGISTRY")
print("=" * 80)

print("Shape:", registry.shape)
print("Columns:", list(registry.columns))


registry_name_col = find_column(
    registry.columns,
    [
        "canonical_name",
        "drug_name",
        "drug",
        "name",
        "raw_drug_name",
        "normalized_name",
        "canonical_drug",
        "canonical_key",
        "drug_key",
    ],
)

if registry_name_col is None:
    raise RuntimeError(
        "Could not automatically find drug-name column in registry.\n"
        f"Columns found: {list(registry.columns)}"
    )

print("Using registry drug column:", registry_name_col)

registry["normalized_drug"] = (
    registry[registry_name_col]
    .map(normalize_name)
)


# ============================================================
# 3. Match CellResDB drugs to PHAROS
# ============================================================

registry_names = set(
    registry["normalized_drug"].dropna().unique()
)

cellres["matched_pharos"] = (
    cellres["normalized_drug"].isin(registry_names)
)

matched = cellres[cellres["matched_pharos"]].copy()
unmatched = cellres[~cellres["matched_pharos"]].copy()

print("\n" + "=" * 80)
print("DRUG MATCHING")
print("=" * 80)

print(
    f"Matched CellResDB drugs: "
    f"{len(matched)}/{len(cellres)} "
    f"({100 * len(matched) / len(cellres):.2f}%)"
)

print("\nMatched drugs:")
print(
    matched[
        ["drug", "drugbank_id", "treatment_type"]
    ].to_string(index=False)
)

print("\nUnmatched drugs:")
print(
    unmatched[
        ["drug", "drugbank_id", "treatment_type"]
    ].to_string(index=False)
)


# ============================================================
# 4. Load PHAROS combination dataset
# ============================================================

combo = pd.read_parquet(COMBO)

print("\n" + "=" * 80)
print("PHAROS COMBINATION DATA")
print("=" * 80)

print("Shape:", combo.shape)
print("Columns:", list(combo.columns))


drug_a_col = find_column(
    combo.columns,
    [
        "drug_a_key",
        "drug_a",
        "drug1",
        "drug_1",
        "drug_row",
        "drugA",
        "canonical_drug_a",
        "drug_a_name",
    ],
)

drug_b_col = find_column(
    combo.columns,
    [
        "drug_b_key",
        "drug_b",
        "drug2",
        "drug_2",
        "drug_col",
        "drugB",
        "canonical_drug_b",
        "drug_b_name",
    ],
)

if drug_a_col is None or drug_b_col is None:
    raise RuntimeError(
        "Could not automatically identify Drug A / Drug B columns.\n"
        f"Columns found: {list(combo.columns)}"
    )

print("Drug A column:", drug_a_col)
print("Drug B column:", drug_b_col)


combo["_drug_a_norm"] = combo[drug_a_col].map(normalize_name)
combo["_drug_b_norm"] = combo[drug_b_col].map(normalize_name)

matched_names = set(
    matched["normalized_drug"].dropna()
)

combo["cellres_a"] = combo["_drug_a_norm"].isin(matched_names)
combo["cellres_b"] = combo["_drug_b_norm"].isin(matched_names)

combo["cellres_any"] = (
    combo["cellres_a"] | combo["cellres_b"]
)

combo["cellres_both"] = (
    combo["cellres_a"] & combo["cellres_b"]
)


# ============================================================
# 5. Row-level coverage
# ============================================================

total_rows = len(combo)

rows_any = int(combo["cellres_any"].sum())
rows_both = int(combo["cellres_both"].sum())

print("\n" + "=" * 80)
print("ROW COVERAGE")
print("=" * 80)

print(f"Total rows: {total_rows:,}")

print(
    f"≥1 CellResDB drug: "
    f"{rows_any:,} "
    f"({100 * rows_any / total_rows:.2f}%)"
)

print(
    f"Both drugs in CellResDB: "
    f"{rows_both:,} "
    f"({100 * rows_both / total_rows:.2f}%)"
)


# ============================================================
# 6. Unique pair coverage
# ============================================================

pairs = (
    combo[
        [
            "_drug_a_norm",
            "_drug_b_norm",
            "cellres_any",
            "cellres_both",
        ]
    ]
    .drop_duplicates(
        ["_drug_a_norm", "_drug_b_norm"]
    )
)

total_pairs = len(pairs)
pairs_any = int(pairs["cellres_any"].sum())
pairs_both = int(pairs["cellres_both"].sum())

print("\n" + "=" * 80)
print("PAIR COVERAGE")
print("=" * 80)

print(f"Unique drug pairs: {total_pairs:,}")

print(
    f"Pairs with ≥1 CellResDB drug: "
    f"{pairs_any:,} "
    f"({100 * pairs_any / total_pairs:.2f}%)"
)

print(
    f"Pairs with both CellResDB drugs: "
    f"{pairs_both:,} "
    f"({100 * pairs_both / total_pairs:.2f}%)"
)


# ============================================================
# 7. Save everything
# ============================================================

cellres.to_csv(
    OUTDIR / "cellresdb_drugs.csv",
    index=False,
)

matched.to_csv(
    OUTDIR / "cellresdb_matched_drugs.csv",
    index=False,
)

unmatched.to_csv(
    OUTDIR / "cellresdb_unmatched_drugs.csv",
    index=False,
)

combo[
    [
        drug_a_col,
        drug_b_col,
        "cellres_a",
        "cellres_b",
        "cellres_any",
        "cellres_both",
    ]
].to_parquet(
    OUTDIR / "pharos_cellresdb_coverage.parquet",
    index=False,
)

summary = pd.DataFrame(
    [
        {
            "cellresdb_drugs": len(cellres),
            "matched_cellresdb_drugs": len(matched),
            "matched_drug_percent":
                100 * len(matched) / len(cellres),

            "pharos_rows": total_rows,
            "rows_any_cellresdb": rows_any,
            "rows_any_percent":
                100 * rows_any / total_rows,

            "rows_both_cellresdb": rows_both,
            "rows_both_percent":
                100 * rows_both / total_rows,

            "unique_pairs": total_pairs,
            "pairs_any_cellresdb": pairs_any,
            "pairs_any_percent":
                100 * pairs_any / total_pairs,

            "pairs_both_cellresdb": pairs_both,
            "pairs_both_percent":
                100 * pairs_both / total_pairs,
        }
    ]
)

summary.to_csv(
    OUTDIR / "cellresdb_coverage_summary.csv",
    index=False,
)

print("\nSaved to:")
print(OUTDIR)

print("\nDONE.")
