from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

CELLRES = ROOT / "data/raw/cellresdb/Supplementary_Data_2.xlsx"
REGISTRY = ROOT / "data/processed/v4/drug_registry.parquet"
V4_SPLIT = ROOT / "data/processed/v4/training/mocktail_v4_split_seed42.parquet"

OUTDIR = ROOT / "data/processed/v5/cellresdb"
OUTDIR.mkdir(parents=True, exist_ok=True)


def norm(x):
    if pd.isna(x):
        return None
    return re.sub(r"[^a-z0-9]+", "", str(x).strip().lower()) or None


# Only aliases where the identity is clear.
SAFE_ALIASES = {
    "albumboundpaclitaxel": "paclitaxel",
    "albuminboundpaclitaxel": "paclitaxel",
    "pomalidomidep": "pomalidomide",
}


# ------------------------------------------------------------
# CellResDB
# ------------------------------------------------------------
cellres = pd.read_excel(CELLRES, header=1)
cellres.columns = [
    "drug",
    "drugbank_id",
    "frequency",
    "treatment_type",
]

cellres["normalized_original"] = cellres["drug"].map(norm)

cellres["normalized_drug"] = cellres["normalized_original"].map(
    lambda x: SAFE_ALIASES.get(x, x)
)

cellres["alias_used"] = (
    cellres["normalized_original"] != cellres["normalized_drug"]
)


# ------------------------------------------------------------
# PHAROS registry
# ------------------------------------------------------------
registry = pd.read_parquet(REGISTRY)

registry["normalized_drug"] = registry["drug_name"].map(norm)

pharos_names = set(
    registry["normalized_drug"].dropna().unique()
)

cellres["matched_pharos"] = (
    cellres["normalized_drug"].isin(pharos_names)
)

matched = cellres[cellres["matched_pharos"]].copy()
unmatched = cellres[~cellres["matched_pharos"]].copy()


# ------------------------------------------------------------
# Exact V4 modelling population
# ------------------------------------------------------------
v4 = pd.read_parquet(V4_SPLIT)

print("=" * 80)
print("V4 SPLIT")
print("=" * 80)
print("Shape:", v4.shape)
print("Columns:", list(v4.columns))

if "drug_a" not in v4.columns or "drug_b" not in v4.columns:
    raise RuntimeError(
        "Expected drug_a and drug_b columns.\n"
        f"Found: {list(v4.columns)}"
    )

v4["_drug_a_norm"] = v4["drug_a"].map(norm)
v4["_drug_b_norm"] = v4["drug_b"].map(norm)

matched_names = set(matched["normalized_drug"])

v4["cellres_a"] = v4["_drug_a_norm"].isin(matched_names)
v4["cellres_b"] = v4["_drug_b_norm"].isin(matched_names)

v4["cellres_any"] = v4["cellres_a"] | v4["cellres_b"]
v4["cellres_both"] = v4["cellres_a"] & v4["cellres_b"]


# ------------------------------------------------------------
# Statistics
# ------------------------------------------------------------
n_rows = len(v4)
rows_any = int(v4["cellres_any"].sum())
rows_both = int(v4["cellres_both"].sum())

pairs = (
    v4[
        [
            "_drug_a_norm",
            "_drug_b_norm",
            "cellres_any",
            "cellres_both",
        ]
    ]
    .drop_duplicates(["_drug_a_norm", "_drug_b_norm"])
)

n_pairs = len(pairs)
pairs_any = int(pairs["cellres_any"].sum())
pairs_both = int(pairs["cellres_both"].sum())


print("\n" + "=" * 80)
print("REFINED DRUG MATCHING")
print("=" * 80)

print(
    f"Matched CellResDB drugs: "
    f"{len(matched)}/{len(cellres)} "
    f"({100 * len(matched) / len(cellres):.2f}%)"
)

print("\nMatched:")
print(
    matched[
        [
            "drug",
            "drugbank_id",
            "treatment_type",
            "alias_used",
            "normalized_drug",
        ]
    ].to_string(index=False)
)

print("\nStill unmatched:")
print(
    unmatched[
        ["drug", "drugbank_id", "treatment_type"]
    ].to_string(index=False)
)


print("\n" + "=" * 80)
print("EXACT V4 MODELLING ROW COVERAGE")
print("=" * 80)

print(f"Total V4 rows: {n_rows:,}")

print(
    f"≥1 CellResDB drug: "
    f"{rows_any:,} "
    f"({100 * rows_any / n_rows:.2f}%)"
)

print(
    f"Both CellResDB drugs: "
    f"{rows_both:,} "
    f"({100 * rows_both / n_rows:.2f}%)"
)


print("\n" + "=" * 80)
print("EXACT V4 PAIR COVERAGE")
print("=" * 80)

print(f"Unique pairs: {n_pairs:,}")

print(
    f"≥1 CellResDB drug: "
    f"{pairs_any:,} "
    f"({100 * pairs_any / n_pairs:.2f}%)"
)

print(
    f"Both CellResDB drugs: "
    f"{pairs_both:,} "
    f"({100 * pairs_both / n_pairs:.2f}%)"
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------
matched.to_csv(
    OUTDIR / "cellresdb_matched_drugs_refined.csv",
    index=False,
)

unmatched.to_csv(
    OUTDIR / "cellresdb_unmatched_drugs_refined.csv",
    index=False,
)

summary = pd.DataFrame([{
    "cellresdb_drugs": len(cellres),
    "matched_cellresdb_drugs": len(matched),
    "matched_drug_percent":
        100 * len(matched) / len(cellres),

    "v4_rows": n_rows,
    "rows_any_cellresdb": rows_any,
    "rows_any_percent":
        100 * rows_any / n_rows,

    "rows_both_cellresdb": rows_both,
    "rows_both_percent":
        100 * rows_both / n_rows,

    "v4_pairs": n_pairs,
    "pairs_any_cellresdb": pairs_any,
    "pairs_any_percent":
        100 * pairs_any / n_pairs,

    "pairs_both_cellresdb": pairs_both,
    "pairs_both_percent":
        100 * pairs_both / n_pairs,
}])

summary.to_csv(
    OUTDIR / "cellresdb_coverage_refined_summary.csv",
    index=False,
)

print("\nSaved:")
print(OUTDIR / "cellresdb_coverage_refined_summary.csv")
print("\nDONE.")
