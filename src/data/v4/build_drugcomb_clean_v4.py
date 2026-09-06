from pathlib import Path
import numpy as np
import pandas as pd


RAW_PATH = Path(
    "data/raw/combination/drugcombs_scored.csv"
)

CELL_MAP_PATH = Path(
    "data/processed/v4/drugcomb_depmap_cell_mapping_v4.csv"
)

OUT_PATH = Path(
    "data/processed/v4/pharos_combo_clean_v4.parquet"
)


print("=" * 70)
print("BUILDING PHAROS MOCKTAIL V4 CLEAN DRUGCOMB DATASET")
print("=" * 70)


# ------------------------------------------------------------
# Load
# ------------------------------------------------------------

df = pd.read_csv(RAW_PATH)
mapping = pd.read_csv(CELL_MAP_PATH)

print(f"Raw rows: {len(df):,}")


# ------------------------------------------------------------
# Cell mapping
# ------------------------------------------------------------

cell_map = dict(
    zip(
        mapping["drugcomb_cell"],
        mapping["depmap_id"],
    )
)

df["depmap_id"] = df["Cell line"].map(cell_map)

df = df[
    df["depmap_id"].notna()
].copy()

print(f"After V4 cell mapping: {len(df):,}")


# ------------------------------------------------------------
# Clean drug names
# ------------------------------------------------------------

df["Drug1"] = (
    df["Drug1"]
    .astype(str)
    .str.strip()
)

df["Drug2"] = (
    df["Drug2"]
    .astype(str)
    .str.strip()
)

df = df[
    (df["Drug1"] != "")
    &
    (df["Drug2"] != "")
].copy()


# ------------------------------------------------------------
# Numeric synergy scores
# ------------------------------------------------------------

score_cols = [
    "ZIP",
    "Bliss",
    "Loewe",
    "HSA",
]

for col in score_cols:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce",
    )


# ZIP is our required target
df = df[
    np.isfinite(df["ZIP"])
].copy()

print(f"After finite ZIP filter: {len(df):,}")


# Same pathological ZIP filtering used in V3
df = df[
    df["ZIP"].abs() <= 100
].copy()

print(f"After |ZIP| <= 100: {len(df):,}")


# ------------------------------------------------------------
# Canonicalize Drug A / Drug B
# ------------------------------------------------------------

drug_a = np.where(
    df["Drug1"] <= df["Drug2"],
    df["Drug1"],
    df["Drug2"],
)

drug_b = np.where(
    df["Drug1"] <= df["Drug2"],
    df["Drug2"],
    df["Drug1"],
)

df["drug_a"] = drug_a
df["drug_b"] = drug_b


# ------------------------------------------------------------
# Aggregate repeated pair-cell measurements
# ------------------------------------------------------------

group_cols = [
    "drug_a",
    "drug_b",
    "depmap_id",
]

clean = (
    df
    .groupby(
        group_cols,
        as_index=False,
    )
    .agg(
        zip_score=("ZIP", "median"),
        bliss_score=("Bliss", "median"),
        loewe_score=("Loewe", "median"),
        hsa_score=("HSA", "median"),
        replicate_count=("ID", "count"),
        original_cell_name=("Cell line", "first"),
    )
)


# ------------------------------------------------------------
# Final checks
# ------------------------------------------------------------

clean = clean.sort_values(
    [
        "drug_a",
        "drug_b",
        "depmap_id",
    ]
).reset_index(drop=True)


print()
print("-" * 70)
print("V4 CLEAN DATASET")
print("-" * 70)

print(f"Rows: {len(clean):,}")
print(
    "Unique pairs:",
    f"{clean[['drug_a', 'drug_b']].drop_duplicates().shape[0]:,}"
)
print(
    "Unique drugs:",
    f"{len(set(clean['drug_a']) | set(clean['drug_b'])):,}"
)
print(
    "Unique cells:",
    f"{clean['depmap_id'].nunique():,}"
)

print(
    "ZIP mean/std:",
    f"{clean['zip_score'].mean():.3f}",
    "/",
    f"{clean['zip_score'].std():.3f}",
)

print(
    "ZIP min/max:",
    f"{clean['zip_score'].min():.3f}",
    "/",
    f"{clean['zip_score'].max():.3f}",
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

OUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

clean.to_parquet(
    OUT_PATH,
    index=False,
)

print()
print("Saved:")
print(OUT_PATH)

print()
print("=" * 70)
print("DONE")
print("=" * 70)
