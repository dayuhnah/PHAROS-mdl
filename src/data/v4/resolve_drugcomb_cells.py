from pathlib import Path
import re
import pandas as pd

MODEL_PATH = Path("data/raw/depmap/Model.csv")
RAW_PATH = Path("data/raw/combination/drugcombs_scored.csv")
OLD_MAP_PATH = Path(
    "data/processed/combination/drugcomb_depmap_cell_mapping.csv"
)

OUT_DIR = Path("data/processed/v4")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def norm(x):
    return re.sub(
        r"[^A-Z0-9]",
        "",
        str(x).upper()
    )


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

models = pd.read_csv(MODEL_PATH)
raw = pd.read_csv(RAW_PATH, usecols=["Cell line"])
old_map = pd.read_csv(OLD_MAP_PATH)

raw_cells = sorted(
    raw["Cell line"].dropna().unique()
)

# ------------------------------------------------------------
# Existing trusted V3 mappings
# ------------------------------------------------------------

existing = {}

for _, r in old_map.iterrows():
    existing[norm(r["Cell line"])] = r["depmap_id"]

# ------------------------------------------------------------
# Build full DepMap name lookup
# ------------------------------------------------------------

model_candidates = {}

for _, r in models.iterrows():

    model_id = r["ModelID"]

    for col in [
        "CellLineName",
        "StrippedCellLineName",
    ]:

        value = r.get(col)

        if pd.isna(value):
            continue

        key = norm(value)

        model_candidates.setdefault(
            key,
            set()
        ).add(model_id)

# ------------------------------------------------------------
# Resolve DrugComb cells
# ------------------------------------------------------------

resolved = []
ambiguous = []
unmatched = []

for cell in raw_cells:

    key = norm(cell)

    # Preserve existing known mappings first
    if key in existing:

        resolved.append({
            "drugcomb_cell": cell,
            "cell_key": key,
            "depmap_id": existing[key],
            "match_method": "existing_v3_mapping",
            "confidence": "high",
        })

        continue

    candidates = model_candidates.get(
        key,
        set()
    )

    if len(candidates) == 1:

        resolved.append({
            "drugcomb_cell": cell,
            "cell_key": key,
            "depmap_id": next(iter(candidates)),
            "match_method": "full_depmap_name_match",
            "confidence": "high",
        })

    elif len(candidates) > 1:

        ambiguous.append({
            "drugcomb_cell": cell,
            "cell_key": key,
            "candidate_model_ids": "|".join(
                sorted(candidates)
            ),
        })

    else:

        unmatched.append({
            "drugcomb_cell": cell,
            "cell_key": key,
        })


resolved = pd.DataFrame(resolved)
ambiguous = pd.DataFrame(ambiguous)
unmatched = pd.DataFrame(unmatched)

# ------------------------------------------------------------
# Count DrugComb rows covered
# ------------------------------------------------------------

raw["_key"] = raw["Cell line"].map(norm)

resolved_keys = set(
    resolved["cell_key"]
)

covered_rows = raw["_key"].isin(
    resolved_keys
).sum()

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

resolved.to_csv(
    OUT_DIR / "drugcomb_depmap_cell_mapping_v4.csv",
    index=False,
)

ambiguous.to_csv(
    OUT_DIR / "drugcomb_ambiguous_cells_v4.csv",
    index=False,
)

unmatched.to_csv(
    OUT_DIR / "drugcomb_unmatched_cells_v4.csv",
    index=False,
)

print("=" * 60)
print("PHAROS V4 CELL RESOLUTION")
print("=" * 60)

print("DrugComb cell names:", len(raw_cells))
print("Resolved:", len(resolved))
print("Ambiguous:", len(ambiguous))
print("Unmatched:", len(unmatched))

print()
print(
    "Raw DrugComb rows covered:",
    f"{covered_rows:,}"
)

print(
    "Raw DrugComb rows total:",
    f"{len(raw):,}"
)

print()
print("Saved V4 cell mapping.")
