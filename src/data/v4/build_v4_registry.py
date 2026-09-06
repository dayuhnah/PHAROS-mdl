from pathlib import Path
import re
import pandas as pd
from rdkit import Chem


CLEAN_PATH = Path(
    "data/processed/v4/pharos_combo_clean_v4.parquet"
)

OLD_PATH = Path(
    "data/processed/combination/combo_drug_smiles_local_clean.csv"
)

PUBCHEM_PATH = Path(
    "data/processed/v4/pubchem_drug_resolution_v4.csv"
)

OUT_DIR = Path("data/processed/v4")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def norm(x):
    return re.sub(
        r"[^A-Z0-9]",
        "",
        str(x).upper()
    )


def canonicalize(smiles):
    if pd.isna(smiles):
        return None

    mol = Chem.MolFromSmiles(str(smiles))

    if mol is None:
        return None

    return Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=True
    )


# ============================================================
# LOAD
# ============================================================

clean = pd.read_parquet(CLEAN_PATH)
old = pd.read_csv(OLD_PATH)
new = pd.read_csv(PUBCHEM_PATH)


# ============================================================
# FIND OLD SMILES COLUMN
# ============================================================

smiles_candidates = [
    "canonical_parent_smiles",
    "canonical_smiles",
    "smiles",
    "SMILES",
]

old_smiles_col = next(
    (
        c for c in smiles_candidates
        if c in old.columns
    ),
    None
)

if old_smiles_col is None:
    raise ValueError(
        f"No SMILES column found in old file. "
        f"Columns: {list(old.columns)}"
    )


# ============================================================
# BUILD DRUG REGISTRY
# ============================================================

records = {}


# ------------------------------------------------------------
# Existing local structures
# ------------------------------------------------------------

for _, row in old.iterrows():

    key = norm(row["drug_key"])

    smiles = canonicalize(
        row[old_smiles_col]
    )

    if smiles is None:
        continue

    records[key] = {
        "drug_key": key,
        "drug_name": row.get(
            "drug_name",
            row["drug_key"]
        ),
        "canonical_smiles": smiles,
        "source": "existing_local",
        "confidence": "high",
        "pubchem_cid": None,
        "inchikey": None,
    }


# ------------------------------------------------------------
# PubChem additions
# ------------------------------------------------------------

new = new[
    new["status"] == "resolved"
].copy()

for _, row in new.iterrows():

    key = norm(row["drug_key"])

    # Preserve existing local mapping if already present
    if key in records:
        continue

    smiles = canonicalize(
        row["canonical_parent_smiles"]
    )

    if smiles is None:
        continue

    records[key] = {
        "drug_key": key,
        "drug_name": row[
            "representative_name"
        ],
        "canonical_smiles": smiles,
        "source": row[
            "resolution_source"
        ],
        "confidence": row[
            "confidence"
        ],
        "pubchem_cid": row[
            "pubchem_cid"
        ],
        "inchikey": row[
            "pubchem_inchikey"
        ],
    }


registry = pd.DataFrame(
    records.values()
)

registry = registry.sort_values(
    "drug_key"
).reset_index(drop=True)


# ============================================================
# SAVE DRUG REGISTRY
# ============================================================

registry_path = (
    OUT_DIR / "drug_registry.parquet"
)

registry.to_parquet(
    registry_path,
    index=False
)


# ============================================================
# BUILD MODEL-READY DATASET
# ============================================================

smiles_map = dict(
    zip(
        registry["drug_key"],
        registry["canonical_smiles"]
    )
)

clean["drug_a_key"] = (
    clean["drug_a"].map(norm)
)

clean["drug_b_key"] = (
    clean["drug_b"].map(norm)
)

clean["drug_a_smiles"] = (
    clean["drug_a_key"].map(smiles_map)
)

clean["drug_b_smiles"] = (
    clean["drug_b_key"].map(smiles_map)
)


model_ready = clean[
    clean["drug_a_smiles"].notna()
    &
    clean["drug_b_smiles"].notna()
].copy()


# ============================================================
# SAVE MODEL-READY DATA
# ============================================================

model_path = (
    OUT_DIR /
    "pharos_mocktail_v4_model_ready.parquet"
)

model_ready.to_parquet(
    model_path,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("=" * 60)
print("PHAROS MOCKTAIL V4 DATASET")
print("=" * 60)

print(
    "Registry drugs:",
    f"{len(registry):,}"
)

print(
    "Clean rows:",
    f"{len(clean):,}"
)

print(
    "Model-ready rows:",
    f"{len(model_ready):,}"
)

print(
    "Coverage:",
    f"{100 * len(model_ready) / len(clean):.2f}%"
)

print(
    "Unique model-ready pairs:",
    f"{model_ready[['drug_a','drug_b']].drop_duplicates().shape[0]:,}"
)

print(
    "Unique model-ready drugs:",
    f"{len(set(model_ready['drug_a']) | set(model_ready['drug_b'])):,}"
)

print(
    "Unique model-ready cells:",
    f"{model_ready['depmap_id'].nunique():,}"
)

print()
print("Saved:")
print(registry_path)
print(model_path)
