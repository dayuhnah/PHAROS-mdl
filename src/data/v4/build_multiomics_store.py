from pathlib import Path
import re
import numpy as np
import pandas as pd

BASE = Path("data/raw/depmap")
DATA = Path("data/processed/v4/pharos_mocktail_v4_multiomics.parquet")
OUT = Path("data/processed/v4/multiomics")
OUT.mkdir(parents=True, exist_ok=True)

cells = sorted(
    pd.read_parquet(DATA)["depmap_id"]
    .astype(str)
    .unique()
)

print("Target cells:", len(cells))


def gene_name(x):
    return re.sub(r"\s*\(\d+\)$", "", str(x))


# =========================================================
# EXPRESSION
# =========================================================

expr = pd.read_csv(
    BASE / "OmicsExpressionProteinCodingGenesTPMLogp1.csv"
)

expr = expr.rename(columns={"Unnamed: 0": "ModelID"})
expr["ModelID"] = expr["ModelID"].astype(str)

expr = expr[
    expr["ModelID"].isin(cells)
].copy()

expr = expr.set_index("ModelID")

expr.columns = [
    gene_name(c) for c in expr.columns
]

expr.to_parquet(
    OUT / "expression.parquet"
)

print("Expression:", expr.shape)


# =========================================================
# CNV
# =========================================================

cnv = pd.read_csv(
    BASE / "OmicsCNGeneWGS.csv"
)

cnv["ModelID"] = cnv["ModelID"].astype(str)

cnv = cnv[
    cnv["ModelID"].isin(cells)
].copy()

# Prefer default DepMap record
if "IsDefaultEntryForModel" in cnv.columns:
    cnv = cnv.sort_values(
        "IsDefaultEntryForModel",
        ascending=False
    )

cnv = cnv.drop_duplicates(
    "ModelID",
    keep="first"
)

meta = [
    "Unnamed: 0",
    "SequencingID",
    "ModelConditionID",
    "ModelID",
    "IsDefaultEntryForMC",
    "IsDefaultEntryForModel",
]

cnv = cnv.drop(
    columns=[c for c in meta if c in cnv.columns]
)

cnv.index = (
    pd.read_csv(
        BASE / "OmicsCNGeneWGS.csv"
    )
    .query("ModelID in @cells")
    .sort_values(
        "IsDefaultEntryForModel",
        ascending=False
    )
    .drop_duplicates("ModelID")["ModelID"]
    .astype(str)
    .values[:len(cnv)]
)

cnv.columns = [
    gene_name(c) for c in cnv.columns
]

cnv.to_parquet(
    OUT / "cnv.parquet"
)

print("CNV:", cnv.shape)


# =========================================================
# CRISPR
# =========================================================

crispr = pd.read_csv(
    BASE / "CRISPRGeneEffect.csv"
)

crispr = crispr.rename(
    columns={"Unnamed: 0": "ModelID"}
)

crispr["ModelID"] = crispr["ModelID"].astype(str)

crispr = crispr[
    crispr["ModelID"].isin(cells)
].copy()

crispr = crispr.set_index("ModelID")

crispr.columns = [
    gene_name(c) for c in crispr.columns
]

crispr.to_parquet(
    OUT / "crispr.parquet"
)

print("CRISPR:", crispr.shape)


# =========================================================
# MUTATION
# =========================================================

mut = pd.read_csv(
    BASE / "OmicsSomaticMutations.csv",
    usecols=[
        "ModelID",
        "HugoSymbol",
        "IsDefaultEntryForModel",
    ]
)

mut["ModelID"] = mut["ModelID"].astype(str)

mut = mut[
    mut["ModelID"].isin(cells)
].copy()

if "IsDefaultEntryForModel" in mut.columns:
    mut = mut[
        mut["IsDefaultEntryForModel"] == "Yes"
    ].copy()

mut = mut[
    mut["HugoSymbol"].notna()
].copy()

# Binary mutation matrix
mutation = (
    mut.assign(value=1)
    .drop_duplicates(
        ["ModelID", "HugoSymbol"]
    )
    .pivot(
        index="ModelID",
        columns="HugoSymbol",
        values="value"
    )
    .fillna(0)
    .astype(np.int8)
)

mutation.to_parquet(
    OUT / "mutation.parquet"
)

print("Mutation:", mutation.shape)


print()
print("DONE")
