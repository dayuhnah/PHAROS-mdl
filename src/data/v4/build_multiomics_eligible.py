import pandas as pd
from pathlib import Path

BASE = Path("data/raw/depmap")
IN_PATH = Path(
    "data/processed/v4/pharos_mocktail_v4_model_ready.parquet"
)
OUT_PATH = Path(
    "data/processed/v4/pharos_mocktail_v4_multiomics.parquet"
)

df = pd.read_parquet(IN_PATH)


def ids(path, column):
    return set(
        pd.read_csv(
            path,
            usecols=[column]
        )[column]
        .dropna()
        .astype(str)
    )


expr = ids(
    BASE / "OmicsExpressionProteinCodingGenesTPMLogp1.csv",
    "Unnamed: 0"
)

cnv = ids(
    BASE / "OmicsCNGeneWGS.csv",
    "ModelID"
)

mut = ids(
    BASE / "OmicsSomaticMutations.csv",
    "ModelID"
)

crispr = ids(
    BASE / "CRISPRGeneEffect.csv",
    "Unnamed: 0"
)


cell = df["depmap_id"].astype(str)

df["has_expression"] = cell.isin(expr)
df["has_cnv"] = cell.isin(cnv)
df["has_mutation"] = cell.isin(mut)
df["has_crispr"] = cell.isin(crispr)

df["omics_count"] = (
    df[
        [
            "has_expression",
            "has_cnv",
            "has_mutation",
            "has_crispr",
        ]
    ]
    .astype(int)
    .sum(axis=1)
)

eligible = df[
    df["omics_count"] > 0
].copy()

eligible.to_parquet(
    OUT_PATH,
    index=False
)

print("Original:", f"{len(df):,}")
print("Eligible:", f"{len(eligible):,}")
print("Dropped no-omics:", f"{(df['omics_count'] == 0).sum():,}")
print("Coverage:", f"{len(eligible)/len(df)*100:.2f}%")
print("Cells:", eligible["depmap_id"].nunique())
print("Saved:", OUT_PATH)
