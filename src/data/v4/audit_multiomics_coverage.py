import pandas as pd

BASE = "data/raw/depmap"
MODEL_READY = "data/processed/v4/pharos_mocktail_v4_model_ready.parquet"

df = pd.read_parquet(MODEL_READY)
cells = set(df["depmap_id"].astype(str))

# Expression
expr = pd.read_csv(
    f"{BASE}/OmicsExpressionProteinCodingGenesTPMLogp1.csv",
    usecols=["Unnamed: 0"]
)
expr_cells = set(expr["Unnamed: 0"].astype(str))

# CNV
cnv = pd.read_csv(
    f"{BASE}/OmicsCNGeneWGS.csv",
    usecols=["ModelID"]
)
cnv_cells = set(cnv["ModelID"].dropna().astype(str))

# Mutation
mut = pd.read_csv(
    f"{BASE}/OmicsSomaticMutations.csv",
    usecols=["ModelID"]
)
mut_cells = set(mut["ModelID"].dropna().astype(str))

# CRISPR
crispr = pd.read_csv(
    f"{BASE}/CRISPRGeneEffect.csv",
    usecols=["Unnamed: 0"]
)
crispr_cells = set(crispr["Unnamed: 0"].astype(str))


def report(name, available):
    matched = cells & available
    missing = cells - available

    print(f"{name}:")
    print(f"  Covered: {len(matched)}/{len(cells)}")
    print(f"  Missing: {len(missing)}")
    print()


print("=" * 50)
print("PHAROS V4 MULTI-OMICS COVERAGE")
print("=" * 50)

print("Mocktail cells:", len(cells))
print()

report("Expression", expr_cells)
report("CNV", cnv_cells)
report("Mutation", mut_cells)
report("CRISPR", crispr_cells)

all_four = (
    cells
    & expr_cells
    & cnv_cells
    & mut_cells
    & crispr_cells
)

print("All four modalities:")
print(f"  Covered: {len(all_four)}/{len(cells)}")
print(f"  Missing at least one: {len(cells - all_four)}")
