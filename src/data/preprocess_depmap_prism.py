from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

RESPONSE_PATH = RAW_DIR / "primary-screen-replicate-collapsed-logfold-change.csv"
EXPRESSION_PATH = RAW_DIR / "OmicsExpressionProteinCodingGenesTPMLogp1.csv"
TREATMENT_INFO_PATH = RAW_DIR / "prism-repurposing-20q2-primary-screen-replicate-treatment-info.csv"

OUTPUT_PAIRS_PATH = PROCESSED_DIR / "pharos_depmap_response_pairs.parquet"
OUTPUT_EXPRESSION_PATH = PROCESSED_DIR / "pharos_depmap_expression.parquet"


def load_response() -> pd.DataFrame:
    print("Loading PRISM response matrix...")
    response = pd.read_csv(RESPONSE_PATH)

    response = response.rename(columns={response.columns[0]: "depmap_id"})

    print(f"Response matrix shape: {response.shape}")

    response_long = response.melt(
        id_vars="depmap_id",
        var_name="treatment",
        value_name="logfold_change",
    )

    response_long = response_long.dropna(subset=["logfold_change"]).copy()

    # Treatment format:
    # BRD-A00077618-236-07-6::2.5::HTS
    parts = response_long["treatment"].str.split("::", expand=True)

    response_long["broad_id"] = parts[0]
    response_long["dose"] = pd.to_numeric(parts[1], errors="coerce")
    response_long["screen"] = parts[2]

    print(f"Response long shape: {response_long.shape}")
    return response_long


def load_expression() -> pd.DataFrame:
    print("Loading DepMap expression matrix...")
    expression = pd.read_csv(EXPRESSION_PATH)
    expression = expression.rename(columns={expression.columns[0]: "depmap_id"})

    print(f"Expression matrix shape: {expression.shape}")
    return expression


def load_treatment_info() -> pd.DataFrame:
    print("Loading PRISM treatment info...")
    treatment_info = pd.read_csv(TREATMENT_INFO_PATH)

    treatment_info["broad_id"] = treatment_info["broad_id"].astype(str)
    treatment_info["dose"] = pd.to_numeric(treatment_info["dose"], errors="coerce")

    keep_cols = [
        "broad_id",
        "name",
        "dose",
        "perturbation_type",
        "screen_id",
        "moa",
        "target",
        "disease.area",
        "indication",
        "smiles",
        "phase",
    ]

    keep_cols = [col for col in keep_cols if col in treatment_info.columns]
    treatment_info = treatment_info[keep_cols].drop_duplicates()

    print(f"Treatment info shape after keeping useful columns: {treatment_info.shape}")
    return treatment_info


def main() -> None:
    response_long = load_response()
    expression = load_expression()
    treatment_info = load_treatment_info()

    common_cells = sorted(set(response_long["depmap_id"]) & set(expression["depmap_id"]))
    print(f"Common DepMap IDs between response and expression: {len(common_cells)}")

    response_long = response_long[response_long["depmap_id"].isin(common_cells)].copy()
    expression = expression[expression["depmap_id"].isin(common_cells)].copy()

    print("Merging response pairs with treatment metadata...")
    merged = response_long.merge(
        treatment_info,
        on=["broad_id", "dose"],
        how="left",
    )

    smiles_coverage = merged["smiles"].notna().mean() * 100
    name_coverage = merged["name"].notna().mean() * 100

    print(f"Final response pairs shape: {merged.shape}")
    print(f"Final expression matrix shape: {expression.shape}")
    print(f"Compound name coverage: {name_coverage:.2f}%")
    print(f"SMILES coverage: {smiles_coverage:.2f}%")

    print("\nPreview of processed response pairs:")
    preview_cols = [
        "depmap_id",
        "broad_id",
        "name",
        "dose",
        "logfold_change",
        "moa",
        "target",
        "smiles",
    ]
    preview_cols = [col for col in preview_cols if col in merged.columns]
    print(merged[preview_cols].head())

    merged.to_parquet(OUTPUT_PAIRS_PATH, index=False)
    expression.to_parquet(OUTPUT_EXPRESSION_PATH, index=False)

    print(f"\nSaved response pairs to: {OUTPUT_PAIRS_PATH}")
    print(f"Saved expression matrix to: {OUTPUT_EXPRESSION_PATH}")


if __name__ == "__main__":
    main()