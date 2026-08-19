from pathlib import Path
import re

import pandas as pd


COMBO_DRUG_PATH = Path(
    "data/processed/combination/"
    "pharos_combo_unique_drugs.csv"
)

PHAROS_RESPONSE_PATH = Path(
    "data/processed/"
    "pharos_depmap_response_pairs.parquet"
)

OUTPUT_DIR = Path(
    "data/processed/combination"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


MATCHED_PATH = (
    OUTPUT_DIR
    / "combo_drugs_matched_to_pharos.csv"
)

UNMATCHED_PATH = (
    OUTPUT_DIR
    / "combo_drugs_unmatched.csv"
)


def normalize_drug_name(value):

    if pd.isna(value):
        return None

    value = str(value).upper()

    # Remove punctuation / spaces
    return re.sub(
        r"[^A-Z0-9]",
        "",
        value,
    )


def main():

    print(
        "========================================"
    )
    print(
        "PHAROS-COMBO SMILES COVERAGE"
    )
    print(
        "========================================"
    )

    # --------------------------------------
    # Load combination drug names
    # --------------------------------------

    combo_drugs = pd.read_csv(
        COMBO_DRUG_PATH
    )

    pharos = pd.read_parquet(
        PHAROS_RESPONSE_PATH,
        columns=[
            "name",
            "smiles",
        ],
    )

    pharos = pharos.dropna(
        subset=[
            "name",
            "smiles",
        ]
    ).copy()

    # --------------------------------------
    # Normalize names
    # --------------------------------------

    combo_drugs[
        "_drug_key"
    ] = combo_drugs[
        "drug_name"
    ].map(
        normalize_drug_name
    )

    pharos[
        "_drug_key"
    ] = pharos[
        "name"
    ].map(
        normalize_drug_name
    )

    # --------------------------------------
    # Build unambiguous name -> SMILES map
    # --------------------------------------

    grouped = (
        pharos[
            [
                "_drug_key",
                "name",
                "smiles",
            ]
        ]
        .drop_duplicates()
        .groupby(
            "_drug_key"
        )
    )

    smiles_map = {}

    ambiguous_keys = set()

    for key, group in grouped:

        unique_smiles = (
            group["smiles"]
            .dropna()
            .astype(str)
            .unique()
        )

        if len(unique_smiles) == 1:

            smiles_map[key] = (
                unique_smiles[0]
            )

        elif len(unique_smiles) > 1:

            ambiguous_keys.add(
                key
            )

    print(
        f"\nPHAROS unambiguous drug-name mappings: "
        f"{len(smiles_map):,}"
    )

    print(
        f"Ambiguous normalized PHAROS names: "
        f"{len(ambiguous_keys):,}"
    )

    # --------------------------------------
    # Map DrugComb drugs
    # --------------------------------------

    combo_drugs[
        "smiles"
    ] = combo_drugs[
        "_drug_key"
    ].map(
        smiles_map
    )

    combo_drugs[
        "match_status"
    ] = "unmatched"

    combo_drugs.loc[
        combo_drugs[
            "smiles"
        ].notna(),
        "match_status",
    ] = "matched"

    combo_drugs.loc[
        combo_drugs[
            "_drug_key"
        ].isin(
            ambiguous_keys
        ),
        "match_status",
    ] = "ambiguous"

    # --------------------------------------
    # Split results
    # --------------------------------------

    matched = combo_drugs[
        combo_drugs[
            "match_status"
        ]
        == "matched"
    ].copy()

    unmatched = combo_drugs[
        combo_drugs[
            "match_status"
        ]
        != "matched"
    ].copy()

    # --------------------------------------
    # Stats
    # --------------------------------------

    total = len(
        combo_drugs
    )

    matched_n = len(
        matched
    )

    unmatched_n = len(
        unmatched
    )

    print(
        f"\nCombination drugs: "
        f"{total:,}"
    )

    print(
        f"Matched to existing PHAROS SMILES: "
        f"{matched_n:,}"
    )

    print(
        f"Still unresolved: "
        f"{unmatched_n:,}"
    )

    print(
        f"Coverage: "
        f"{100 * matched_n / total:.2f}%"
    )

    print(
        f"\nAmbiguous DrugComb names: "
        f"{(
            combo_drugs[
                'match_status'
            ]
            == 'ambiguous'
        ).sum():,}"
    )

    print(
        "\nExample matched drugs:"
    )

    print(
        matched[
            [
                "drug_name",
                "smiles",
            ]
        ]
        .head(30)
        .to_string(
            index=False
        )
    )

    print(
        "\nExample unresolved drugs:"
    )

    print(
        unmatched[
            [
                "drug_name",
                "match_status",
            ]
        ]
        .head(50)
        .to_string(
            index=False
        )
    )

    # --------------------------------------
    # Save
    # --------------------------------------

    matched[
        [
            "drug_name",
            "smiles",
        ]
    ].to_csv(
        MATCHED_PATH,
        index=False,
    )

    unmatched[
        [
            "drug_name",
            "match_status",
        ]
    ].to_csv(
        UNMATCHED_PATH,
        index=False,
    )

    print(
        "\n========================================"
    )

    print(
        "SAVED"
    )

    print(
        "========================================"
    )

    print(
        f"Matched:\n"
        f"{MATCHED_PATH}"
    )

    print(
        f"\nUnmatched:\n"
        f"{UNMATCHED_PATH}"
    )


if __name__ == "__main__":
    main()