from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import clamp


# ============================================================
# Paths
# ============================================================

INPUT_PATH = Path(
    "data/processed/combination/pharos_combo_drug_table.csv"
)

OUTPUT_PATH = Path(
    "data/processed/combination/pharos_combo_clamp_embeddings.npz"
)

MODEL_DIR = Path(
    "data/models/clamp_clip"
)


# ============================================================
# Settings
# ============================================================

BATCH_SIZE = 32

DEVICE = "cpu"


# ============================================================
# Find columns automatically
# ============================================================

def find_column(
    df,
    candidates,
    description,
):

    lower_to_original = {
        str(col).lower(): col
        for col in df.columns
    }

    for candidate in candidates:

        key = candidate.lower()

        if key in lower_to_original:

            return lower_to_original[
                key
            ]

    raise ValueError(
        f"Could not find {description} column.\n"
        f"Available columns: {list(df.columns)}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "======================================"
    )

    print(
        "PHAROS-COMBO CLAMP EMBEDDING CACHE"
    )

    print(
        "======================================"
    )

    # --------------------------------------------------------
    # Load drug table
    # --------------------------------------------------------

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            f"Missing input file:\n"
            f"{INPUT_PATH}"
        )

    print(
        f"\nLoading:\n{INPUT_PATH}"
    )

    df = pd.read_csv(
        INPUT_PATH
    )

    print(
        f"Rows in drug table: "
        f"{len(df):,}"
    )

    print(
        f"Columns: "
        f"{list(df.columns)}"
    )

    # --------------------------------------------------------
    # Identify drug / SMILES columns
    # --------------------------------------------------------

    drug_col = find_column(

        df,

        candidates=[
            "drug",
            "drug_name",
            "name",
            "compound",
            "compound_name",
        ],

        description="drug name",
    )

    smiles_col = find_column(

        df,

        candidates=[
            "smiles",
            "canonical_smiles",
            "canonicalsmiles",
        ],

        description="SMILES",
    )

    print(
        f"\nDrug column: {drug_col}"
    )

    print(
        f"SMILES column: {smiles_col}"
    )

    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    df = (
        df[
            [
                drug_col,
                smiles_col,
            ]
        ]
        .dropna()
        .copy()
    )

    df[
        drug_col
    ] = (
        df[
            drug_col
        ]
        .astype(str)
        .str.strip()
    )

    df[
        smiles_col
    ] = (
        df[
            smiles_col
        ]
        .astype(str)
        .str.strip()
    )

    df = df[
        (
            df[drug_col] != ""
        )
        &
        (
            df[smiles_col] != ""
        )
    ]

    # One embedding per drug
    df = (
        df
        .drop_duplicates(
            subset=[
                drug_col
            ]
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"\nUsable unique drugs: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # Load pretrained CLAMP
    # --------------------------------------------------------

    print(
        "\nLoading pretrained CLAMP..."
    )

    model = clamp.CLAMP(

        path_dir=str(
            MODEL_DIR
        ),

        device=DEVICE,
    )

    model.eval()

    print(
        "✅ Pretrained CLAMP loaded"
    )

    # --------------------------------------------------------
    # Encode
    # --------------------------------------------------------

    drug_names = (
        df[
            drug_col
        ]
        .tolist()
    )

    smiles = (
        df[
            smiles_col
        ]
        .tolist()
    )

    embedding_batches = []

    print(
        "\nGenerating frozen CLAMP embeddings..."
    )

    for start in tqdm(
        range(
            0,
            len(smiles),
            BATCH_SIZE,
        ),
        desc="CLAMP",
    ):

        end = min(
            start + BATCH_SIZE,
            len(smiles),
        )

        batch_smiles = (
            smiles[
                start:end
            ]
        )

        with torch.no_grad():

            embeddings = (
                model.encode_smiles(
                    batch_smiles
                )
            )

        embeddings = (
            embeddings
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        embedding_batches.append(
            embeddings
        )

    embeddings = np.concatenate(
        embedding_batches,
        axis=0,
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print(
        "\n======================================"
    )

    print(
        "CLAMP CACHE SUMMARY"
    )

    print(
        "======================================"
    )

    print(
        f"Drugs encoded: "
        f"{len(drug_names):,}"
    )

    print(
        f"Embedding matrix: "
        f"{embeddings.shape}"
    )

    print(
        f"Embedding dtype: "
        f"{embeddings.dtype}"
    )

    if (
        embeddings.shape[0]
        != len(drug_names)
    ):

        raise RuntimeError(
            "Embedding count does not "
            "match drug count."
        )

    if not np.isfinite(
        embeddings
    ).all():

        raise RuntimeError(
            "CLAMP embeddings contain "
            "NaN or infinite values."
        )

    print(
        "Finite embeddings: YES"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(

        OUTPUT_PATH,

        drug_names=np.asarray(
            drug_names,
            dtype=str,
        ),

        smiles=np.asarray(
            smiles,
            dtype=str,
        ),

        embeddings=embeddings,
    )

    print(
        "\n✅ CLAMP CACHE CREATED"
    )

    print(
        f"\nSaved to:\n"
        f"{OUTPUT_PATH}"
    )

    print(
        "\nExample:"
    )

    print(
        f"Drug: {drug_names[0]}"
    )

    print(
        f"SMILES: {smiles[0]}"
    )

    print(
        f"Embedding shape: "
        f"{embeddings[0].shape}"
    )

    print(
        f"First 10 values:\n"
        f"{embeddings[0][:10]}"
    )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    main()