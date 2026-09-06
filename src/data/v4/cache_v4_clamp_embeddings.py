from pathlib import Path

import clamp
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


REGISTRY_PATH = Path(
    "data/processed/v4/drug_registry.parquet"
)

MODEL_DIR = Path(
    "data/models/clamp_clip"
)

OUTPUT_PATH = Path(
    "data/processed/v4/pharos_mocktail_v4_clamp_embeddings.npz"
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

BATCH_SIZE = 128


def main():

    print("=" * 60)
    print("PHAROS MOCKTAIL V4 CLAMP CACHE")
    print("=" * 60)

    # --------------------------------------------------------
    # Load V4 drug registry
    # --------------------------------------------------------

    df = pd.read_parquet(REGISTRY_PATH)

    required = [
        "drug_key",
        "canonical_smiles",
    ]

    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"Missing column: {col}"
            )

    df = df[
        df["canonical_smiles"].notna()
    ].copy()

    df["drug_key"] = (
        df["drug_key"]
        .astype(str)
    )

    df["canonical_smiles"] = (
        df["canonical_smiles"]
        .astype(str)
    )

    df = df.drop_duplicates(
        "drug_key"
    ).reset_index(drop=True)

    print(
        "Registry drugs:",
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # Load pretrained CLAMP
    # --------------------------------------------------------

    print(
        "\nLoading pretrained CLAMP..."
    )

    model = clamp.CLAMP(
        path_dir=str(MODEL_DIR),
        device=DEVICE,
    )

    model = model.to(DEVICE)
    model.eval()

    print(
        "CLAMP loaded on:",
        DEVICE
    )

    # --------------------------------------------------------
    # Encode canonical SMILES
    # --------------------------------------------------------

    drug_names = (
        df["drug_key"].tolist()
    )

    smiles = (
        df["canonical_smiles"].tolist()
    )

    all_embeddings = []

    print(
        "\nEncoding drugs..."
    )

    for start in tqdm(
        range(
            0,
            len(smiles),
            BATCH_SIZE,
        )
    ):

        batch = smiles[
            start:start + BATCH_SIZE
        ]

        with torch.no_grad():

            emb = model.encode_smiles(
                batch
            )

        if torch.is_tensor(emb):
            emb = (
                emb
                .detach()
                .cpu()
                .numpy()
            )
        else:
            emb = np.asarray(emb)

        all_embeddings.append(
            emb.astype(np.float32)
        )

    embeddings = np.concatenate(
        all_embeddings,
        axis=0,
    )

    # --------------------------------------------------------
    # Checks
    # --------------------------------------------------------

    if embeddings.shape[0] != len(
        drug_names
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
            "NaN or Inf values."
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

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)

    print(
        "Drugs:",
        f"{len(drug_names):,}"
    )

    print(
        "Embedding shape:",
        embeddings.shape
    )

    print(
        "Embedding dtype:",
        embeddings.dtype
    )

    print(
        "Saved:",
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()
