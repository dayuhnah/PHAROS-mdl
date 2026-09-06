from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm


REGISTRY_PATH = Path(
    "data/processed/v4/drug_registry.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/v4/pharos_mocktail_v4_morgan_fingerprints.npz"
)

RADIUS = 2
N_BITS = 2048


def main():

    df = pd.read_parquet(REGISTRY_PATH)

    df = df[
        df["canonical_smiles"].notna()
    ].drop_duplicates("drug_key").copy()

    print("Registry drugs:", f"{len(df):,}")

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=RADIUS,
        fpSize=N_BITS,
    )

    drug_names = []
    fingerprints = []
    failed = []

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
    ):

        drug = str(row["drug_key"])
        smiles = str(row["canonical_smiles"])

        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            failed.append(drug)
            continue

        fp = generator.GetFingerprintAsNumPy(mol)

        drug_names.append(drug)
        fingerprints.append(
            fp.astype(np.float32)
        )

    fingerprints = np.stack(
        fingerprints
    )

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
        fingerprints=fingerprints,
    )

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print("Drugs:", f"{len(drug_names):,}")
    print("Fingerprint shape:", fingerprints.shape)
    print("Fingerprint dtype:", fingerprints.dtype)
    print("Failed:", len(failed))
    print("Saved:", OUTPUT_PATH)

    if failed:
        print("Failed drugs:", failed[:20])


if __name__ == "__main__":
    main()
