from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.featurize_smiles import smiles_to_morgan_fp


PROCESSED_DIR = Path("data/processed")
RESPONSE_PATH = PROCESSED_DIR / "pharos_depmap_response_pairs.parquet"
OUTPUT_PATH = PROCESSED_DIR / "pharos_drug_fingerprints.npz"


def main():
    print("Loading response pairs...")
    response = pd.read_parquet(RESPONSE_PATH)

    response = response.dropna(subset=["broad_id", "smiles"]).copy()

    drugs = (
        response[["broad_id", "smiles"]]
        .drop_duplicates(subset=["broad_id"])
        .reset_index(drop=True)
    )

    print(f"Unique drugs with SMILES: {len(drugs)}")

    broad_ids = []
    fingerprints = []

    for _, row in tqdm(drugs.iterrows(), total=len(drugs)):
        broad_id = row["broad_id"]
        smiles = row["smiles"]

        fp = smiles_to_morgan_fp(smiles)

        broad_ids.append(broad_id)
        fingerprints.append(fp)

    fingerprints = np.stack(fingerprints).astype(np.float32)
    broad_ids = np.array(broad_ids)

    np.savez_compressed(
        OUTPUT_PATH,
        broad_ids=broad_ids,
        fingerprints=fingerprints,
    )

    print(f"Saved fingerprint cache to: {OUTPUT_PATH}")
    print(f"Fingerprint matrix shape: {fingerprints.shape}")


if __name__ == "__main__":
    main()