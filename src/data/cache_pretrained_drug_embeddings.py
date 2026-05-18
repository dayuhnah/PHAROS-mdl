from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.data.featurize_smiles import clean_smiles


PROCESSED_DIR = Path("data/processed")
RESPONSE_PATH = PROCESSED_DIR / "pharos_depmap_response_pairs.parquet"
OUTPUT_PATH = PROCESSED_DIR / "pharos_drug_embeddings_chemberta.npz"

MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def main():
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")

    print(f"Using device: {device}")
    print(f"Loading pretrained model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

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
    embeddings = []

    batch_size = 64

    for start in tqdm(range(0, len(drugs), batch_size)):
        batch = drugs.iloc[start:start + batch_size]

        batch_ids = batch["broad_id"].astype(str).tolist()
        batch_smiles = [clean_smiles(s) for s in batch["smiles"].tolist()]

        encoded = tokenizer(
            batch_smiles,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )

        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            output = model(**encoded)
            pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])

        broad_ids.extend(batch_ids)
        embeddings.append(pooled.cpu().numpy().astype(np.float32))

    embeddings = np.concatenate(embeddings, axis=0)
    broad_ids = np.array(broad_ids)

    np.savez_compressed(
        OUTPUT_PATH,
        broad_ids=broad_ids,
        embeddings=embeddings,
        model_name=MODEL_NAME,
    )

    print(f"Saved pretrained drug embeddings to: {OUTPUT_PATH}")
    print(f"Embedding matrix shape: {embeddings.shape}")


if __name__ == "__main__":
    main()