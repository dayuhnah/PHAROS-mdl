from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from src.data.featurize_graph import smiles_to_graph


PROCESSED_DIR = Path("data/processed")
RESPONSE_PATH = PROCESSED_DIR / "pharos_depmap_response_pairs.parquet"
OUTPUT_PATH = PROCESSED_DIR / "pharos_drug_graphs.pt"


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

    graph_cache = {}
    failed = 0

    for _, row in tqdm(drugs.iterrows(), total=len(drugs)):
        broad_id = str(row["broad_id"])
        smiles = row["smiles"]

        graph = smiles_to_graph(smiles)

        if graph is None:
            failed += 1
            continue

        graph_cache[broad_id] = graph

    torch.save(graph_cache, OUTPUT_PATH)

    print(f"Saved graph cache to: {OUTPUT_PATH}")
    print(f"Graphs saved: {len(graph_cache)}")
    print(f"Failed SMILES: {failed}")

    if graph_cache:
        first_graph = next(iter(graph_cache.values()))
        print(f"Node feature dim: {first_graph.x.shape[1]}")
        print(f"Edge feature dim: {first_graph.edge_attr.shape[1] if first_graph.edge_attr.numel() > 0 else 6}")


if __name__ == "__main__":
    main()