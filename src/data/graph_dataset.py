from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.dataset import RESISTANCE_GENES, clean_gene_name


class PharosGraphDataset(Dataset):
    def __init__(
        self,
        response_path: str | Path = "data/processed/pharos_depmap_response_pairs.parquet",
        expression_path: str | Path = "data/processed/pharos_depmap_expression.parquet",
        graph_cache_path: str | Path = "data/processed/pharos_drug_graphs.pt",
        max_rows: int | None = 10_000,
    ):
        print("Loading graph cache...")
        self.graph_cache = torch.load(graph_cache_path, weights_only=False)
        print(f"Loaded graphs for {len(self.graph_cache)} drugs.")

        print("Loading response pairs...")
        response = pd.read_parquet(response_path)
        response = response.dropna(subset=["broad_id", "logfold_change", "depmap_id"]).copy()

        response["broad_id"] = response["broad_id"].astype(str)

        # Keep only rows where we have a molecular graph
        response = response[response["broad_id"].isin(self.graph_cache.keys())].copy()

        if max_rows is not None and len(response) > max_rows:
            response = response.sample(n=max_rows, random_state=42).reset_index(drop=True)

        print("Loading expression matrix...")
        expression = pd.read_parquet(expression_path)

        expression = expression.copy()
        expression.columns = [
            "depmap_id" if col == "depmap_id" else clean_gene_name(col)
            for col in expression.columns
        ]

        expression = expression.set_index("depmap_id")

        response = response[response["depmap_id"].isin(expression.index)].reset_index(drop=True)

        self.response = response
        self.expression = expression

        self.gene_cols = list(expression.columns)

        self.available_resistance_genes = [
            gene for gene in RESISTANCE_GENES
            if gene in expression.columns
        ]

        self.non_resistance_gene_cols = [
            gene for gene in self.gene_cols
            if gene not in self.available_resistance_genes
        ]

        print(f"Dataset rows: {len(self.response)}")
        print(f"Expression genes: {len(self.gene_cols)}")
        print(f"Non-resistance expression genes: {len(self.non_resistance_gene_cols)}")
        print(f"Resistance genes: {len(self.available_resistance_genes)}")

    def __len__(self):
        return len(self.response)

    def __getitem__(self, idx):
        row = self.response.iloc[idx]

        depmap_id = row["depmap_id"]
        broad_id = str(row["broad_id"])
        label = row["logfold_change"]

        graph = self.graph_cache[broad_id].clone()

        cell_expr = self.expression.loc[
            depmap_id,
            self.non_resistance_gene_cols,
        ].values.astype(np.float32)

        resistance_expr = self.expression.loc[
            depmap_id,
            self.available_resistance_genes,
        ].values.astype(np.float32)

        graph.cell_expr = torch.tensor(cell_expr, dtype=torch.float32)
        graph.resistance_expr = torch.tensor(resistance_expr, dtype=torch.float32)
        graph.y = torch.tensor([label], dtype=torch.float32)

        return graph