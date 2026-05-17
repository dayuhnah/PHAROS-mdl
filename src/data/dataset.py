from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.featurize_smiles import smiles_to_morgan_fp


RESISTANCE_GENES = [
    "ABCB1", "ABCC1", "ABCG2",
    "BCL2", "BAX", "BAK1", "CASP3", "CASP8", "CASP9", "AKT1", "TP53",
    "BRCA1", "BRCA2", "RAD51", "ERCC1", "MGMT", "ATM", "ATR",
    "GSTP1", "GSTA1", "CYP3A4", "CYP2D6",
    "HIF1A", "VEGFA", "IL6", "CXCR4",
    "DNMT1", "HDAC1", "HDAC2",
]


def clean_gene_name(column_name: str) -> str:
    """
    Convert DepMap expression column names like 'TP53 (7157)' into 'TP53'.
    """
    return column_name.split(" (")[0]


class PharosDepMapDataset(Dataset):
    def __init__(
        self,
        response_path: str | Path = "data/processed/pharos_depmap_response_pairs.parquet",
        expression_path: str | Path = "data/processed/pharos_depmap_expression.parquet",
        max_rows: int | None = 100_000,
        fingerprint_bits: int = 2048,
    ):
        self.fingerprint_bits = fingerprint_bits

        print("Loading response pairs...")
        response = pd.read_parquet(response_path)

        print("Loading expression matrix...")
        expression = pd.read_parquet(expression_path)

        # Keep rows with SMILES and label
        response = response.dropna(subset=["smiles", "logfold_change", "depmap_id"]).copy()

        # Start small for sanity. 2.5M rows is huge.
        if max_rows is not None and len(response) > max_rows:
            response = response.sample(n=max_rows, random_state=42).reset_index(drop=True)

        # Clean expression gene names
        expression = expression.copy()
        expression.columns = [
            "depmap_id" if col == "depmap_id" else clean_gene_name(col)
            for col in expression.columns
        ]

        expression = expression.set_index("depmap_id")

        # Keep only response rows with expression
        response = response[response["depmap_id"].isin(expression.index)].reset_index(drop=True)

        self.response = response
        self.expression = expression

        self.gene_cols = list(expression.columns)

        self.available_resistance_genes = [
            gene for gene in RESISTANCE_GENES if gene in expression.columns
        ]

        print(f"Dataset rows: {len(self.response)}")
        print(f"Expression genes: {len(self.gene_cols)}")
        print(f"Available resistance genes: {len(self.available_resistance_genes)}")
        print(self.available_resistance_genes)

    def __len__(self):
        return len(self.response)

    def __getitem__(self, idx):
        row = self.response.iloc[idx]

        depmap_id = row["depmap_id"]
        smiles = row["smiles"]
        label = row["logfold_change"]

        drug_fp = smiles_to_morgan_fp(
            smiles,
            n_bits=self.fingerprint_bits,
        )

        cell_expr = self.expression.loc[depmap_id, self.gene_cols].values.astype(np.float32)

        resistance_expr = self.expression.loc[
            depmap_id,
            self.available_resistance_genes,
        ].values.astype(np.float32)

        return {
            "drug_fp": torch.tensor(drug_fp, dtype=torch.float32),
            "cell_expr": torch.tensor(cell_expr, dtype=torch.float32),
            "resistance_expr": torch.tensor(resistance_expr, dtype=torch.float32),
            "label": torch.tensor([label], dtype=torch.float32),
        }