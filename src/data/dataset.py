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
    Convert DepMap expression column names like
    'TP53 (7157)' into 'TP53'.
    """
    return column_name.split(" (")[0]


class PharosDepMapDataset(Dataset):

    def __init__(
        self,
        response_path: str | Path = (
            "data/processed/pharos_depmap_response_pairs.parquet"
        ),
        expression_path: str | Path = (
            "data/processed/pharos_depmap_expression.parquet"
        ),
        fingerprint_cache_path: str | Path = (
            "data/processed/pharos_drug_fingerprints.npz"
        ),
        drug_graph_cache_path: str | Path = (
            "data/processed/pharos_drug_graphs.pt"
        ),
        max_rows: int | None = 100_000,
        fingerprint_bits: int = 2048,
        cell_embedding_cache_path: str | Path | None = None,
        use_cell_embeddings: bool = False,
        drug_embedding_cache_path: str | Path | None = None,
        use_drug_embeddings: bool = False,
        use_drug_graphs: bool = False,
    ):
        self.fingerprint_bits = fingerprint_bits

        # ---------------------------------
        # Morgan fingerprint cache
        # ---------------------------------
        self.fingerprint_cache = self._load_fingerprint_cache(
            fingerprint_cache_path
        )

        # ---------------------------------
        # Optional pretrained cell embeddings
        # ---------------------------------
        self.use_cell_embeddings = use_cell_embeddings
        self.cell_embedding_cache = self._load_cell_embedding_cache(
            cell_embedding_cache_path
        )

        # ---------------------------------
        # Optional pretrained drug embeddings
        # ---------------------------------
        self.use_drug_embeddings = use_drug_embeddings
        self.drug_embedding_cache = self._load_drug_embedding_cache(
            drug_embedding_cache_path
        )

        # ---------------------------------
        # Molecular graph cache
        # ---------------------------------
        self.use_drug_graphs = use_drug_graphs

        if self.use_drug_graphs:
            self.drug_graph_cache = self._load_drug_graph_cache(
                drug_graph_cache_path
            )
        else:
            self.drug_graph_cache = {}

        # =================================
        # Load response data
        # =================================
        print("Loading response pairs...")
        response = pd.read_parquet(response_path)

        print("Loading expression matrix...")
        expression = pd.read_parquet(expression_path)

        # ---------------------------------
        # Require usable response rows
        # ---------------------------------
        response = response.dropna(
            subset=[
                "smiles",
                "logfold_change",
                "depmap_id",
                "broad_id",
            ]
        ).copy()

        # ---------------------------------
        # Fixed pilot sample
        # ---------------------------------
        if max_rows is not None and len(response) > max_rows:
            response = response.sample(
                n=max_rows,
                random_state=42,
            ).reset_index(drop=True)

        # ---------------------------------
        # Clean expression gene names
        # ---------------------------------
        expression = expression.copy()
        expression.columns = [
            (
                "depmap_id"
                if col == "depmap_id"
                else clean_gene_name(col)
            )
            for col in expression.columns
        ]
        expression = expression.set_index("depmap_id")

        # ---------------------------------
        # Keep cells with expression
        # ---------------------------------
        response = response[
            response["depmap_id"].isin(expression.index)
        ].reset_index(drop=True)

        # ---------------------------------
        # If using molecular graphs:
        # keep only graph-covered drugs
        # ---------------------------------
        if self.use_drug_graphs:
            before = len(response)
            graph_drug_ids = set(self.drug_graph_cache.keys())

            response = response[
                response["broad_id"]
                .astype(str)
                .isin(graph_drug_ids)
            ].reset_index(drop=True)

            removed = before - len(response)
            print(
                f"Removed {removed} response rows "
                f"without molecular graphs."
            )

        self.response = response
        self.expression = expression

        # =================================
        # Gene partitions
        # =================================
        self.gene_cols = list(expression.columns)

        self.available_resistance_genes = [
            gene
            for gene in RESISTANCE_GENES
            if gene in expression.columns
        ]

        self.non_resistance_gene_cols = [
            gene
            for gene in self.gene_cols
            if gene not in self.available_resistance_genes
        ]

        # =================================
        # PPI gene alignment
        # =================================
        ppi_graph_path = Path(
            "data/processed/pharos_ppi_graph.pt"
        )

        if not ppi_graph_path.exists():
            raise FileNotFoundError(
                f"PPI graph not found: {ppi_graph_path}"
            )

        ppi_graph = torch.load(
            ppi_graph_path,
            map_location="cpu",
            weights_only=False,
        )

        if "genes" not in ppi_graph:
            raise KeyError(
                "PPI graph file does not contain a 'genes' field."
            )

        self.ppi_gene_cols = list(ppi_graph["genes"])

        # ---------------------------------
        # Validate PPI genes
        # ---------------------------------
        missing_ppi_genes = [
            gene
            for gene in self.ppi_gene_cols
            if gene not in expression.columns
        ]

        if missing_ppi_genes:
            raise ValueError(
                "PPI graph contains genes that are missing from "
                "the expression matrix: "
                f"{missing_ppi_genes[:20]}"
            )

        # PPI graph should contain only non-resistance genes.
        ppi_resistance_overlap = [
            gene
            for gene in self.ppi_gene_cols
            if gene in self.available_resistance_genes
        ]

        if ppi_resistance_overlap:
            raise ValueError(
                "PPI genes unexpectedly overlap with resistance genes: "
                f"{ppi_resistance_overlap}"
            )

        # Optional sanity check: duplicate gene names would make node-to-
        # expression alignment ambiguous.
        if len(set(self.ppi_gene_cols)) != len(self.ppi_gene_cols):
            raise ValueError(
                "PPI graph contains duplicate gene names; exact node-to-gene "
                "alignment cannot be guaranteed."
            )

        print(f"Dataset rows: {len(self.response)}")
        print(f"Expression genes: {len(self.gene_cols)}")
        print(
            "Available resistance genes: "
            f"{len(self.available_resistance_genes)}"
        )
        print(self.available_resistance_genes)
        print(
            "Non-resistance expression genes: "
            f"{len(self.non_resistance_gene_cols)}"
        )
        print(
            "PPI-aligned expression genes: "
            f"{len(self.ppi_gene_cols)}"
        )

        if self.use_drug_graphs:
            print(
                f"Loaded molecular graph mode with "
                f"{len(self.drug_graph_cache)} cached drugs."
            )

    # =====================================
    # Cache loading
    # =====================================

    def _load_drug_graph_cache(
        self,
        drug_graph_cache_path: str | Path,
    ) -> dict:
        """Load cached PyTorch Geometric molecular graphs."""
        path = Path(drug_graph_cache_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Drug graph cache not found: {path}"
            )

        cache = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        if not isinstance(cache, dict):
            raise TypeError(
                "Expected molecular graph cache to be a dictionary."
            )

        cache = {
            str(broad_id): graph
            for broad_id, graph in cache.items()
        }

        print(
            f"Loaded cached molecular graphs for {len(cache)} drugs."
        )
        return cache

    def _load_drug_embedding_cache(
        self,
        drug_embedding_cache_path: str | Path | None,
    ) -> dict[str, np.ndarray]:
        if drug_embedding_cache_path is None:
            return {}

        drug_embedding_cache_path = Path(
            drug_embedding_cache_path
        )

        if not drug_embedding_cache_path.exists():
            print(
                "Drug embedding cache not found. "
                "Falling back to Morgan fingerprints."
            )
            return {}

        cache = np.load(
            drug_embedding_cache_path,
            allow_pickle=True,
        )

        broad_ids = cache["broad_ids"]
        embeddings = cache["embeddings"]

        embedding_map = {
            str(broad_id): embeddings[i].astype(np.float32)
            for i, broad_id in enumerate(broad_ids)
        }

        print(
            f"Loaded pretrained drug embeddings "
            f"for {len(embedding_map)} drugs."
        )
        return embedding_map

    def _load_fingerprint_cache(
        self,
        fingerprint_cache_path: str | Path,
    ) -> dict[str, np.ndarray]:
        fingerprint_cache_path = Path(
            fingerprint_cache_path
        )

        if not fingerprint_cache_path.exists():
            print(
                "Fingerprint cache not found. "
                "Falling back to on-the-fly RDKit featurization."
            )
            return {}

        cache = np.load(
            fingerprint_cache_path,
            allow_pickle=True,
        )

        broad_ids = cache["broad_ids"]
        fingerprints = cache["fingerprints"]

        fingerprint_map = {
            str(broad_id): fingerprints[i].astype(np.float32)
            for i, broad_id in enumerate(broad_ids)
        }

        print(
            f"Loaded cached fingerprints for "
            f"{len(fingerprint_map)} drugs."
        )
        return fingerprint_map

    def _load_cell_embedding_cache(
        self,
        cell_embedding_cache_path: str | Path | None,
    ) -> dict[str, np.ndarray]:
        if cell_embedding_cache_path is None:
            return {}

        cell_embedding_cache_path = Path(
            cell_embedding_cache_path
        )

        if not cell_embedding_cache_path.exists():
            print(
                "Cell embedding cache not found. "
                "Falling back to raw expression."
            )
            return {}

        cache = np.load(
            cell_embedding_cache_path,
            allow_pickle=True,
        )

        depmap_ids = cache["depmap_ids"]
        embeddings = cache["embeddings"]

        embedding_map = {
            str(depmap_id): embeddings[i].astype(np.float32)
            for i, depmap_id in enumerate(depmap_ids)
        }

        print(
            f"Loaded cached cell embeddings "
            f"for {len(embedding_map)} cell lines."
        )
        return embedding_map

    # =====================================
    # Dataset
    # =====================================

    def __len__(self):
        return len(self.response)

    def __getitem__(self, idx):
        row = self.response.iloc[idx]

        depmap_id = str(row["depmap_id"])
        broad_id = str(row["broad_id"])
        smiles = row["smiles"]
        label = row["logfold_change"]

        # ---------------------------------
        # Morgan fingerprint
        # ---------------------------------
        if broad_id in self.fingerprint_cache:
            drug_fp = self.fingerprint_cache[broad_id]
        else:
            drug_fp = smiles_to_morgan_fp(
                smiles,
                n_bits=self.fingerprint_bits,
            )

        # ---------------------------------
        # Optional pretrained drug embedding
        # ---------------------------------
        if (
            self.use_drug_embeddings
            and broad_id in self.drug_embedding_cache
        ):
            drug_embedding = self.drug_embedding_cache[broad_id]
        else:
            drug_embedding = drug_fp

        # ---------------------------------
        # General expression
        # ---------------------------------
        cell_expr = (
            self.expression.loc[
                depmap_id,
                self.non_resistance_gene_cols,
            ]
            .values
            .astype(np.float32)
        )

        # ---------------------------------
        # PPI-aligned expression
        # IMPORTANT: self.ppi_gene_cols preserves the exact
        # node order stored in pharos_ppi_graph.pt.
        # ---------------------------------
        ppi_expression = (
            self.expression.loc[
                depmap_id,
                self.ppi_gene_cols,
            ]
            .values
            .astype(np.float32)
        )

        # ---------------------------------
        # Optional cell embedding
        # ---------------------------------
        if (
            self.use_cell_embeddings
            and depmap_id in self.cell_embedding_cache
        ):
            cell_embedding = self.cell_embedding_cache[depmap_id]
        else:
            cell_embedding = cell_expr

        # ---------------------------------
        # Resistance expression
        # ---------------------------------
        resistance_expr = (
            self.expression.loc[
                depmap_id,
                self.available_resistance_genes,
            ]
            .values
            .astype(np.float32)
        )

        # ---------------------------------
        # Output
        # ---------------------------------
        output = {
            "drug_fp": torch.tensor(
                drug_fp,
                dtype=torch.float32,
            ),
            "cell_expr": torch.tensor(
                cell_expr,
                dtype=torch.float32,
            ),
            "ppi_expression": torch.tensor(
                ppi_expression,
                dtype=torch.float32,
            ),
            "resistance_expr": torch.tensor(
                resistance_expr,
                dtype=torch.float32,
            ),
            "label": torch.tensor(
                [label],
                dtype=torch.float32,
            ),
            "cell_embedding": torch.tensor(
                cell_embedding,
                dtype=torch.float32,
            ),
            "drug_embedding": torch.tensor(
                drug_embedding,
                dtype=torch.float32,
            ),
            "broad_id": broad_id,
            "depmap_id": depmap_id,
        }

        # ---------------------------------
        # Molecular graph
        # ---------------------------------
        if self.use_drug_graphs:
            output["drug_graph"] = (
                self.drug_graph_cache[broad_id].clone()
            )

        return output

    # =====================================
    # Metadata
    # =====================================

    def get_metadata(self) -> pd.DataFrame:
        cols = [
            "depmap_id",
            "broad_id",
            "name",
            "dose",
            "logfold_change",
            "moa",
            "target",
            "smiles",
        ]

        cols = [
            col
            for col in cols
            if col in self.response.columns
        ]

        return self.response[cols].copy()