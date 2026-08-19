from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

from src.data.dataset import clean_gene_name


RESISTANCE_GENES = [
    "ABCB1",
    "ABCC1",
    "ABCG2",
    "BCL2",
    "BAX",
    "BAK1",
    "CASP3",
    "CASP8",
    "CASP9",
    "AKT1",
    "TP53",
    "BRCA1",
    "BRCA2",
    "RAD51",
    "ERCC1",
    "MGMT",
    "ATM",
    "ATR",
    "GSTP1",
    "GSTA1",
    "CYP3A4",
    "CYP2D6",
    "HIF1A",
    "VEGFA",
    "IL6",
    "CXCR4",
    "DNMT1",
    "HDAC1",
    "HDAC2",
]


class PharosComboDataset(Dataset):

    def __init__(
        self,
        combo_path=(
            "data/processed/combination/"
            "pharos_combo_local_smiles.parquet"
        ),
        expression_path=(
            "data/processed/"
            "pharos_depmap_expression.parquet"
        ),
        fingerprint_cache_path=(
            "data/processed/combination/"
            "pharos_combo_drug_fingerprints.npz"
        ),
        graph_cache_path=(
            "data/processed/combination/"
            "pharos_combo_drug_graphs.pt"
        ),
        ppi_graph_path=(
            "data/processed/"
            "pharos_ppi_graph.pt"
        ),
        target="zip_score",
        max_rows=None,
        seed=42,
    ):

        print("Loading PHAROS-Combo dataset...")

        # ====================================================
        # Combination data
        # ====================================================

        self.combo = pd.read_parquet(
            combo_path
        ).copy()

        if target not in self.combo.columns:
            raise ValueError(
                f"Target '{target}' not found."
            )

        self.target = target

        # ====================================================
        # Expression
        # ====================================================

        print("Loading expression matrix...")

        expression = pd.read_parquet(
            expression_path
        ).copy()

        expression.columns = [
            (
                "depmap_id"
                if col == "depmap_id"
                else clean_gene_name(col)
            )
            for col in expression.columns
        ]

        expression = expression.set_index(
            "depmap_id"
        )

        # ====================================================
        # Resistance genes
        # ====================================================

        self.available_resistance_genes = [
            gene
            for gene in RESISTANCE_GENES
            if gene in expression.columns
        ]

        if (
            len(self.available_resistance_genes)
            != len(RESISTANCE_GENES)
        ):

            missing = sorted(
                set(RESISTANCE_GENES)
                -
                set(
                    self.available_resistance_genes
                )
            )

            raise ValueError(
                f"Missing resistance genes: {missing}"
            )

        # ====================================================
        # General expression genes
        # ====================================================

        self.non_resistance_gene_cols = [
            gene
            for gene in expression.columns
            if gene
            not in self.available_resistance_genes
        ]

        # ====================================================
        # PPI graph
        # ====================================================

        self.ppi_graph = torch.load(
            ppi_graph_path,
            map_location="cpu",
            weights_only=False,
        )

        self.ppi_gene_cols = [
            clean_gene_name(gene)
            for gene
            in self.ppi_graph["genes"]
        ]

        missing_ppi = [
            gene
            for gene in self.ppi_gene_cols
            if gene not in expression.columns
        ]

        if missing_ppi:
            raise ValueError(
                f"{len(missing_ppi)} PPI genes "
                "missing from expression data."
            )

        # ====================================================
        # Fingerprints
        # ====================================================

        print("Loading combo fingerprints...")

        fp_cache = np.load(
            fingerprint_cache_path,
            allow_pickle=True,
        )

        drug_names = (
            fp_cache["drug_names"]
            .astype(str)
        )

        fingerprints = (
            fp_cache["fingerprints"]
            .astype(np.float32)
        )

        # Convert ONCE to tensors.
        self.fingerprint_cache = {
            str(name):
                torch.from_numpy(
                    fingerprints[i]
                )
            for i, name
            in enumerate(drug_names)
        }

        # ====================================================
        # Graphs
        # ====================================================

        print("Loading combo molecular graphs...")

        self.graph_cache = torch.load(
            graph_cache_path,
            map_location="cpu",
            weights_only=False,
        )

        # ====================================================
        # Filter usable rows
        # ====================================================

        before = len(self.combo)

        usable_drugs = (
            set(self.fingerprint_cache.keys())
            &
            set(self.graph_cache.keys())
        )

        usable_cells = set(
            expression.index.astype(str)
        )

        self.combo = self.combo[
            self.combo["drug_a"]
            .astype(str)
            .isin(usable_drugs)
            &
            self.combo["drug_b"]
            .astype(str)
            .isin(usable_drugs)
            &
            self.combo["depmap_id"]
            .astype(str)
            .isin(usable_cells)
        ].copy()

        self.combo = self.combo[
            np.isfinite(
                self.combo[self.target]
            )
        ].copy()

        removed = (
            before
            -
            len(self.combo)
        )

        # ====================================================
        # Optional subset
        # ====================================================

        if (
            max_rows is not None
            and len(self.combo) > max_rows
        ):

            self.combo = (
                self.combo
                .sample(
                    n=max_rows,
                    random_state=seed,
                )
                .reset_index(drop=True)
            )

        else:

            self.combo = (
                self.combo
                .reset_index(drop=True)
            )

        # ====================================================
        # CACHE CELL FEATURES ONCE
        # ====================================================

        print(
            "Caching cell tensors..."
        )

        unique_cells = (
            self.combo["depmap_id"]
            .astype(str)
            .unique()
        )

        self.cell_expr_cache = {}
        self.ppi_expression_cache = {}
        self.resistance_cache = {}

        for depmap_id in unique_cells:

            self.cell_expr_cache[
                depmap_id
            ] = torch.from_numpy(
                expression.loc[
                    depmap_id,
                    self.non_resistance_gene_cols,
                ]
                .to_numpy(
                    dtype=np.float32
                )
            )

            self.ppi_expression_cache[
                depmap_id
            ] = torch.from_numpy(
                expression.loc[
                    depmap_id,
                    self.ppi_gene_cols,
                ]
                .to_numpy(
                    dtype=np.float32
                )
            )

            self.resistance_cache[
                depmap_id
            ] = torch.from_numpy(
                expression.loc[
                    depmap_id,
                    self.available_resistance_genes,
                ]
                .to_numpy(
                    dtype=np.float32
                )
            )

        # Expression dataframe no longer needed
        # during __getitem__.
        del expression

        # ====================================================
        # CACHE ROW METADATA
        #
        # Avoid pandas .iloc during every __getitem__.
        # ====================================================

        self.drug_a_names = (
            self.combo["drug_a"]
            .astype(str)
            .tolist()
        )

        self.drug_b_names = (
            self.combo["drug_b"]
            .astype(str)
            .tolist()
        )

        self.depmap_ids = (
            self.combo["depmap_id"]
            .astype(str)
            .tolist()
        )

        self.labels = (
            self.combo[self.target]
            .to_numpy(dtype=np.float32)
        )

        # ====================================================
        # Diagnostics
        # ====================================================

        print(
            "\n========================================"
        )

        print(
            "PHAROS-COMBO DATASET"
        )

        print(
            "========================================"
        )

        print(
            f"Rows: {len(self.combo):,}"
        )

        print(
            f"Removed unusable rows: {removed:,}"
        )

        print(
            f"Unique drug pairs: "
            f"{len(self.get_unique_pairs()):,}"
        )

        print(
            f"Unique drugs: "
            f"{len(self.get_unique_drugs()):,}"
        )

        print(
            f"Unique cells: "
            f"{len(unique_cells):,}"
        )

        print(
            f"Cached cells: "
            f"{len(self.cell_expr_cache):,}"
        )

        print(
            f"General expression genes: "
            f"{len(self.non_resistance_gene_cols):,}"
        )

        print(
            f"PPI genes: "
            f"{len(self.ppi_gene_cols):,}"
        )

        print(
            f"Resistance genes: "
            f"{len(self.available_resistance_genes):,}"
        )

        print(
            f"Target: {self.target}"
        )

    # ========================================================
    # Dataset API
    # ========================================================

    def __len__(self):

        return len(
            self.labels
        )

    def __getitem__(
        self,
        idx,
    ):

        drug_a = (
            self.drug_a_names[idx]
        )

        drug_b = (
            self.drug_b_names[idx]
        )

        depmap_id = (
            self.depmap_ids[idx]
        )

        label = (
            self.labels[idx]
        )

        return {

            "drug_a_graph":
                self.graph_cache[
                    drug_a
                ],

            "drug_a_fp":
                self.fingerprint_cache[
                    drug_a
                ],

            "drug_b_graph":
                self.graph_cache[
                    drug_b
                ],

            "drug_b_fp":
                self.fingerprint_cache[
                    drug_b
                ],

            "cell_expr":
                self.cell_expr_cache[
                    depmap_id
                ],

            "ppi_expression":
                self.ppi_expression_cache[
                    depmap_id
                ],

            "resistance_expr":
                self.resistance_cache[
                    depmap_id
                ],

            "label":
                torch.tensor(
                    [label],
                    dtype=torch.float32,
                ),

            "drug_a":
                drug_a,

            "drug_b":
                drug_b,

            "depmap_id":
                depmap_id,
        }

    # ========================================================
    # Metadata
    # ========================================================

    def get_metadata(self):

        return self.combo.copy()

    def get_unique_pairs(self):

        return (
            self.combo[
                [
                    "drug_a",
                    "drug_b",
                ]
            ]
            .drop_duplicates()
        )

    def get_unique_drugs(self):

        return (
            set(
                self.combo[
                    "drug_a"
                ].astype(str)
            )
            |
            set(
                self.combo[
                    "drug_b"
                ].astype(str)
            )
        )