from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.combo_dataset import PharosComboDataset
from src.data.dataset import clean_gene_name


def _normalize_drug_name(value: object) -> str:
    """
    Normalize drug names only for robust lookup fallback.

    Exact PHAROS names are always tried first. The normalized lookup
    is used for capitalization/punctuation variants such as
    DASATINIB vs Dasatinib.
    """
    if value is None:
        return ""

    text = str(value).strip()

    if not text or text.lower() == "nan":
        return ""

    return re.sub(
        r"[^a-z0-9]+",
        "",
        text.lower(),
    )


class PharosComboScDrugActDataset(PharosComboDataset):
    """
    PHAROS-Combo dataset with scDrugAct-derived, drug-specific
    resistance expression features.

    The stable V1 PharosComboDataset remains unchanged. This V2 dataset
    adds, for each drug d and cancer cell c:

        r_(d,c) = x_c * m_d

    where:
      x_c = DepMap expression over the 2,399 usable scDrugAct genes
      m_d = binary scDrugAct resistance mask for that PHAROS drug

    Returned additions:
      - drug_a_scdrugact_expr: [G]
      - drug_b_scdrugact_expr: [G]
      - drug_a_scdrugact_mask: [G]
      - drug_b_scdrugact_mask: [G]
      - drug_a_scdrugact_available: [1]
      - drug_b_scdrugact_available: [1]
      - drug_a_scdrugact_gene_count: [1]
      - drug_b_scdrugact_gene_count: [1]

    The original 29-gene `resistance_expr` from PharosComboDataset is
    retained unchanged as the always-available core resistance prior.
    """

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
        scdrugact_mask_path=(
            "data/processed/combination/scdrugact/"
            "pharos_scdrugact_resistance_masks.npz"
        ),
        target="zip_score",
        max_rows=None,
        seed=42,
    ):

        # ----------------------------------------------------
        # Load the existing, stable PHAROS-Combo dataset first.
        # ----------------------------------------------------
        super().__init__(
            combo_path=combo_path,
            expression_path=expression_path,
            fingerprint_cache_path=fingerprint_cache_path,
            graph_cache_path=graph_cache_path,
            ppi_graph_path=ppi_graph_path,
            target=target,
            max_rows=max_rows,
            seed=seed,
        )

        print(
            "\nLoading scDrugAct resistance features..."
        )

        scdrugact_mask_path = Path(
            scdrugact_mask_path
        )

        if not scdrugact_mask_path.exists():
            raise FileNotFoundError(
                "Missing scDrugAct resistance mask cache:\n"
                f"{scdrugact_mask_path}"
            )

        # ====================================================
        # Load scDrugAct mask matrix
        # ====================================================

        sc_cache = np.load(
            scdrugact_mask_path,
            allow_pickle=False,
        )

        required_keys = {
            "mask",
            "drug_names",
            "genes",
        }

        missing_keys = (
            required_keys
            - set(sc_cache.files)
        )

        if missing_keys:
            raise ValueError(
                "scDrugAct mask cache is missing keys: "
                + ", ".join(
                    sorted(missing_keys)
                )
            )

        mask_np = (
            sc_cache["mask"]
            .astype(
                np.float32,
                copy=False,
            )
        )

        sc_drug_names = (
            sc_cache["drug_names"]
            .astype(str)
        )

        sc_genes = [
            clean_gene_name(
                str(gene)
            )
            for gene
            in sc_cache["genes"].astype(str)
        ]

        if (
            mask_np.ndim != 2
            or mask_np.shape[0]
            != len(sc_drug_names)
            or mask_np.shape[1]
            != len(sc_genes)
        ):
            raise ValueError(
                "Invalid scDrugAct cache dimensions: "
                f"mask={mask_np.shape}, "
                f"drugs={len(sc_drug_names)}, "
                f"genes={len(sc_genes)}"
            )

        if len(set(sc_genes)) != len(sc_genes):
            raise ValueError(
                "Duplicate cleaned gene symbols found "
                "in scDrugAct mask cache."
            )

        self.scdrugact_gene_cols = list(
            sc_genes
        )

        self.scdrugact_gene_dim = len(
            self.scdrugact_gene_cols
        )

        # Exact-name mask lookup.
        self.scdrugact_mask_cache = {
            str(drug):
                torch.from_numpy(
                    mask_np[i]
                )
            for i, drug
            in enumerate(sc_drug_names)
        }

        # ----------------------------------------------------
        # Robust normalized-name fallback.
        #
        # If two PHAROS rows collapse to the same normalized name,
        # their masks MUST be identical. This specifically handles
        # capitalization duplicates such as DASATINIB/Dasatinib and
        # TRICIRIBINE/Triciribine without hiding conflicting data.
        # ----------------------------------------------------

        normalized_mask_cache = {}

        for i, drug in enumerate(
            sc_drug_names
        ):
            norm = _normalize_drug_name(
                drug
            )

            if not norm:
                continue

            current = torch.from_numpy(
                mask_np[i]
            )

            if norm in normalized_mask_cache:
                previous = (
                    normalized_mask_cache[
                        norm
                    ]
                )

                if not torch.equal(
                    previous,
                    current,
                ):
                    raise ValueError(
                        "Conflicting scDrugAct masks for "
                        f"normalized drug name '{norm}'."
                    )

            else:
                normalized_mask_cache[
                    norm
                ] = current

        self.scdrugact_normalized_mask_cache = (
            normalized_mask_cache
        )

        self.scdrugact_zero_mask = torch.zeros(
            self.scdrugact_gene_dim,
            dtype=torch.float32,
        )

        # ====================================================
        # Align scDrugAct genes to actual DepMap expression
        # ====================================================

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

        # Case-insensitive scDrugAct -> DepMap gene alignment.
        #
        # Example:
        # scDrugAct: C11ORF58
        # DepMap:    C11orf58
        #
        # These represent the same gene.

        depmap_gene_lookup = {}
        ambiguous_gene_keys = set()

        for col in expression.columns:
            key = str(col).strip().upper()

            if key in depmap_gene_lookup:
                if depmap_gene_lookup[key] != col:
                    ambiguous_gene_keys.add(key)
            else:
                depmap_gene_lookup[key] = col

        if ambiguous_gene_keys:
            raise ValueError(
                "Ambiguous case-insensitive DepMap gene symbols: "
                f"{sorted(ambiguous_gene_keys)[:20]}"
            )

        self.scdrugact_expression_cols = []
        missing_genes = []

        for gene in self.scdrugact_gene_cols:
            key = str(gene).strip().upper()

            resolved_col = depmap_gene_lookup.get(
                key
            )

            if resolved_col is None:
                missing_genes.append(
                    gene
                )
            else:
                self.scdrugact_expression_cols.append(
                    resolved_col
                )

        if missing_genes:
            raise ValueError(
                f"{len(missing_genes)} scDrugAct genes are genuinely "
                "missing from DepMap after case-insensitive alignment. "
                f"First missing genes: {missing_genes[:20]}"
            )

        if (
            len(self.scdrugact_expression_cols)
            != self.scdrugact_gene_dim
        ):
            raise RuntimeError(
                "scDrugAct-to-DepMap gene alignment length mismatch."
            )

        case_adjusted = sum(
            1
            for gene, col in zip(
                self.scdrugact_gene_cols,
                self.scdrugact_expression_cols,
            )
            if str(gene) != str(col)
        )

        print(
            f"Resolved {case_adjusted:,} scDrugAct gene symbols "
            "using DepMap case/name adjustment."
        )

        usable_cells = set(
            expression.index.astype(str)
        )

        missing_cells = sorted(
            set(
                self.depmap_ids
            )
            - usable_cells
        )

        if missing_cells:
            raise ValueError(
                f"{len(missing_cells)} PHAROS cells "
                "missing from expression data."
            )

        # Cache the SAME 2,399-gene expression vector once per cell.
        # We do not precompute every drug-cell masked vector because
        # doing so would waste substantial memory.
        self.scdrugact_cell_expr_cache = {}

        for depmap_id in sorted(
            set(
                self.depmap_ids
            )
        ):
            self.scdrugact_cell_expr_cache[
                depmap_id
            ] = torch.from_numpy(
                expression.loc[
                    depmap_id,
                    self.scdrugact_expression_cols,
                ]
                .to_numpy(
                    dtype=np.float32
                )
            )

        del expression

        # ====================================================
        # Resolve masks for all drugs actually used in this dataset
        # ====================================================

        used_drugs = sorted(
            self.get_unique_drugs()
        )

        self.scdrugact_resolved_mask_cache = {}
        self.scdrugact_available_cache = {}
        self.scdrugact_gene_count_cache = {}

        exact_resolved = 0
        normalized_resolved = 0
        unresolved = 0

        for drug in used_drugs:

            if drug in self.scdrugact_mask_cache:
                mask = (
                    self.scdrugact_mask_cache[
                        drug
                    ]
                )
                exact_resolved += 1

            else:
                norm = _normalize_drug_name(
                    drug
                )

                mask = (
                    self.scdrugact_normalized_mask_cache
                    .get(
                        norm
                    )
                )

                if mask is not None:
                    normalized_resolved += 1

                else:
                    mask = (
                        self.scdrugact_zero_mask
                    )
                    unresolved += 1

            # `available` means the database actually supplies at least
            # one usable resistance-associated gene for this drug.
            gene_count = int(
                torch.count_nonzero(
                    mask
                ).item()
            )

            self.scdrugact_resolved_mask_cache[
                drug
            ] = mask

            self.scdrugact_available_cache[
                drug
            ] = (
                gene_count > 0
            )

            self.scdrugact_gene_count_cache[
                drug
            ] = gene_count

        # ====================================================
        # Dataset-level coverage diagnostics
        # ====================================================

        a_available = np.asarray(
            [
                self.scdrugact_available_cache[
                    drug
                ]
                for drug
                in self.drug_a_names
            ],
            dtype=bool,
        )

        b_available = np.asarray(
            [
                self.scdrugact_available_cache[
                    drug
                ]
                for drug
                in self.drug_b_names
            ],
            dtype=bool,
        )

        at_least_one = (
            a_available
            | b_available
        )

        both = (
            a_available
            & b_available
        )

        # Count unique normalized matched compounds instead of
        # capitalization-specific names.
        matched_used_norms = {
            _normalize_drug_name(
                drug
            )
            for drug
            in used_drugs
            if self.scdrugact_available_cache[
                drug
            ]
        }

        print(
            "\n========================================"
        )
        print(
            "SCDRUGACT RESISTANCE V2"
        )
        print(
            "========================================"
        )

        print(
            f"scDrugAct genes: "
            f"{self.scdrugact_gene_dim:,}"
        )

        print(
            f"Cached cell resistance-expression "
            f"vectors: "
            f"{len(self.scdrugact_cell_expr_cache):,}"
        )

        print(
            f"Used drugs resolved by exact name: "
            f"{exact_resolved:,}"
        )

        print(
            f"Used drugs resolved by normalized fallback: "
            f"{normalized_resolved:,}"
        )

        print(
            f"Used drugs with no scDrugAct mask: "
            f"{unresolved:,}"
        )

        print(
            f"Unique normalized used drugs with "
            f"non-zero scDrugAct evidence: "
            f"{len(matched_used_norms):,}"
        )

        print(
            f"Rows with Drug A evidence: "
            f"{a_available.sum():,} "
            f"({100 * a_available.mean():.2f}%)"
        )

        print(
            f"Rows with Drug B evidence: "
            f"{b_available.sum():,} "
            f"({100 * b_available.mean():.2f}%)"
        )

        print(
            f"Rows with >=1 matched drug: "
            f"{at_least_one.sum():,} "
            f"({100 * at_least_one.mean():.2f}%)"
        )

        print(
            f"Rows with both drugs matched: "
            f"{both.sum():,} "
            f"({100 * both.mean():.2f}%)"
        )

        print(
            f"Rows with neither drug matched: "
            f"{(~at_least_one).sum():,} "
            f"({100 * (~at_least_one).mean():.2f}%)"
        )

        print(
            "Core 29-gene resistance branch: retained"
        )

    # ========================================================
    # Mask resolution helper
    # ========================================================

    def _get_scdrugact_mask(
        self,
        drug: str,
    ) -> torch.Tensor:

        return (
            self.scdrugact_resolved_mask_cache[
                drug
            ]
        )

    # ========================================================
    # Dataset API
    # ========================================================

    def __getitem__(
        self,
        idx,
    ):

        sample = super().__getitem__(
            idx
        )

        drug_a = self.drug_a_names[
            idx
        ]

        drug_b = self.drug_b_names[
            idx
        ]

        depmap_id = self.depmap_ids[
            idx
        ]

        cell_sc_expr = (
            self.scdrugact_cell_expr_cache[
                depmap_id
            ]
        )

        mask_a = self._get_scdrugact_mask(
            drug_a
        )

        mask_b = self._get_scdrugact_mask(
            drug_b
        )

        # Real DepMap expression, selected by the real scDrugAct
        # drug-specific resistance associations.
        drug_a_sc_expr = (
            cell_sc_expr
            * mask_a
        )

        drug_b_sc_expr = (
            cell_sc_expr
            * mask_b
        )

        sample.update(
            {
                "drug_a_scdrugact_expr":
                    drug_a_sc_expr,

                "drug_b_scdrugact_expr":
                    drug_b_sc_expr,

                "drug_a_scdrugact_mask":
                    mask_a,

                "drug_b_scdrugact_mask":
                    mask_b,

                "drug_a_scdrugact_available":
                    torch.tensor(
                        [
                            float(
                                self.scdrugact_available_cache[
                                    drug_a
                                ]
                            )
                        ],
                        dtype=torch.float32,
                    ),

                "drug_b_scdrugact_available":
                    torch.tensor(
                        [
                            float(
                                self.scdrugact_available_cache[
                                    drug_b
                                ]
                            )
                        ],
                        dtype=torch.float32,
                    ),

                "drug_a_scdrugact_gene_count":
                    torch.tensor(
                        [
                            self.scdrugact_gene_count_cache[
                                drug_a
                            ]
                        ],
                        dtype=torch.float32,
                    ),

                "drug_b_scdrugact_gene_count":
                    torch.tensor(
                        [
                            self.scdrugact_gene_count_cache[
                                drug_b
                            ]
                        ],
                        dtype=torch.float32,
                    ),
            }
        )

        return sample
