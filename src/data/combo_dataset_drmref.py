from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.combo_dataset import PharosComboDataset
from src.data.dataset import clean_gene_name


def _normalize_drug_name(value: object) -> str:
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


class PharosComboDRMrefDataset(PharosComboDataset):
    """
    PHAROS-Combo dataset with DRMref real resistant-vs-sensitive
    cancer-cell expression signatures.

    For each drug d and cell c:

        r_DRMref(d,c) = x_c * w_d

    x_c:
        DepMap expression over DRMref-compatible genes.

    w_d:
        Signed DRMref resistant-vs-sensitive log2FC vector.

    The existing 29-gene `resistance_expr` is retained unchanged.
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
        drmref_cache_path=(
            "data/processed/combination/drmref/"
            "pharos_drmref_resistance_weights.npz"
        ),
        target="zip_score",
        max_rows=None,
        seed=42,
    ):
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
            "\nLoading DRMref resistant-cell signatures..."
        )

        drmref_cache_path = Path(
            drmref_cache_path
        )

        if not drmref_cache_path.exists():
            raise FileNotFoundError(
                "Missing DRMref resistance cache:\n"
                f"{drmref_cache_path}"
            )

        cache = np.load(
            drmref_cache_path,
            allow_pickle=False,
        )

        required = {
            "weights",
            "mask",
            "drug_names",
            "genes",
        }

        missing = required - set(
            cache.files
        )

        if missing:
            raise ValueError(
                "DRMref cache missing keys: "
                + ", ".join(
                    sorted(missing)
                )
            )

        weights_np = cache[
            "weights"
        ].astype(
            np.float32,
            copy=False,
        )

        mask_np = cache[
            "mask"
        ].astype(
            np.float32,
            copy=False,
        )

        drug_names = cache[
            "drug_names"
        ].astype(str)

        genes = [
            clean_gene_name(
                str(gene)
            )
            for gene in cache[
                "genes"
            ].astype(str)
        ]

        if weights_np.ndim != 2:
            raise ValueError(
                "DRMref weights must be 2-D."
            )

        if mask_np.shape != weights_np.shape:
            raise ValueError(
                "DRMref weight/mask shapes differ: "
                f"{weights_np.shape} vs {mask_np.shape}"
            )

        if weights_np.shape[0] != len(
            drug_names
        ):
            raise ValueError(
                "DRMref drug count mismatch."
            )

        if weights_np.shape[1] != len(
            genes
        ):
            raise ValueError(
                "DRMref gene count mismatch."
            )

        if not np.isfinite(
            weights_np
        ).all():
            raise ValueError(
                "DRMref weights contain NaN/Inf."
            )

        if len(set(genes)) != len(
            genes
        ):
            raise ValueError(
                "Duplicate cleaned DRMref genes."
            )

        if not np.array_equal(
            weights_np != 0,
            mask_np != 0,
        ):
            raise ValueError(
                "DRMref mask does not match "
                "non-zero signed weights."
            )

        self.drmref_gene_cols = list(
            genes
        )

        self.drmref_gene_dim = len(
            self.drmref_gene_cols
        )

        # Exact cache.
        self.drmref_weight_cache = {
            str(drug):
                torch.from_numpy(
                    weights_np[i]
                )
            for i, drug
            in enumerate(
                drug_names
            )
        }

        self.drmref_mask_cache = {
            str(drug):
                torch.from_numpy(
                    mask_np[i]
                )
            for i, drug
            in enumerate(
                drug_names
            )
        }

        # Normalized fallback cache.
        norm_weights = {}
        norm_masks = {}

        for i, drug in enumerate(
            drug_names
        ):
            norm = _normalize_drug_name(
                drug
            )

            if not norm:
                continue

            current_w = torch.from_numpy(
                weights_np[i]
            )

            current_m = torch.from_numpy(
                mask_np[i]
            )

            if norm in norm_weights:
                if not torch.equal(
                    norm_weights[
                        norm
                    ],
                    current_w,
                ):
                    raise ValueError(
                        "Conflicting DRMref weights "
                        f"for '{norm}'."
                    )

                if not torch.equal(
                    norm_masks[
                        norm
                    ],
                    current_m,
                ):
                    raise ValueError(
                        "Conflicting DRMref masks "
                        f"for '{norm}'."
                    )
            else:
                norm_weights[
                    norm
                ] = current_w

                norm_masks[
                    norm
                ] = current_m

        self.drmref_normalized_weight_cache = (
            norm_weights
        )

        self.drmref_normalized_mask_cache = (
            norm_masks
        )

        self.drmref_zero_weights = torch.zeros(
            self.drmref_gene_dim,
            dtype=torch.float32,
        )

        self.drmref_zero_mask = torch.zeros(
            self.drmref_gene_dim,
            dtype=torch.float32,
        )

        # ====================================================
        # Case-insensitive DRMref -> DepMap gene alignment
        # ====================================================

        expression = pd.read_parquet(
            expression_path
        ).copy()

        expression.columns = [
            (
                "depmap_id"
                if col == "depmap_id"
                else clean_gene_name(
                    col
                )
            )
            for col in expression.columns
        ]

        expression = expression.set_index(
            "depmap_id"
        )

        depmap_lookup = {}
        ambiguous = set()

        for col in expression.columns:
            key = str(
                col
            ).strip().upper()

            if key in depmap_lookup:
                if depmap_lookup[
                    key
                ] != col:
                    ambiguous.add(
                        key
                    )
            else:
                depmap_lookup[
                    key
                ] = col

        if ambiguous:
            raise ValueError(
                "Ambiguous DepMap gene symbols: "
                f"{sorted(ambiguous)[:20]}"
            )

        self.drmref_expression_cols = []
        missing_genes = []

        for gene in self.drmref_gene_cols:
            resolved = depmap_lookup.get(
                str(
                    gene
                ).strip().upper()
            )

            if resolved is None:
                missing_genes.append(
                    gene
                )
            else:
                self.drmref_expression_cols.append(
                    resolved
                )

        if missing_genes:
            raise ValueError(
                f"{len(missing_genes)} DRMref genes "
                "missing from DepMap after "
                "case-insensitive alignment. "
                f"First missing: {missing_genes[:20]}"
            )

        case_adjusted = sum(
            str(gene) != str(col)
            for gene, col
            in zip(
                self.drmref_gene_cols,
                self.drmref_expression_cols,
            )
        )

        print(
            f"Resolved {case_adjusted:,} DRMref "
            "gene symbols using DepMap "
            "case/name adjustment."
        )

        expression.index = (
            expression.index
            .astype(str)
        )

        dataset_cells = {
            str(x)
            for x in self.depmap_ids
        }

        missing_cells = (
            dataset_cells
            - set(
                expression.index
            )
        )

        if missing_cells:
            raise ValueError(
                f"{len(missing_cells)} PHAROS "
                "cells missing from expression."
            )

        self.drmref_cell_expr_cache = {}

        for depmap_id in sorted(
            dataset_cells
        ):
            self.drmref_cell_expr_cache[
                depmap_id
            ] = torch.from_numpy(
                expression.loc[
                    depmap_id,
                    self.drmref_expression_cols,
                ]
                .to_numpy(
                    dtype=np.float32
                )
            )

        del expression

        # ====================================================
        # Resolve each used PHAROS drug
        # ====================================================

        used_drugs = sorted(
            self.get_unique_drugs()
        )

        self.drmref_resolved_weight_cache = {}
        self.drmref_resolved_mask_cache = {}
        self.drmref_available_cache = {}
        self.drmref_gene_count_cache = {}

        exact = 0
        fallback = 0
        unresolved = 0

        for drug in used_drugs:
            if drug in self.drmref_weight_cache:
                weights = (
                    self.drmref_weight_cache[
                        drug
                    ]
                )

                mask = (
                    self.drmref_mask_cache[
                        drug
                    ]
                )

                exact += 1
            else:
                norm = _normalize_drug_name(
                    drug
                )

                weights = (
                    self.drmref_normalized_weight_cache
                    .get(
                        norm
                    )
                )

                mask = (
                    self.drmref_normalized_mask_cache
                    .get(
                        norm
                    )
                )

                if (
                    weights is None
                    or mask is None
                ):
                    weights = (
                        self.drmref_zero_weights
                    )

                    mask = (
                        self.drmref_zero_mask
                    )

                    unresolved += 1
                else:
                    fallback += 1

            count = int(
                torch.count_nonzero(
                    mask
                ).item()
            )

            self.drmref_resolved_weight_cache[
                drug
            ] = weights

            self.drmref_resolved_mask_cache[
                drug
            ] = mask

            self.drmref_available_cache[
                drug
            ] = (
                count > 0
            )

            self.drmref_gene_count_cache[
                drug
            ] = count

        # ====================================================
        # Coverage diagnostics
        # ====================================================

        a = np.asarray(
            [
                self.drmref_available_cache[
                    drug
                ]
                for drug
                in self.drug_a_names
            ],
            dtype=bool,
        )

        b = np.asarray(
            [
                self.drmref_available_cache[
                    drug
                ]
                for drug
                in self.drug_b_names
            ],
            dtype=bool,
        )

        any_match = a | b
        both = a & b
        one = a ^ b

        matched_norms = {
            _normalize_drug_name(
                drug
            )
            for drug
            in used_drugs
            if self.drmref_available_cache[
                drug
            ]
        }

        print(
            "\n========================================"
        )

        print(
            "DRMREF REAL RESISTANT-CELL FEATURES"
        )

        print(
            "========================================"
        )

        print(
            f"DRMref genes: "
            f"{self.drmref_gene_dim:,}"
        )

        print(
            f"Cached cell vectors: "
            f"{len(self.drmref_cell_expr_cache):,}"
        )

        print(
            f"Used drugs resolved by exact name: "
            f"{exact:,}"
        )

        print(
            f"Used drugs resolved by normalized fallback: "
            f"{fallback:,}"
        )

        print(
            f"Used drugs with no cache row: "
            f"{unresolved:,}"
        )

        print(
            f"Unique normalized drugs with "
            f"non-zero DRMref evidence: "
            f"{len(matched_norms):,}"
        )

        print(
            f"Rows with Drug A evidence: "
            f"{a.sum():,} "
            f"({100 * a.mean():.2f}%)"
        )

        print(
            f"Rows with Drug B evidence: "
            f"{b.sum():,} "
            f"({100 * b.mean():.2f}%)"
        )

        print(
            f"Rows with >=1 matched drug: "
            f"{any_match.sum():,} "
            f"({100 * any_match.mean():.2f}%)"
        )

        print(
            f"Rows with exactly one matched drug: "
            f"{one.sum():,} "
            f"({100 * one.mean():.2f}%)"
        )

        print(
            f"Rows with both drugs matched: "
            f"{both.sum():,} "
            f"({100 * both.mean():.2f}%)"
        )

        print(
            f"Rows with neither drug matched: "
            f"{(~any_match).sum():,} "
            f"({100 * (~any_match).mean():.2f}%)"
        )

        print(
            "Core 29-gene resistance branch: retained"
        )

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

        depmap_id = str(
            self.depmap_ids[
                idx
            ]
        )

        cell_expr = (
            self.drmref_cell_expr_cache[
                depmap_id
            ]
        )

        weights_a = (
            self.drmref_resolved_weight_cache[
                drug_a
            ]
        )

        weights_b = (
            self.drmref_resolved_weight_cache[
                drug_b
            ]
        )

        mask_a = (
            self.drmref_resolved_mask_cache[
                drug_a
            ]
        )

        mask_b = (
            self.drmref_resolved_mask_cache[
                drug_b
            ]
        )

        # Signed resistant-cell feature:
        # DepMap cell expression × DRMref log2FC.
        drmref_expr_a = (
            cell_expr
            * weights_a
        )

        drmref_expr_b = (
            cell_expr
            * weights_b
        )

        sample.update(
            {
                "drug_a_drmref_expr":
                    drmref_expr_a,

                "drug_b_drmref_expr":
                    drmref_expr_b,

                "drug_a_drmref_weights":
                    weights_a,

                "drug_b_drmref_weights":
                    weights_b,

                "drug_a_drmref_mask":
                    mask_a,

                "drug_b_drmref_mask":
                    mask_b,

                "drug_a_drmref_available":
                    torch.tensor(
                        [
                            float(
                                self.drmref_available_cache[
                                    drug_a
                                ]
                            )
                        ],
                        dtype=torch.float32,
                    ),

                "drug_b_drmref_available":
                    torch.tensor(
                        [
                            float(
                                self.drmref_available_cache[
                                    drug_b
                                ]
                            )
                        ],
                        dtype=torch.float32,
                    ),

                "drug_a_drmref_gene_count":
                    torch.tensor(
                        [
                            self.drmref_gene_count_cache[
                                drug_a
                            ]
                        ],
                        dtype=torch.float32,
                    ),

                "drug_b_drmref_gene_count":
                    torch.tensor(
                        [
                            self.drmref_gene_count_cache[
                                drug_b
                            ]
                        ],
                        dtype=torch.float32,
                    ),
            }
        )

        return sample
