from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def norm(x):
    return re.sub(
        r"[^A-Z0-9]",
        "",
        str(x).upper(),
    )


class PharosMocktailV4Dataset(Dataset):

    def __init__(
        self,
        split="train",
        seed=42,
    ):

        self.split = split
        self.seed = seed

        BASE = Path("data/processed/v4")
        TRAIN = BASE / "training"
        OMICS = BASE / "multiomics"

        # ====================================================
        # RESPONSE DATA
        # ====================================================

        split_path = (
            TRAIN /
            f"mocktail_v4_split_seed{seed}.parquet"
        )

        df = pd.read_parquet(split_path)

        df = df[
            df["split"] == split
        ].reset_index(drop=True)

        self.df = df

        print(
            f"{split} rows:",
            f"{len(df):,}"
        )

        # ====================================================
        # MORGAN
        # ====================================================

        morgan = np.load(
            BASE /
            "pharos_mocktail_v4_morgan_fingerprints.npz"
        )

        self.morgan = {
            norm(name): torch.from_numpy(
                fp.astype(np.float32)
            )
            for name, fp
            in zip(
                morgan["drug_names"],
                morgan["fingerprints"],
            )
        }

        # ====================================================
        # CLAMP
        # ====================================================

        clamp = np.load(
            BASE /
            "pharos_mocktail_v4_clamp_embeddings.npz"
        )

        self.clamp = {
            norm(name): torch.from_numpy(
                emb.astype(np.float32)
            )
            for name, emb
            in zip(
                clamp["drug_names"],
                clamp["embeddings"],
            )
        }

        # ====================================================
        # MULTI-OMICS
        # ====================================================

        self.expression = self._load_cell_matrix(
            TRAIN /
            f"expression_general_seed{seed}.parquet"
        )

        self.cnv = self._load_cell_matrix(
            TRAIN /
            f"cnv_general_seed{seed}.parquet"
        )

        self.mutation = self._load_cell_matrix(
            TRAIN /
            f"mutation_general_seed{seed}.parquet"
        )

        self.crispr = self._load_cell_matrix(
            TRAIN /
            f"crispr_general_seed{seed}.parquet"
        )

        # Save actual dimensions for model construction
        self.expression_dim = len(
            next(iter(self.expression.values()))
        )

        self.cnv_dim = len(
            next(iter(self.cnv.values()))
        )

        self.mutation_dim = len(
            next(iter(self.mutation.values()))
        )

        self.crispr_dim = len(
            next(iter(self.crispr.values()))
        )

        # ====================================================
        # OMICS AVAILABILITY MASKS
        # ====================================================

        masks = pd.read_parquet(
            OMICS /
            "modality_masks.parquet"
        )

        masks.index = masks.index.astype(str)

        self.omics_masks = {}

        for cell in masks.index:

            row = masks.loc[cell]

            self.omics_masks[cell] = torch.tensor(
                [
                    float(row["has_expression"]),
                    float(row["has_cnv"]),
                    float(row["has_mutation"]),
                    float(row["has_crispr"]),
                ],
                dtype=torch.float32,
            )

        # ====================================================
        # CORE29 EXPRESSION
        # ====================================================

        core_path = (
            TRAIN /
            f"core29_expression_seed{seed}.parquet"
        )

        core = pd.read_parquet(core_path)
        core.index = core.index.astype(str)

        self.core29 = {
            cell: torch.tensor(
                row.values.astype(np.float32)
            )
            for cell, row
            in core.iterrows()
        }

        # Core29 is available only when expression exists
        self.core29_available = {
            cell: torch.tensor(
                [
                    float(
                        masks.loc[
                            cell,
                            "has_expression"
                        ]
                    )
                ],
                dtype=torch.float32,
            )
            for cell in masks.index
        }

        # ====================================================
        # DRMref
        # ====================================================

        drm = np.load(
            "data/processed/combination/drmref/"
            "pharos_drmref_resistance_weights.npz"
        )

        drm_genes = [
            str(g)
            for g in drm["genes"]
        ]

        # ----------------------------------------------------
        # Cell expression for DRMref genes
        #
        # IMPORTANT:
        # DRMref uses raw DepMap expression × DRMref log2FC,
        # not the z-scored V4 general-expression branch.
        # ----------------------------------------------------

        raw_expression = pd.read_parquet(
            OMICS /
            "expression_aligned.parquet"
        )

        raw_expression.index = (
            raw_expression.index.astype(str)
        )

        raw_expression = raw_expression.reindex(
            columns=drm_genes
        )

        raw_expression = (
            raw_expression
            .fillna(0.0)
            .astype(np.float32)
        )

        self.drmref_cell_expression = {
            cell: torch.tensor(
                row.values,
                dtype=torch.float32,
            )
            for cell, row
            in raw_expression.iterrows()
        }

        # ----------------------------------------------------
        # DRMref drug weights
        # ----------------------------------------------------

        zero_drm = torch.zeros(
            len(drm_genes),
            dtype=torch.float32,
        )

        self.zero_drmref = zero_drm

        self.drmref_weights = {}
        self.drmref_available = {}

        for name, weights, mask in zip(
            drm["drug_names"],
            drm["weights"],
            drm["mask"],
        ):

            key = norm(name)

            active = bool(
                np.asarray(mask).sum() > 0
            )

            # Duplicate normalized names can exist.
            # Prefer an active DRMref entry.
            if (
                key in self.drmref_available
                and
                self.drmref_available[key]
                and
                not active
            ):
                continue

            self.drmref_weights[key] = (
                torch.tensor(
                    weights.astype(np.float32)
                )
            )

            self.drmref_available[key] = active

        # ====================================================
        # VALIDATION
        # ====================================================

        drugs = (
            set(df["drug_a_key"].map(norm))
            |
            set(df["drug_b_key"].map(norm))
        )

        missing_morgan = (
            drugs - set(self.morgan)
        )

        missing_clamp = (
            drugs - set(self.clamp)
        )

        if missing_morgan:
            raise RuntimeError(
                f"Missing Morgan drugs: "
                f"{len(missing_morgan)}"
            )

        if missing_clamp:
            raise RuntimeError(
                f"Missing CLAMP drugs: "
                f"{len(missing_clamp)}"
            )

        print(
            "Morgan drugs:",
            len(self.morgan)
        )

        print(
            "CLAMP drugs:",
            len(self.clamp)
        )

        print(
            "Omics dimensions:",
            self.expression_dim,
            self.cnv_dim,
            self.mutation_dim,
            self.crispr_dim,
        )

        active_drm = sum(
            self.drmref_available.values()
        )

        print(
            "Active DRMref drugs:",
            active_drm
        )

    # ========================================================
    # HELPERS
    # ========================================================

    @staticmethod
    def _load_cell_matrix(path):

        df = pd.read_parquet(path)

        df.index = df.index.astype(str)

        return {
            cell: torch.tensor(
                row.values.astype(np.float32)
            )
            for cell, row
            in df.iterrows()
        }

    # ========================================================
    # DATASET
    # ========================================================

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        drug_a = norm(
            row["drug_a_key"]
        )

        drug_b = norm(
            row["drug_b_key"]
        )

        cell = str(
            row["depmap_id"]
        )

        # ----------------------------------------------------
        # DRMref
        # ----------------------------------------------------

        weight_a = self.drmref_weights.get(
            drug_a,
            self.zero_drmref,
        )

        weight_b = self.drmref_weights.get(
            drug_b,
            self.zero_drmref,
        )

        available_a = float(
            self.drmref_available.get(
                drug_a,
                False,
            )
        )

        available_b = float(
            self.drmref_available.get(
                drug_b,
                False,
            )
        )

        cell_drm_expr = (
            self.drmref_cell_expression[cell]
        )

        drmref_a = (
            cell_drm_expr
            * weight_a
        )

        drmref_b = (
            cell_drm_expr
            * weight_b
        )

        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        return {

            "drug_a_fp":
                self.morgan[drug_a],

            "drug_a_clamp":
                self.clamp[drug_a],

            "drug_b_fp":
                self.morgan[drug_b],

            "drug_b_clamp":
                self.clamp[drug_b],

            "expression":
                self.expression[cell],

            "cnv":
                self.cnv[cell],

            "mutation":
                self.mutation[cell],

            "crispr":
                self.crispr[cell],

            "omics_mask":
                self.omics_masks[cell],

            "resistance_expr":
                self.core29[cell],

            "core29_available":
                self.core29_available[cell],

            "drug_a_drmref_expr":
                drmref_a,

            "drug_b_drmref_expr":
                drmref_b,

            "drug_a_drmref_available":
                torch.tensor(
                    [available_a],
                    dtype=torch.float32,
                ),

            "drug_b_drmref_available":
                torch.tensor(
                    [available_b],
                    dtype=torch.float32,
                ),

            "label":
                torch.tensor(
                    [float(row["zip_score"])],
                    dtype=torch.float32,
                ),

            "drug_a":
                drug_a,

            "drug_b":
                drug_b,

            "depmap_id":
                cell,
        }
