import torch
import torch.nn as nn

from src.models.pharos_combo_clamp_hybrid import (
    MorganCLAMPHybridEncoder,
)
from src.models.pharos_combo_clamp_resistance_drmref import (
    DRMrefPairResidualEncoder,
)
from src.models.resistance_film import ResistanceFiLM
from src.models.v4.multiomics_cell_encoder import (
    MultiOmicsCellEncoder,
)


class PharosMocktailV4(nn.Module):

    def __init__(
        self,
        fingerprint_dim=2048,
        clamp_dim=768,

        expression_dim=19176,
        cnv_dim=19926,
        mutation_dim=15606,
        crispr_dim=18502,

        resistance_dim=29,
        drmref_dim=6481,

        drug_dim=256,
        pair_dim=256,
        cell_dim=256,
        resistance_out_dim=64,

        hidden_dim=512,
        dropout=0.2,
    ):
        super().__init__()

        # ====================================================
        # DRUG ENCODER
        # ====================================================

        self.drug_encoder = MorganCLAMPHybridEncoder(
            fingerprint_dim=fingerprint_dim,
            clamp_dim=clamp_dim,
            out_dim=drug_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.drug_pair_projection = nn.Sequential(
            nn.Linear(
                drug_dim * 3,
                pair_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(
                pair_dim,
                pair_dim,
            ),
            nn.GELU(),
        )

        # ====================================================
        # MULTI-OMICS CELL ENCODER
        # ====================================================

        self.cell_encoder = MultiOmicsCellEncoder(
            expression_dim=expression_dim,
            cnv_dim=cnv_dim,
            mutation_dim=mutation_dim,
            crispr_dim=crispr_dim,

            modality_dim=128,
            output_dim=cell_dim,

            dropout=dropout,
        )

        # ====================================================
        # CORE29 RESISTANCE
        # ====================================================

        self.core_resistance = ResistanceFiLM(
            drug_dim=pair_dim,
            resistance_dim=resistance_dim,
            hidden_dim=128,
            output_dim=resistance_out_dim,
            dropout=dropout,
        )

        # ====================================================
        # DRMref
        # ====================================================

        self.drmref_encoder = DRMrefPairResidualEncoder(
            input_dim=drmref_dim,
            per_drug_hidden_dim=256,
            per_drug_out_dim=64,
            pair_out_dim=resistance_out_dim,
            dropout=dropout,
        )

        self.drmref_alpha_logit = nn.Parameter(
            torch.tensor(0.0)
        )

        # ====================================================
        # FINAL FUSION
        # ====================================================

        fusion_dim = (
            pair_dim
            + cell_dim
            + resistance_out_dim
        )

        self.fusion = nn.Sequential(
            nn.Linear(
                fusion_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(
                hidden_dim,
                256,
            ),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(
                256,
                1,
            ),
        )


    def forward(
        self,

        # Drug A
        drug_a_fp,
        drug_a_clamp,

        # Drug B
        drug_b_fp,
        drug_b_clamp,

        # Multi-omics
        expression,
        cnv,
        mutation,
        crispr,
        omics_mask,

        # Core29
        resistance_expr,
        core29_available,

        # DRMref
        drug_a_drmref_expr,
        drug_b_drmref_expr,
        drug_a_drmref_available,
        drug_b_drmref_available,

        return_details=False,
    ):

        # ====================================================
        # DRUGS
        # ====================================================

        z_a = self.drug_encoder(
            drug_a_fp,
            drug_a_clamp,
        )

        z_b = self.drug_encoder(
            drug_b_fp,
            drug_b_clamp,
        )

        # Permutation-invariant pair features
        pair_features = torch.cat(
            [
                z_a + z_b,
                torch.abs(
                    z_a - z_b
                ),
                z_a * z_b,
            ],
            dim=-1,
        )

        z_pair = self.drug_pair_projection(
            pair_features
        )

        # ====================================================
        # CELL
        # ====================================================

        if return_details:

            cell_details = self.cell_encoder(
                expression,
                cnv,
                mutation,
                crispr,
                omics_mask,
                return_weights=True,
            )

            z_cell = cell_details[
                "cell_embedding"
            ]

        else:

            z_cell = self.cell_encoder(
                expression,
                cnv,
                mutation,
                crispr,
                omics_mask,
            )

        # ====================================================
        # CORE29
        # ====================================================

        z_core = self.core_resistance(
            resistance_expr,
            z_pair,
        )

        core29_available = (
            core29_available.float()
        )

        # Completely shut Core29 off when unavailable /
        # intentionally ablated.
        z_core = (
            z_core
            * core29_available
        )

        # ====================================================
        # DRMref
        # ====================================================

        z_drmref = self.drmref_encoder(
            drug_a_drmref_expr,
            drug_b_drmref_expr,
            drug_a_drmref_available,
            drug_b_drmref_available,
        )

        drug_drm_available = torch.clamp(
            drug_a_drmref_available.float()
            +
            drug_b_drmref_available.float(),
            min=0.0,
            max=1.0,
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # DRMref depends on cell expression, NOT Core29.
        #
        # This keeps DRMref active during the no-Core29
        # ablation whenever:
        #
        #   1. expression exists for the cell
        #   2. DRMref exists for at least one drug
        # ----------------------------------------------------

        expression_available = (
            omics_mask[:, 0:1].float()
        )

        drm_available = (
            drug_drm_available
            * expression_available
        )

        alpha = torch.sigmoid(
            self.drmref_alpha_logit
        )

        z_drmref_gated = (
            z_drmref
            * drm_available
        )

        # ====================================================
        # RESISTANCE
        # ====================================================

        z_resistance = (
            z_core
            +
            alpha * z_drmref_gated
        )

        # ====================================================
        # FINAL PREDICTION
        # ====================================================

        combined = torch.cat(
            [
                z_pair,
                z_cell,
                z_resistance,
            ],
            dim=-1,
        )

        prediction = self.fusion(
            combined
        )

        if not return_details:
            return prediction

        return {
            "prediction":
                prediction,

            "drug_a_embedding":
                z_a,

            "drug_b_embedding":
                z_b,

            "drug_pair_embedding":
                z_pair,

            "cell_embedding":
                z_cell,

            "modality_weights":
                cell_details[
                    "modality_weights"
                ],

            "core29_embedding":
                z_core,

            "drmref_embedding":
                z_drmref,

            "drmref_gated_embedding":
                z_drmref_gated,

            "resistance_embedding":
                z_resistance,

            "core29_available":
                core29_available,

            "expression_available":
                expression_available,

            "drmref_drug_available":
                drug_drm_available,

            "drmref_available":
                drm_available,

            "drmref_alpha":
                alpha,
        }
