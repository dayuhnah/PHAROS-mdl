import torch
import torch.nn as nn

from src.models.pharos_combo_clamp_hybrid import (
    MorganCLAMPHybridEncoder,
)

from src.models.pharos_combo_clamp_resistance_drmref import (
    DRMrefPairResidualEncoder,
)

from src.models.v4.multiomics_cell_encoder import (
    MultiOmicsCellEncoder,
)

from src.models.v5.cellresdb_resistance_encoder_v51 import (
    CellResDBResistanceEncoderV51,
)


class PharosMocktailV51(nn.Module):

    def __init__(
        self,
        fingerprint_dim=2048,
        clamp_dim=768,
        expression_dim=19176,
        cnv_dim=19926,
        mutation_dim=15606,
        crispr_dim=18502,
        drmref_dim=6481,
        drug_dim=256,
        pair_dim=256,
        cell_dim=256,
        resistance_out_dim=64,
        hidden_dim=512,
        dropout=0.2,
        cellresdb_cache_path=(
            "data/processed/v5/cellresdb/"
            "resistance_bank_cohort/model_cache/"
            "cellresdb_token_tensors.pt"
        ),
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
        # CELLRESDB PATIENT-DERIVED RESISTANCE
        #
        # V5.1 improvements:
        #
        # 1. Condition retrieval on drug pair + cell state
        # 2. NULL / no-evidence attention token
        # 3. Learned reliability gate
        # ====================================================

        self.cellresdb_encoder = (
            CellResDBResistanceEncoderV51(
                cache_path=cellresdb_cache_path,
                pair_dim=pair_dim,
                cell_dim=cell_dim,
                hidden_dim=128,
                out_dim=resistance_out_dim,
                num_heads=4,
                dropout=dropout,
            )
        )

        # ====================================================
        # CELLRESDB RELEVANCE GATE
        #
        # Determines how strongly patient-derived evidence
        # should contribute for THIS drug pair + cell.
        #
        # Inputs:
        #   drug pair      = 256
        #   cell           = 256
        #   CellResDB      = 64
        #
        # Output:
        #   scalar g in [0,1]
        # ====================================================

        self.cellresdb_gate = nn.Sequential(
            nn.Linear(
                pair_dim
                + cell_dim
                + resistance_out_dim,
                128,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                128,
                1,
            ),
            nn.Sigmoid(),
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
        #
        # Keep matched dimensionality with V4/V5:
        #
        # drug pair       256
        # cell            256
        # resistance       64
        #
        # total            576
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

        # ----------------------------------------------------
        # Drug A
        # ----------------------------------------------------
        drug_a_fp,
        drug_a_clamp,

        # ----------------------------------------------------
        # Drug B
        # ----------------------------------------------------
        drug_b_fp,
        drug_b_clamp,

        # ----------------------------------------------------
        # Multi-omics
        # ----------------------------------------------------
        expression,
        cnv,
        mutation,
        crispr,
        omics_mask,

        # ----------------------------------------------------
        # DRMref
        # ----------------------------------------------------
        drug_a_drmref_expr,
        drug_b_drmref_expr,
        drug_a_drmref_available,
        drug_b_drmref_available,

        cellresdb_token_relevance=None,

        return_details=False,
    ):

        # ====================================================
        # DRUG A + DRUG B
        # ====================================================

        z_a = self.drug_encoder(
            drug_a_fp,
            drug_a_clamp,
        )

        z_b = self.drug_encoder(
            drug_b_fp,
            drug_b_clamp,
        )

        # ----------------------------------------------------
        # Permutation-invariant drug-pair representation
        #
        # Swapping A and B produces the same pair features.
        # ----------------------------------------------------

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
        # MULTI-OMICS CELL
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
        # CELLRESDB
        #
        # Always retrieve attention information because
        # NULL/no-evidence relevance must also affect training.
        # ====================================================

        cellresdb_details = (
            self.cellresdb_encoder(
                z_pair,
                z_cell,
                token_relevance=cellresdb_token_relevance,
                return_attention=True,
            )
        )

        z_cellresdb = (
            cellresdb_details["embedding"]
        )

        # ====================================================
        # CELLRESDB RELEVANCE GATING
        # ====================================================

        cellresdb_gate_input = torch.cat(
            [
                z_pair,
                z_cell,
                z_cellresdb,
            ],
            dim=-1,
        )

        cellresdb_gate = self.cellresdb_gate(
            cellresdb_gate_input
        )

        # NULL attention means:
        # "CellResDB has little/no useful evidence
        #  for this drug-pair/cell example."
        null_attention = (
            cellresdb_details[
                "null_attention"
            ]
        )

        evidence_relevance = (
            1.0 - null_attention
        )

        effective_cellresdb_gate = (
            cellresdb_gate
            * evidence_relevance
        )

        z_cellresdb_gated = (
            z_cellresdb
            * effective_cellresdb_gate
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

        # DRMref requires cell expression.
        #
        # This remains independent from CellResDB.
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
        # RESISTANCE FUSION
        #
        # V4:
        #
        #   Core29
        #       +
        #   DRMref
        #
        # V5:
        #
        #   CellResDB
        #       +
        #   DRMref
        #
        # V5.1:
        #
        #   gated CellResDB
        #       +
        #   DRMref
        # ====================================================

        z_resistance = (
            z_cellresdb_gated
            +
            alpha
            * z_drmref_gated
        )

        # ====================================================
        # FINAL PHAROS MOCKTAIL PREDICTION
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

        # ====================================================
        # INTERPRETABILITY OUTPUT
        # ====================================================

        return {
            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            "prediction":
                prediction,

            # ------------------------------------------------
            # Drug representations
            # ------------------------------------------------

            "drug_a_embedding":
                z_a,

            "drug_b_embedding":
                z_b,

            "drug_pair_embedding":
                z_pair,

            # ------------------------------------------------
            # Cell
            # ------------------------------------------------

            "cell_embedding":
                z_cell,

            "modality_weights":
                cell_details[
                    "modality_weights"
                ],

            # ------------------------------------------------
            # CellResDB
            # ------------------------------------------------

            "cellresdb_embedding":
                z_cellresdb,

            "cellresdb_gated_embedding":
                z_cellresdb_gated,

            "cellresdb_token_attention":
                cellresdb_details[
                    "token_attention"
                ],

            "cellresdb_pathway_attention":
                cellresdb_details[
                    "pathway_attention"
                ],

            "cellresdb_null_attention":
                null_attention,

            "cellresdb_gate":
                cellresdb_gate,

            "cellresdb_evidence_relevance":
                evidence_relevance,

            "cellresdb_effective_gate":
                effective_cellresdb_gate,

            # ------------------------------------------------
            # DRMref
            # ------------------------------------------------

            "drmref_embedding":
                z_drmref,

            "drmref_gated_embedding":
                z_drmref_gated,

            "drmref_drug_available":
                drug_drm_available,

            "drmref_available":
                drm_available,

            "drmref_alpha":
                alpha,

            # ------------------------------------------------
            # General resistance
            # ------------------------------------------------

            "resistance_embedding":
                z_resistance,

            "expression_available":
                expression_available,
        }
