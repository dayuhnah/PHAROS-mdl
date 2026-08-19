import torch
import torch.nn as nn

from src.models.pharos_combo_clamp_hybrid import (
    MorganCLAMPHybridEncoder,
)

from src.models.resistance_film import (
    ResistanceFiLM,
)


class DRMrefPairResidualEncoder(nn.Module):
    """
    Encode real DRMref resistant-vs-sensitive cancer-cell signatures.

    Inputs:
        drug_a_drmref_expr: [B, G]
        drug_b_drmref_expr: [B, G]
        drug_a_available:   [B, 1]
        drug_b_available:   [B, 1]

    The same per-drug encoder is used for Drug A and Drug B.

    The pair representation is permutation-invariant:
        zA + zB
        |zA - zB|
        zA * zB

    Unavailable drugs are explicitly forced to zero after encoding,
    because Linear-layer biases could otherwise create non-zero
    embeddings from all-zero DRMref inputs.
    """

    def __init__(
        self,
        input_dim=6481,
        per_drug_hidden_dim=256,
        per_drug_out_dim=64,
        pair_out_dim=64,
        dropout=0.2,
    ):
        super().__init__()

        self.shared_encoder = nn.Sequential(
            nn.LayerNorm(
                input_dim
            ),
            nn.Linear(
                input_dim,
                per_drug_hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                per_drug_hidden_dim,
                per_drug_out_dim,
            ),
            nn.LayerNorm(
                per_drug_out_dim
            ),
            nn.GELU(),
        )

        pair_input_dim = (
            per_drug_out_dim * 3
        )

        self.pair_encoder = nn.Sequential(
            nn.Linear(
                pair_input_dim,
                128,
            ),
            nn.LayerNorm(
                128
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                128,
                pair_out_dim,
            ),
            nn.LayerNorm(
                pair_out_dim
            ),
            nn.GELU(),
        )

    def forward(
        self,
        drug_a_drmref_expr,
        drug_b_drmref_expr,
        drug_a_available,
        drug_b_available,
        return_details=False,
    ):

        if drug_a_available.dim() == 1:
            drug_a_available = (
                drug_a_available
                .unsqueeze(-1)
            )

        if drug_b_available.dim() == 1:
            drug_b_available = (
                drug_b_available
                .unsqueeze(-1)
            )

        drug_a_available = (
            drug_a_available.to(
                dtype=drug_a_drmref_expr.dtype,
                device=drug_a_drmref_expr.device,
            )
        )

        drug_b_available = (
            drug_b_available.to(
                dtype=drug_b_drmref_expr.dtype,
                device=drug_b_drmref_expr.device,
            )
        )

        z_a = (
            self.shared_encoder(
                drug_a_drmref_expr
            )
            * drug_a_available
        )

        z_b = (
            self.shared_encoder(
                drug_b_drmref_expr
            )
            * drug_b_available
        )

        z_sum = (
            z_a
            + z_b
        )

        z_absdiff = torch.abs(
            z_a
            - z_b
        )

        z_product = (
            z_a
            * z_b
        )

        pair_features = torch.cat(
            [
                z_sum,
                z_absdiff,
                z_product,
            ],
            dim=-1,
        )

        z_pair = (
            self.pair_encoder(
                pair_features
            )
        )

        availability_sum = (
            drug_a_available
            + drug_b_available
        )

        pair_available = torch.clamp(
            availability_sum,
            min=0.0,
            max=1.0,
        )

        both_available = (
            drug_a_available
            * drug_b_available
        )

        # No DRMref evidence -> exactly zero database embedding.
        z_pair = (
            z_pair
            * pair_available
        )

        if return_details:
            return {
                "z_drmref_a":
                    z_a,

                "z_drmref_b":
                    z_b,

                "z_drmref_pair":
                    z_pair,

                "drmref_pair_available":
                    pair_available,

                "drmref_availability_sum":
                    availability_sum,

                "drmref_both_available":
                    both_available,
            }

        return z_pair


class PharosComboCLAMPResistanceDRMref(nn.Module):
    """
    PHAROS-Combo Resistance V3.

    This model preserves the successful V1 resistance pathway:

        29 core resistance genes
            -> pair-conditioned ResistanceFiLM
            -> z_core
            -> final prediction

    and adds DRMref as a gated residual:

        z_resistance = z_core + alpha * z_drmref

    where alpha is learned from the drug-pair embedding and DRMref
    availability, and alpha is forced to 0 when neither drug has
    DRMref evidence.

    Therefore, for rows with no DRMref evidence:

        z_resistance == z_core

    structurally preserving the V1 path instead of replacing it with
    an additional fusion MLP.
    """

    def __init__(
        self,
        fingerprint_dim=2048,
        clamp_dim=768,
        cell_dim=19176,
        resistance_dim=29,
        drmref_dim=6481,

        drug_out_dim=256,
        pair_out_dim=256,
        cell_out_dim=256,
        resistance_out_dim=64,
        drmref_out_dim=64,

        drmref_hidden_dim=256,
        drmref_per_drug_dim=64,

        hidden_dim=512,
        dropout=0.2,
    ):
        super().__init__()

        # ====================================================
        # Shared Morgan + frozen CLAMP drug encoder
        # ====================================================

        self.drug_encoder = (
            MorganCLAMPHybridEncoder(
                fingerprint_dim=fingerprint_dim,
                clamp_dim=clamp_dim,
                out_dim=drug_out_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
        )

        # ====================================================
        # Symmetric drug-pair encoder
        # ====================================================

        self.pair_encoder = nn.Sequential(
            nn.Linear(
                drug_out_dim * 3,
                hidden_dim,
            ),
            nn.LayerNorm(
                hidden_dim
            ),
            nn.ReLU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                hidden_dim,
                pair_out_dim,
            ),
            nn.LayerNorm(
                pair_out_dim
            ),
            nn.ReLU(),
        )

        # ====================================================
        # General cell encoder
        # ====================================================

        self.cell_encoder = nn.Sequential(
            nn.Linear(
                cell_dim,
                hidden_dim,
            ),
            nn.LayerNorm(
                hidden_dim
            ),
            nn.ReLU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                hidden_dim,
                cell_out_dim,
            ),
            nn.LayerNorm(
                cell_out_dim
            ),
            nn.ReLU(),
        )

        # ====================================================
        # Original V1 core resistance pathway
        # ====================================================

        self.resistance_film = ResistanceFiLM(
            drug_dim=pair_out_dim,
            resistance_dim=resistance_dim,
            hidden_dim=128,
            output_dim=resistance_out_dim,
            dropout=dropout,
        )

        # ====================================================
        # DRMref real resistant-cell branch
        # ====================================================

        self.drmref_encoder = (
            DRMrefPairResidualEncoder(
                input_dim=drmref_dim,
                per_drug_hidden_dim=drmref_hidden_dim,
                per_drug_out_dim=drmref_per_drug_dim,
                pair_out_dim=drmref_out_dim,
                dropout=dropout,
            )
        )

        if drmref_out_dim != resistance_out_dim:
            self.drmref_projection = nn.Linear(
                drmref_out_dim,
                resistance_out_dim,
            )
        else:
            self.drmref_projection = nn.Identity()

        # ====================================================
        # Scalar residual gate
        #
        # Input:
        #   pair embedding [256]
        #   availability sum [1]
        #   both-available flag [1]
        #
        # Final bias starts negative so DRMref begins as a small
        # correction rather than overwhelming the already-working
        # core resistance pathway.
        # ====================================================

        self.drmref_gate_hidden = nn.Sequential(
            nn.Linear(
                pair_out_dim + 2,
                64,
            ),
            nn.ReLU(),
            nn.Dropout(
                dropout
            ),
        )

        self.drmref_gate_out = nn.Linear(
            64,
            1,
        )

        nn.init.zeros_(
            self.drmref_gate_out.weight
        )

        nn.init.constant_(
            self.drmref_gate_out.bias,
            -2.0,
        )

        # ====================================================
        # Final fusion -- SAME DIMENSIONS AS V1
        # ====================================================

        fusion_dim = (
            pair_out_dim
            + cell_out_dim
            + resistance_out_dim
        )

        self.fusion = nn.Sequential(
            nn.Linear(
                fusion_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                hidden_dim,
                hidden_dim // 2,
            ),
            nn.ReLU(),
            nn.Dropout(
                dropout
            ),
        )

        self.response_head = nn.Linear(
            hidden_dim // 2,
            1,
        )

    def forward(
        self,
        drug_a_fp,
        drug_a_clamp,
        drug_b_fp,
        drug_b_clamp,
        cell_expr,
        resistance_expr,
        drug_a_drmref_expr,
        drug_b_drmref_expr,
        drug_a_drmref_available,
        drug_b_drmref_available,
        return_details=False,
    ):

        # ====================================================
        # Drug A / Drug B
        # ====================================================

        if return_details:
            (
                z_a,
                gate_a,
                z_morgan_a,
                z_clamp_a,
            ) = self.drug_encoder(
                drug_a_fp,
                drug_a_clamp,
                return_gate=True,
            )

            (
                z_b,
                gate_b,
                z_morgan_b,
                z_clamp_b,
            ) = self.drug_encoder(
                drug_b_fp,
                drug_b_clamp,
                return_gate=True,
            )

        else:
            z_a = self.drug_encoder(
                drug_a_fp,
                drug_a_clamp,
            )

            z_b = self.drug_encoder(
                drug_b_fp,
                drug_b_clamp,
            )

        # ====================================================
        # Permutation-invariant molecular pair
        # ====================================================

        z_sum = (
            z_a
            + z_b
        )

        z_absdiff = torch.abs(
            z_a
            - z_b
        )

        z_product = (
            z_a
            * z_b
        )

        pair_features = torch.cat(
            [
                z_sum,
                z_absdiff,
                z_product,
            ],
            dim=-1,
        )

        z_pair = self.pair_encoder(
            pair_features
        )

        # ====================================================
        # Cell
        # ====================================================

        z_cell = self.cell_encoder(
            cell_expr
        )

        # ====================================================
        # Core V1 resistance
        # ====================================================

        z_core = self.resistance_film(
            resistance_features=resistance_expr,
            drug_embedding=z_pair,
        )

        # ====================================================
        # DRMref branch
        # ====================================================

        if return_details:
            drmref_details = self.drmref_encoder(
                drug_a_drmref_expr,
                drug_b_drmref_expr,
                drug_a_drmref_available,
                drug_b_drmref_available,
                return_details=True,
            )

            z_drmref_raw = (
                drmref_details[
                    "z_drmref_pair"
                ]
            )

            pair_available = (
                drmref_details[
                    "drmref_pair_available"
                ]
            )

            availability_sum = (
                drmref_details[
                    "drmref_availability_sum"
                ]
            )

            both_available = (
                drmref_details[
                    "drmref_both_available"
                ]
            )
        else:
            z_drmref_raw = self.drmref_encoder(
                drug_a_drmref_expr,
                drug_b_drmref_expr,
                drug_a_drmref_available,
                drug_b_drmref_available,
            )

            if drug_a_drmref_available.dim() == 1:
                drug_a_drmref_available = (
                    drug_a_drmref_available
                    .unsqueeze(-1)
                )

            if drug_b_drmref_available.dim() == 1:
                drug_b_drmref_available = (
                    drug_b_drmref_available
                    .unsqueeze(-1)
                )

            availability_sum = (
                drug_a_drmref_available
                + drug_b_drmref_available
            )

            pair_available = torch.clamp(
                availability_sum,
                min=0.0,
                max=1.0,
            )

            both_available = (
                drug_a_drmref_available
                * drug_b_drmref_available
            )

        z_drmref = (
            self.drmref_projection(
                z_drmref_raw
            )
        )

        gate_input = torch.cat(
            [
                z_pair,
                availability_sum.to(
                    dtype=z_pair.dtype,
                    device=z_pair.device,
                ),
                both_available.to(
                    dtype=z_pair.dtype,
                    device=z_pair.device,
                ),
            ],
            dim=-1,
        )

        alpha = torch.sigmoid(
            self.drmref_gate_out(
                self.drmref_gate_hidden(
                    gate_input
                )
            )
        )

        # No DRMref evidence -> alpha exactly 0.
        alpha = (
            alpha
            * pair_available.to(
                dtype=alpha.dtype,
                device=alpha.device,
            )
        )

        z_resistance = (
            z_core
            + alpha
            * z_drmref
        )

        # ====================================================
        # Final prediction
        # ====================================================

        fused = torch.cat(
            [
                z_pair,
                z_cell,
                z_resistance,
            ],
            dim=-1,
        )

        hidden = self.fusion(
            fused
        )

        prediction = self.response_head(
            hidden
        )

        if return_details:
            return {
                "prediction":
                    prediction,

                "z_a":
                    z_a,

                "z_b":
                    z_b,

                "z_pair":
                    z_pair,

                "z_cell":
                    z_cell,

                "z_core_resistance":
                    z_core,

                "z_drmref_resistance":
                    z_drmref,

                "z_resistance":
                    z_resistance,

                "drmref_alpha":
                    alpha,

                "drmref_pair_available":
                    pair_available,

                "drmref_availability_sum":
                    availability_sum,

                "drmref_both_available":
                    both_available,

                "z_drmref_a":
                    drmref_details[
                        "z_drmref_a"
                    ],

                "z_drmref_b":
                    drmref_details[
                        "z_drmref_b"
                    ],

                "drug_a_clamp_gate":
                    gate_a,

                "drug_b_clamp_gate":
                    gate_b,

                "z_morgan_a":
                    z_morgan_a,

                "z_clamp_a":
                    z_clamp_a,

                "z_morgan_b":
                    z_morgan_b,

                "z_clamp_b":
                    z_clamp_b,
            }

        return prediction
