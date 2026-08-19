import torch
import torch.nn as nn


class MorganCLAMPHybridEncoder(nn.Module):
    """
    Hybrid drug encoder using:

    1. Morgan fingerprint
    2. Frozen cached CLAMP embedding

    Both are projected to the same latent dimension and
    combined using a learned feature-wise gate.

    gate -> 1 means favour CLAMP
    gate -> 0 means favour Morgan
    """

    def __init__(
        self,
        fingerprint_dim=2048,
        clamp_dim=768,
        out_dim=256,
        hidden_dim=512,
        dropout=0.2,
    ):
        super().__init__()

        # ====================================================
        # Morgan branch
        # ====================================================

        self.morgan_encoder = nn.Sequential(
            nn.Linear(
                fingerprint_dim,
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
                out_dim,
            ),
            nn.LayerNorm(
                out_dim
            ),
            nn.ReLU(),
        )

        # ====================================================
        # Frozen CLAMP embedding projection
        #
        # CLAMP itself is already frozen and cached.
        # Only this projection is trainable.
        # ====================================================

        self.clamp_encoder = nn.Sequential(
            nn.Linear(
                clamp_dim,
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
                out_dim,
            ),
            nn.LayerNorm(
                out_dim
            ),
            nn.ReLU(),
        )

        # ====================================================
        # Feature-wise fusion gate
        # ====================================================

        self.gate_network = nn.Sequential(
            nn.Linear(
                out_dim * 2,
                out_dim,
            ),
            nn.Sigmoid(),
        )

    def forward(
        self,
        drug_fp,
        drug_clamp,
        return_gate=False,
    ):

        z_morgan = self.morgan_encoder(
            drug_fp
        )

        z_clamp = self.clamp_encoder(
            drug_clamp
        )

        gate_input = torch.cat(
            [
                z_morgan,
                z_clamp,
            ],
            dim=-1,
        )

        clamp_gate = self.gate_network(
            gate_input
        )

        # gate = 1 -> CLAMP
        # gate = 0 -> Morgan

        z_drug = (
            clamp_gate
            * z_clamp
            +
            (
                1.0
                - clamp_gate
            )
            * z_morgan
        )

        if return_gate:

            return (
                z_drug,
                clamp_gate,
                z_morgan,
                z_clamp,
            )

        return z_drug


class PharosComboCLAMPHybrid(nn.Module):
    """
    PHAROS-Combo hybrid pretrained drug model.

    Drug representation:
        Morgan fingerprint
            +
        frozen cached CLAMP embedding
            ↓
        learned gated fusion

    Drug pair:
        zA + zB
        |zA - zB|
        zA * zB

    Cell:
        Full gene-expression MLP

    OFF:
        Molecular GNN
        PPI-GNN
        Resistance FiLM
    """

    def __init__(
        self,
        fingerprint_dim=2048,
        clamp_dim=768,
        cell_dim=19176,

        drug_out_dim=256,
        pair_out_dim=256,
        cell_out_dim=256,

        hidden_dim=512,
        dropout=0.2,
    ):
        super().__init__()

        # ====================================================
        # Shared Drug A / Drug B encoder
        # ====================================================

        self.drug_encoder = MorganCLAMPHybridEncoder(
            fingerprint_dim=fingerprint_dim,
            clamp_dim=clamp_dim,
            out_dim=drug_out_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # ====================================================
        # Permutation-invariant pair encoder
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
        # Full-expression cell encoder
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
        # Final response head
        # ====================================================

        fusion_dim = (
            pair_out_dim
            + cell_out_dim
        )

        self.response_head = nn.Sequential(
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

            nn.Linear(
                hidden_dim // 2,
                1,
            ),
        )

    def forward(
        self,

        drug_a_fp,
        drug_a_clamp,

        drug_b_fp,
        drug_b_clamp,

        cell_expr,

        return_details=False,
    ):

        # ====================================================
        # Drug A / B
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
        # Symmetric drug-pair representation
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
        # Response
        # ====================================================

        fused = torch.cat(
            [
                z_pair,
                z_cell,
            ],
            dim=-1,
        )

        prediction = self.response_head(
            fused
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