import torch
import torch.nn as nn

from src.models.hybrid_drug_encoder import (
    HybridDrugEncoder,
)


class PharosComboDrugGNNAblation(nn.Module):
    """
    PHAROS-Combo molecular-GNN ablation.

    Drug:
        Morgan fingerprint
            +
        Molecular GINE
            ↓
        adaptive gated fusion

    Drug A and Drug B use the SAME shared encoder.

    Pair:
        A + B
        |A - B|
        A * B

    Cell:
        Full gene-expression MLP only

    Removed:
        - PPI-GNN
        - Resistance FiLM

    Purpose:
        Test whether the hybrid molecular representation
        improves held-out drug-pair generalisation over
        Morgan fingerprints alone.
    """

    def __init__(
        self,
        drug_fp_dim=2048,
        cell_dim=19176,

        node_dim=18,
        edge_dim=6,

        drug_gnn_hidden_dim=128,

        drug_out_dim=256,
        pair_out_dim=256,
        cell_out_dim=256,

        hidden_dim=512,

        dropout=0.2,
    ):
        super().__init__()

        # ====================================================
        # Shared Hybrid Drug Encoder
        # ====================================================

        self.drug_encoder = HybridDrugEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,

            fingerprint_dim=drug_fp_dim,

            gnn_hidden_dim=drug_gnn_hidden_dim,

            out_dim=drug_out_dim,

            dropout=dropout,
        )

        # ====================================================
        # Symmetric pair encoder
        #
        # Same general pair construction as baseline:
        #
        # A + B
        # |A - B|
        # A * B
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
        #
        # No PPI branch.
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
        # Response head
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

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,

        drug_a_graph,
        drug_a_fp,

        drug_b_graph,
        drug_b_fp,

        cell_expr,

        return_details=False,
    ):

        # ====================================================
        # Shared hybrid drug encoder
        # ====================================================

        if return_details:

            (
                z_a,
                gate_a,
                z_graph_a,
                z_fp_a,
            ) = self.drug_encoder(

                drug_a_graph,
                drug_a_fp,

                return_gate=True,
            )

            (
                z_b,
                gate_b,
                z_graph_b,
                z_fp_b,
            ) = self.drug_encoder(

                drug_b_graph,
                drug_b_fp,

                return_gate=True,
            )

        else:

            z_a = self.drug_encoder(
                drug_a_graph,
                drug_a_fp,
            )

            z_b = self.drug_encoder(
                drug_b_graph,
                drug_b_fp,
            )

        # ====================================================
        # Symmetric pair features
        # ====================================================

        z_sum = (
            z_a
            + z_b
        )

        z_diff = torch.abs(
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
                z_diff,
                z_product,
            ],
            dim=-1,
        )

        z_pair = self.pair_encoder(
            pair_features
        )

        # ====================================================
        # Cell expression
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

        # ====================================================
        # Diagnostics
        # ====================================================

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

                "drug_a_gate":
                    gate_a,

                "drug_b_gate":
                    gate_b,

                "z_graph_a":
                    z_graph_a,

                "z_fp_a":
                    z_fp_a,

                "z_graph_b":
                    z_graph_b,

                "z_fp_b":
                    z_fp_b,
            }

        return prediction