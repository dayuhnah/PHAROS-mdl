import torch
import torch.nn as nn

from src.models.hybrid_drug_encoder import (
    HybridDrugEncoder,
)


class DrugPairEncoder(nn.Module):
    """
    PHAROS-Combo drug-pair encoder.

    Drug A and Drug B are encoded using the SAME
    HybridDrugEncoder (shared weights).

    Each drug:
        Morgan fingerprint
            +
        Molecular GINE
            ↓
        gated hybrid representation
            ↓
        z_a / z_b

    Pair representation:
        z_sum  = z_a + z_b
        z_diff = |z_a - z_b|
        z_prod = z_a * z_b

        [z_sum, z_diff, z_prod]
            ↓
        pair interaction MLP
            ↓
        z_pair

    Because all pair operations are symmetric:

        encode(A, B) = encode(B, A)

    This makes the representation permutation invariant.
    """

    def __init__(
        self,
        node_dim=18,
        edge_dim=6,
        fingerprint_dim=2048,

        gnn_hidden_dim=128,
        drug_out_dim=256,

        pair_hidden_dim=512,
        pair_out_dim=256,

        dropout=0.2,
    ):
        super().__init__()

        self.drug_out_dim = (
            drug_out_dim
        )

        self.pair_out_dim = (
            pair_out_dim
        )

        # =====================================================
        # ONE shared encoder
        # =====================================================

        self.drug_encoder = (
            HybridDrugEncoder(
                node_dim=node_dim,
                edge_dim=edge_dim,
                fingerprint_dim=(
                    fingerprint_dim
                ),
                gnn_hidden_dim=(
                    gnn_hidden_dim
                ),
                out_dim=drug_out_dim,
                dropout=dropout,
            )
        )

        # =====================================================
        # Symmetric pair interaction
        #
        # 256 + 256 + 256 = 768
        # =====================================================

        pair_input_dim = (
            drug_out_dim
            * 3
        )

        self.pair_encoder = (
            nn.Sequential(

                nn.Linear(
                    pair_input_dim,
                    pair_hidden_dim,
                ),

                nn.LayerNorm(
                    pair_hidden_dim
                ),

                nn.ReLU(),

                nn.Dropout(
                    dropout
                ),

                nn.Linear(
                    pair_hidden_dim,
                    pair_out_dim,
                ),

                nn.LayerNorm(
                    pair_out_dim
                ),

                nn.ReLU(),
            )
        )

        # =====================================================
        # Pair refinement
        # =====================================================

        self.refinement = (
            nn.Sequential(

                nn.Linear(
                    pair_out_dim,
                    pair_out_dim,
                ),

                nn.ReLU(),

                nn.Dropout(
                    dropout
                ),

                nn.Linear(
                    pair_out_dim,
                    pair_out_dim,
                ),
            )
        )

        self.output_norm = (
            nn.LayerNorm(
                pair_out_dim
            )
        )

    # =========================================================
    # Forward
    # =========================================================

    def forward(
        self,

        drug_a_graph,
        drug_a_fp,

        drug_b_graph,
        drug_b_fp,

        return_details=False,
    ):

        # =====================================================
        # Encode Drug A
        # =====================================================

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

        else:

            z_a = self.drug_encoder(

                drug_a_graph,
                drug_a_fp,
            )

        # =====================================================
        # Encode Drug B
        #
        # IMPORTANT:
        # This calls THE SAME self.drug_encoder.
        # No second set of weights exists.
        # =====================================================

        if return_details:

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

            z_b = self.drug_encoder(

                drug_b_graph,
                drug_b_fp,
            )

        # =====================================================
        # Symmetric drug-pair interactions
        # =====================================================

        z_sum = (
            z_a
            +
            z_b
        )

        z_diff = torch.abs(
            z_a
            -
            z_b
        )

        z_product = (
            z_a
            *
            z_b
        )

        # =====================================================
        # Concatenate symmetric interactions
        # =====================================================

        pair_features = torch.cat(
            [
                z_sum,
                z_diff,
                z_product,
            ],
            dim=-1,
        )

        # =====================================================
        # Learn pair representation
        # =====================================================

        z_pair_base = (
            self.pair_encoder(
                pair_features
            )
        )

        refined = (
            self.refinement(
                z_pair_base
            )
        )

        z_pair = (
            self.output_norm(
                z_pair_base
                +
                refined
            )
        )

        # =====================================================
        # Diagnostics
        # =====================================================

        if return_details:

            return {

                "z_pair":
                    z_pair,

                "z_a":
                    z_a,

                "z_b":
                    z_b,

                "z_sum":
                    z_sum,

                "z_diff":
                    z_diff,

                "z_product":
                    z_product,

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

        return z_pair