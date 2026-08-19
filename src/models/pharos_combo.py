import torch
import torch.nn as nn

from src.models.drug_pair_encoder import (
    DrugPairEncoder,
)

from src.models.hybrid_cell_encoder import (
    HybridCellEncoder,
)

from src.models.resistance_film import (
    ResistanceFiLM,
)


class PharosComboModel(nn.Module):
    """
    PHAROS-Combo

    Drug A:
        Morgan + Molecular GINE
                ↓

    Drug B:
        Morgan + Molecular GINE
                ↓

        Shared DrugPairEncoder
                ↓
            z_pair

    Cell:
        Full expression
            +
        PPI-GNN
            ↓
        adaptive fusion
            ↓
          z_cell

    Resistance:
        29 resistance genes
            ↓
        FiLM conditioned on z_pair
            ↓
        z_resistance

    Final:
        z_pair
        +
        z_cell
        +
        z_resistance
            ↓
        ZIP synergy prediction
    """

    def __init__(
        self,

        drug_fp_dim,
        cell_dim,
        resistance_dim,

        ppi_edge_index,
        ppi_edge_weight,
        num_ppi_genes,

        node_dim=18,
        edge_dim=6,

        hidden_dim=512,

        drug_gnn_hidden_dim=128,
        ppi_hidden_dim=64,

        drug_out_dim=256,
        pair_out_dim=256,
        cell_out_dim=256,
        resistance_out_dim=64,

        dropout=0.2,
    ):
        super().__init__()

        # =====================================================
        # Drug-pair encoder
        # =====================================================

        self.drug_pair_encoder = (
            DrugPairEncoder(

                node_dim=node_dim,
                edge_dim=edge_dim,

                fingerprint_dim=(
                    drug_fp_dim
                ),

                gnn_hidden_dim=(
                    drug_gnn_hidden_dim
                ),

                drug_out_dim=(
                    drug_out_dim
                ),

                pair_hidden_dim=(
                    hidden_dim
                ),

                pair_out_dim=(
                    pair_out_dim
                ),

                dropout=dropout,
            )
        )

        # =====================================================
        # Hybrid cell encoder
        # =====================================================

        self.cell_encoder = (
            HybridCellEncoder(

                full_expression_dim=(
                    cell_dim
                ),

                ppi_edge_index=(
                    ppi_edge_index
                ),

                ppi_edge_weight=(
                    ppi_edge_weight
                ),

                num_ppi_genes=(
                    num_ppi_genes
                ),

                expression_hidden_dim=(
                    hidden_dim
                ),

                ppi_hidden_dim=(
                    ppi_hidden_dim
                ),

                out_dim=(
                    cell_out_dim
                ),

                dropout=dropout,
            )
        )

        # =====================================================
        # Combination-conditioned resistance
        #
        # IMPORTANT:
        #
        # This is no longer conditioned on ONE drug.
        #
        # It is conditioned on the learned drug-pair
        # representation z_pair.
        # =====================================================

        self.resistance_film = (
            ResistanceFiLM(

                drug_dim=(
                    pair_out_dim
                ),

                resistance_dim=(
                    resistance_dim
                ),

                hidden_dim=128,

                output_dim=(
                    resistance_out_dim
                ),

                dropout=dropout,
            )
        )

        # =====================================================
        # Final fusion
        # =====================================================

        fusion_dim = (
            pair_out_dim
            + cell_out_dim
            + resistance_out_dim
        )

        self.bioactivity_bridge = (
            nn.Sequential(

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
            )
        )

        # =====================================================
        # ZIP regression head
        # =====================================================

        self.response_head = (
            nn.Linear(
                hidden_dim // 2,
                1,
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

        cell_expr,
        ppi_expression,
        resistance_expr,

        return_details=False,
    ):

        # =====================================================
        # Drug-pair representation
        # =====================================================

        if return_details:

            pair_output = (
                self.drug_pair_encoder(

                    drug_a_graph=(
                        drug_a_graph
                    ),

                    drug_a_fp=(
                        drug_a_fp
                    ),

                    drug_b_graph=(
                        drug_b_graph
                    ),

                    drug_b_fp=(
                        drug_b_fp
                    ),

                    return_details=True,
                )
            )

            z_pair = (
                pair_output[
                    "z_pair"
                ]
            )

        else:

            z_pair = (
                self.drug_pair_encoder(

                    drug_a_graph=(
                        drug_a_graph
                    ),

                    drug_a_fp=(
                        drug_a_fp
                    ),

                    drug_b_graph=(
                        drug_b_graph
                    ),

                    drug_b_fp=(
                        drug_b_fp
                    ),
                )
            )

        # =====================================================
        # Cell representation
        # =====================================================

        if return_details:

            (
                z_cell,
                cell_gate,
                z_expression,
                z_ppi,
            ) = self.cell_encoder(

                cell_expr,
                ppi_expression,

                return_gate=True,
            )

        else:

            z_cell = (
                self.cell_encoder(

                    cell_expr,
                    ppi_expression,
                )
            )

        # =====================================================
        # Pair-conditioned resistance
        # =====================================================

        z_resistance = (
            self.resistance_film(

                resistance_features=(
                    resistance_expr
                ),

                drug_embedding=(
                    z_pair
                ),
            )
        )

        # =====================================================
        # Final fusion
        # =====================================================

        fused = torch.cat(
            [
                z_pair,
                z_cell,
                z_resistance,
            ],
            dim=1,
        )

        hidden = (
            self.bioactivity_bridge(
                fused
            )
        )

        prediction = (
            self.response_head(
                hidden
            )
        )

        # =====================================================
        # Diagnostics
        # =====================================================

        if return_details:

            return {

                "prediction":
                    prediction,

                "z_pair":
                    z_pair,

                "z_cell":
                    z_cell,

                "z_resistance":
                    z_resistance,

                "cell_gate":
                    cell_gate,

                "z_expression":
                    z_expression,

                "z_ppi":
                    z_ppi,

                # ---------------------------------------------
                # Drug pair internals
                # ---------------------------------------------

                "z_a":
                    pair_output[
                        "z_a"
                    ],

                "z_b":
                    pair_output[
                        "z_b"
                    ],

                "z_sum":
                    pair_output[
                        "z_sum"
                    ],

                "z_diff":
                    pair_output[
                        "z_diff"
                    ],

                "z_product":
                    pair_output[
                        "z_product"
                    ],

                "drug_a_gate":
                    pair_output[
                        "drug_a_gate"
                    ],

                "drug_b_gate":
                    pair_output[
                        "drug_b_gate"
                    ],

                "z_graph_a":
                    pair_output[
                        "z_graph_a"
                    ],

                "z_fp_a":
                    pair_output[
                        "z_fp_a"
                    ],

                "z_graph_b":
                    pair_output[
                        "z_graph_b"
                    ],

                "z_fp_b":
                    pair_output[
                        "z_fp_b"
                    ],
            }

        return prediction