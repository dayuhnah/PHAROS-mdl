import torch
import torch.nn as nn

from src.models.hybrid_drug_encoder import (
    HybridDrugEncoder,
)

from src.models.pharos import (
    ExpressionEncoder,
)

from src.models.resistance_film import (
    ResistanceFiLM,
)


class PharosRXHybridModel(nn.Module):
    """
    PHAROS-RX with a hybrid drug representation:

        Molecular GINE
              +
        Morgan fingerprint
              ↓
        adaptive gated fusion
              ↓
           z_drug

    Cell:
        full gene-expression MLP

    Resistance:
        drug-conditioned FiLM
    """

    def __init__(
        self,
        drug_fp_dim,
        cell_dim,
        resistance_dim,

        node_dim=18,
        edge_dim=6,

        hidden_dim=512,

        drug_gnn_hidden_dim=128,

        drug_out_dim=256,
        cell_out_dim=256,
        resistance_out_dim=64,

        dropout=0.2,
    ):
        super().__init__()

        # ==========================================
        # Hybrid drug encoder
        # ==========================================

        self.drug_encoder = HybridDrugEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            fingerprint_dim=drug_fp_dim,
            gnn_hidden_dim=drug_gnn_hidden_dim,
            out_dim=drug_out_dim,
            dropout=dropout,
        )

        # ==========================================
        # Full expression encoder
        # ==========================================

        self.cell_encoder = ExpressionEncoder(
            cell_dim=cell_dim,
            hidden_dim=hidden_dim,
            out_dim=cell_out_dim,
            dropout=dropout,
        )

        # ==========================================
        # Resistance FiLM
        # ==========================================

        self.resistance_film = ResistanceFiLM(
            drug_dim=drug_out_dim,
            resistance_dim=resistance_dim,
            hidden_dim=128,
            output_dim=resistance_out_dim,
            dropout=dropout,
        )

        # ==========================================
        # Fusion
        # ==========================================

        fusion_dim = (
            drug_out_dim
            + cell_out_dim
            + resistance_out_dim
        )

        self.bioactivity_bridge = nn.Sequential(

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

        self.response_head = nn.Linear(
            hidden_dim // 2,
            1,
        )

    def forward(
        self,
        drug_graph,
        drug_fp,
        cell_expr,
        resistance_expr,
        return_gate=False,
    ):

        # ==========================================
        # Hybrid drug
        # ==========================================

        if return_gate:

            (
                z_drug,
                gate,
                z_graph,
                z_fp,
            ) = self.drug_encoder(
                drug_graph,
                drug_fp,
                return_gate=True,
            )

        else:

            z_drug = self.drug_encoder(
                drug_graph,
                drug_fp,
            )

        # ==========================================
        # Cell state
        # ==========================================

        z_cell = self.cell_encoder(
            cell_expr
        )

        # ==========================================
        # Drug-conditioned resistance
        # ==========================================

        z_resistance = self.resistance_film(
            resistance_features=resistance_expr,
            drug_embedding=z_drug,
        )

        # ==========================================
        # Fusion
        # ==========================================

        fused = torch.cat(
            [
                z_drug,
                z_cell,
                z_resistance,
            ],
            dim=1,
        )

        hidden = self.bioactivity_bridge(
            fused
        )

        prediction = self.response_head(
            hidden
        )

        if return_gate:

            return {
                "prediction":
                    prediction,

                "gate":
                    gate,

                "z_drug":
                    z_drug,

                "z_graph":
                    z_graph,

                "z_fp":
                    z_fp,

                "z_cell":
                    z_cell,

                "z_resistance":
                    z_resistance,
            }

        return prediction