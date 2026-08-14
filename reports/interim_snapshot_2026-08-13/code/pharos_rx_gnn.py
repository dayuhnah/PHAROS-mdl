import torch
import torch.nn as nn

from src.models.cell_encoder import ExpressionEncoder
from src.models.molecular_gnn_encoder import MolecularGNNEncoder
from src.models.resistance_film import ResistanceFiLM


class PharosRXGNNModel(nn.Module):

    def __init__(
        self,
        cell_dim,
        resistance_dim,
        node_dim=18,
        edge_dim=6,
        hidden_dim=512,
        drug_gnn_hidden_dim=128,
        drug_out_dim=256,
        cell_out_dim=256,
        resistance_out_dim=64,
        gnn_heads=4,
        dropout=0.2,
    ):
        super().__init__()

        # =================================
        # Drug molecular GNN
        # =================================

        self.drug_encoder = MolecularGNNEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=drug_gnn_hidden_dim,
            out_dim=drug_out_dim,
            heads=gnn_heads,
            dropout=dropout,
        )

        # =================================
        # General cell expression encoder
        # =================================

        self.cell_encoder = ExpressionEncoder(
            cell_dim=cell_dim,
            hidden_dim=hidden_dim,
            out_dim=cell_out_dim,
            dropout=dropout,
        )

        # =================================
        # Drug-conditioned resistance branch
        # =================================

        self.resistance_film = ResistanceFiLM(
            drug_dim=drug_out_dim,
            resistance_dim=resistance_dim,
            hidden_dim=128,
            output_dim=resistance_out_dim,
            dropout=dropout,
        )

        # =================================
        # Fusion
        # =================================

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

            nn.Dropout(dropout),

            nn.Linear(
                hidden_dim,
                hidden_dim // 2,
            ),
            nn.ReLU(),
        )

        # =================================
        # Response prediction
        # =================================

        self.response_head = nn.Linear(
            hidden_dim // 2,
            1,
        )

    def forward(
        self,
        drug_graph,
        cell_expr,
        resistance_expr,
    ):

        # ---------------------------------
        # Molecular GNN
        # ---------------------------------

        z_drug = self.drug_encoder(
            x=drug_graph.x,
            edge_index=drug_graph.edge_index,
            edge_attr=drug_graph.edge_attr,
            batch=drug_graph.batch,
        )

        # ---------------------------------
        # Cell state
        # ---------------------------------

        z_cell = self.cell_encoder(
            cell_expr
        )

        # ---------------------------------
        # Drug-conditioned resistance
        # ---------------------------------

        z_resistance = self.resistance_film(
            resistance_features=resistance_expr,
            drug_embedding=z_drug,
        )

        # ---------------------------------
        # Multimodal fusion
        # ---------------------------------

        z = torch.cat(
            [
                z_drug,
                z_cell,
                z_resistance,
            ],
            dim=1,
        )

        z_bioactivity = (
            self.bioactivity_bridge(z)
        )

        return self.response_head(
            z_bioactivity
        )