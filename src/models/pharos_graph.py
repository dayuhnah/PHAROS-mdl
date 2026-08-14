import torch
import torch.nn as nn

from src.models.cell_encoder import ExpressionEncoder, ResistanceEncoder
from src.models.drug_encoder import GATv2DrugGraphEncoder


class PharosGraphModel(nn.Module):
    """
    PHAROS variant using a GATv2 molecular graph encoder.

    Drug: RDKit molecular graph
    Cell: DepMap expression encoder
    Resistance: 29-gene resistance encoder
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        cell_dim: int,
        resistance_dim: int,
        hidden_dim: int = 512,
        graph_hidden_dim: int = 128,
        drug_out_dim: int = 256,
        cell_out_dim: int = 256,
        resistance_out_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.drug_encoder = GATv2DrugGraphEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=graph_hidden_dim,
            out_dim=drug_out_dim,
            heads=4,
            dropout=dropout,
        )

        self.cell_encoder = ExpressionEncoder(
            cell_dim=cell_dim,
            hidden_dim=hidden_dim,
            out_dim=cell_out_dim,
            dropout=dropout,
        )

        self.resistance_encoder = ResistanceEncoder(
            resistance_dim=resistance_dim,
            hidden_dim=128,
            out_dim=resistance_out_dim,
            dropout=dropout,
        )

        fusion_dim = drug_out_dim + cell_out_dim + resistance_out_dim

        self.bioactivity_bridge = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        self.response_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, graph_batch, cell_expr, resistance_expr):
        z_drug = self.drug_encoder(
            graph_batch.x,
            graph_batch.edge_index,
            graph_batch.edge_attr,
            graph_batch.batch,
        )

        z_cell = self.cell_encoder(cell_expr)
        z_resistance = self.resistance_encoder(resistance_expr)

        z = torch.cat([z_drug, z_cell, z_resistance], dim=1)
        z_bioactivity = self.bioactivity_bridge(z)

        return self.response_head(z_bioactivity)