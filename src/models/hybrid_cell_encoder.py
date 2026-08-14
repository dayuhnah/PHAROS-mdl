import torch
import torch.nn as nn

from src.models.pharos import ExpressionEncoder
from src.models.ppi_cell_gnn import PPICellGNNEncoder


class HybridCellEncoder(nn.Module):
    """
    Hybrid cancer-cell representation.

    Branch 1:
        Full non-resistance gene expression -> MLP

    Branch 2:
        PPI-aligned gene expression -> PPI GNN

    A feature-wise gate learns how much information
    to use from each representation.
    """

    def __init__(
        self,
        full_expression_dim,
        ppi_edge_index,
        ppi_edge_weight,
        num_ppi_genes,
        expression_hidden_dim=512,
        ppi_hidden_dim=64,
        out_dim=256,
        dropout=0.2,
    ):
        super().__init__()

        self.out_dim = out_dim

        # ==========================================
        # Full transcriptomic branch
        # ==========================================

        self.expression_encoder = ExpressionEncoder(
            cell_dim=full_expression_dim,
            hidden_dim=expression_hidden_dim,
            out_dim=out_dim,
            dropout=dropout,
        )

        # ==========================================
        # PPI biological graph branch
        # ==========================================

        self.ppi_encoder = PPICellGNNEncoder(
            edge_index=ppi_edge_index,
            edge_weight=ppi_edge_weight,
            num_genes=num_ppi_genes,
            gene_embedding_dim=16,
            hidden_dim=ppi_hidden_dim,
            out_dim=out_dim,
            dropout=dropout,
        )

        # ==========================================
        # Adaptive fusion gate
        # ==========================================

        self.gate = nn.Sequential(
            nn.Linear(
                out_dim * 2,
                out_dim,
            ),
            nn.ReLU(),

            nn.Linear(
                out_dim,
                out_dim,
            ),
            nn.Sigmoid(),
        )

        # ==========================================
        # Refinement
        # ==========================================

        self.refinement = nn.Sequential(
            nn.Linear(
                out_dim,
                out_dim,
            ),
            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                out_dim,
                out_dim,
            ),
        )

        self.output_norm = nn.LayerNorm(
            out_dim
        )

    def forward(
        self,
        full_expression,
        ppi_expression,
        return_gate=False,
    ):

        # ==========================================
        # Full expression embedding
        # ==========================================

        z_expression = self.expression_encoder(
            full_expression
        )

        # ==========================================
        # PPI embedding
        # ==========================================

        z_ppi = self.ppi_encoder(
            ppi_expression
        )

        # ==========================================
        # Adaptive fusion
        # ==========================================

        combined = torch.cat(
            [
                z_expression,
                z_ppi,
            ],
            dim=-1,
        )

        gate = self.gate(
            combined
        )

        # gate -> 1:
        # favour PPI GNN
        #
        # gate -> 0:
        # favour full-expression MLP

        fused = (
            gate * z_ppi
            +
            (1.0 - gate) * z_expression
        )

        # ==========================================
        # Residual refinement
        # ==========================================

        refined = self.refinement(
            fused
        )

        z_cell = self.output_norm(
            fused + refined
        )

        if return_gate:
            return (
                z_cell,
                gate,
                z_expression,
                z_ppi,
            )

        return z_cell