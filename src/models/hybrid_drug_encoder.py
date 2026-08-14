import torch
import torch.nn as nn

from src.models.molecular_gnn_encoder import (
    MolecularGNNEncoder,
)


class HybridDrugEncoder(nn.Module):
    """
    Hybrid molecular representation.

    Branch 1:
        Molecular graph -> GINE

    Branch 2:
        Morgan fingerprint -> MLP

    A learned feature-wise gate determines
    how much information to retain from each
    representation for every drug.
    """

    def __init__(
        self,
        node_dim=18,
        edge_dim=6,
        fingerprint_dim=2048,
        gnn_hidden_dim=128,
        out_dim=256,
        dropout=0.2,
    ):
        super().__init__()

        self.out_dim = out_dim

        # ==========================================
        # Molecular graph branch
        # ==========================================

        self.graph_encoder = MolecularGNNEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=gnn_hidden_dim,
            out_dim=out_dim,
            dropout=dropout,
        )

        # ==========================================
        # Morgan fingerprint branch
        # ==========================================

        self.fingerprint_encoder = nn.Sequential(

            nn.Linear(
                fingerprint_dim,
                512,
            ),

            nn.LayerNorm(
                512
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                512,
                out_dim,
            ),

            nn.LayerNorm(
                out_dim
            ),

            nn.ReLU(),
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
        # Post-fusion refinement
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
        drug_graph,
        drug_fp,
        return_gate=False,
    ):

        # ==========================================
        # Graph representation
        # ==========================================

        z_graph = self.graph_encoder(
            x=drug_graph.x,
            edge_index=drug_graph.edge_index,
            edge_attr=drug_graph.edge_attr,
            batch=drug_graph.batch,
        )

        # ==========================================
        # Morgan representation
        # ==========================================

        z_fp = self.fingerprint_encoder(
            drug_fp
        )

        # ==========================================
        # Learn graph vs fingerprint importance
        # ==========================================

        combined = torch.cat(
            [
                z_graph,
                z_fp,
            ],
            dim=-1,
        )

        gate = self.gate(
            combined
        )

        # gate close to 1:
        # favour GNN
        #
        # gate close to 0:
        # favour Morgan

        fused = (
            gate * z_graph
            +
            (1.0 - gate) * z_fp
        )

        # ==========================================
        # Residual refinement
        # ==========================================

        refined = self.refinement(
            fused
        )

        z_drug = self.output_norm(
            fused + refined
        )

        if return_gate:

            return (
                z_drug,
                gate,
                z_graph,
                z_fp,
            )

        return z_drug