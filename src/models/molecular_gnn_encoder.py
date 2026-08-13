import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    GINEConv,
    global_mean_pool,
    global_max_pool,
)


class MolecularGNNEncoder(nn.Module):
    """
    Molecular graph encoder using GINEConv.

    Node features:
        18 atom features

    Edge features:
        6 bond features

    Output:
        Fixed-size drug embedding
    """

    def __init__(
        self,
        node_dim: int = 18,
        edge_dim: int = 6,
        hidden_dim: int = 128,
        out_dim: int = 256,
        heads: int = 4,  # kept for compatibility, unused by GINE
        dropout: float = 0.2,
    ):
        super().__init__()

        self.dropout = dropout

        # ---------------------------------
        # Initial atom projection
        # ---------------------------------

        self.node_projection = nn.Sequential(
            nn.Linear(
                node_dim,
                hidden_dim,
            ),
            nn.ReLU(),
        )

        # =================================
        # GINE layer 1
        # =================================

        mlp1 = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
        )

        self.conv1 = GINEConv(
            nn=mlp1,
            edge_dim=edge_dim,
            train_eps=True,
        )

        self.norm1 = nn.LayerNorm(
            hidden_dim
        )

        # =================================
        # GINE layer 2
        # =================================

        mlp2 = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
        )

        self.conv2 = GINEConv(
            nn=mlp2,
            edge_dim=edge_dim,
            train_eps=True,
        )

        self.norm2 = nn.LayerNorm(
            hidden_dim
        )

        # =================================
        # Graph projection
        # =================================

        self.graph_projection = nn.Sequential(
            nn.Linear(
                hidden_dim * 2,
                out_dim,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        batch,
    ):

        # ---------------------------------
        # Atom projection
        # ---------------------------------

        x = self.node_projection(x)

        # ---------------------------------
        # GINE 1 + residual
        # ---------------------------------

        residual = x

        x = self.conv1(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        x = self.norm1(
            x + residual
        )

        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        # ---------------------------------
        # GINE 2 + residual
        # ---------------------------------

        residual = x

        x = self.conv2(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        x = self.norm2(
            x + residual
        )

        x = F.relu(x)

        # ---------------------------------
        # Graph pooling
        # ---------------------------------

        mean_pool = global_mean_pool(
            x,
            batch,
        )

        max_pool = global_max_pool(
            x,
            batch,
        )

        graph_embedding = torch.cat(
            [
                mean_pool,
                max_pool,
            ],
            dim=-1,
        )

        return self.graph_projection(
            graph_embedding
        )