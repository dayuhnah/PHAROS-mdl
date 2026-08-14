import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import(
    GATv2Conv,
    global_mean_pool,
    global_max_pool
)

class MolecularGNNEncoder(nn.Module):
    def __init__(
            self,
            node_dim: int = 18,
            edge_dim: int = 6,
            hidden_dim: int = 128,
            out_dim: int = 256,
            heads: int = 4,
            dropout: float = 0.2
    ):
        super().__init__()

        self.dropout = dropout

        self.node_projection = nn.Linear(
            node_dim,
            hidden_dim
        )

        self.conv1 = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=heads,
            concat=False,
            edge_dim=edge_dim,
            dropout=dropout
        )

        self.norm1 = nn.LayerNorm(
            hidden_dim
        )

        self.conv2 = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=heads,
            concat=False,
            edge_dim=edge_dim,
            dropout=dropout,
        )

        self.norm2 = nn.LayerNorm(
            hidden_dim
        )

        self.conv3 = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=heads,
            concat=False,
            edge_dim=edge_dim,
            dropout=dropout,
        )

        self.norm3 = nn.LayerNorm(
            hidden_dim
        )

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
        
        x = self.node_projection(x)

        #layer 1

        residual = x

        x = self.conv1(
            x,
            edge_index,
            edge_attr = edge_attr
        )

        x = self.norm1(x+residual)
        x = F.relu(x)
        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training
        )

        #layer 2

        residual = x
        
        x = self.conv2(
            x,
            edge_index,
            edge_attr=edge_attr,
        )

        x = self.norm2(x + residual)
        x = F.relu(x)
        x = F.dropout(
            x,
            p=self.dropout,
            training = self.training
        )

        #layer 3
        residual = x
        
        x = self.conv3(
            x,
            edge_index,
            edge_attr=edge_attr,
        )

        x = self.norm3(x + residual)
        x = F.relu(x)

        mean_embedding = global_mean_pool(
            x,
            batch,
        )

        max_embedding = global_max_pool(
            x,
            batch,
        )

        graph_embedding = torch.cat(
            [
                mean_embedding,
                max_embedding,
            ],
            dim=-1,
        )

        return self.graph_projection(
            graph_embedding
        )
