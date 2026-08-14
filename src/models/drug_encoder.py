import torch.nn as nn


class MorganFingerprintEncoder(nn.Module):
    def __init__(self, drug_dim: int = 2048, hidden_dim: int = 512, out_dim: int = 256, dropout: float = 0.2):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(drug_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, drug_fp):
        return self.encoder(drug_fp)
    
from torch_geometric.nn import GATv2Conv, global_mean_pool


class GATv2DrugGraphEncoder(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 256,
        heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.gat1 = GATv2Conv(
            in_channels=node_dim,
            out_channels=hidden_dim,
            heads=heads,
            edge_dim=edge_dim,
            concat=True,
            dropout=dropout,
        )

        self.gat2 = GATv2Conv(
            in_channels=hidden_dim * heads,
            out_channels=hidden_dim,
            heads=heads,
            edge_dim=edge_dim,
            concat=False,
            dropout=dropout,
        )

        self.out = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.gat1(x, edge_index, edge_attr).relu()
        x = self.gat2(x, edge_index, edge_attr).relu()
        x = global_mean_pool(x, batch)
        return self.out(x)