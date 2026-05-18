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