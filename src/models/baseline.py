import torch
import torch.nn as nn


class BaselineDrugResponseModel(nn.Module):
    def __init__(self, drug_dim, cell_dim, resistance_dim, hidden_dim=512, dropout=0.2):
        super().__init__()

        self.drug_encoder = nn.Sequential(
            nn.Linear(drug_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        self.cell_encoder = nn.Sequential(
            nn.Linear(cell_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        self.resistance_encoder = nn.Sequential(
            nn.Linear(resistance_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        fusion_dim = (hidden_dim // 2) + (hidden_dim // 2) + 64

        self.predictor = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, drug_fp, cell_expr, resistance_expr):
        z_drug = self.drug_encoder(drug_fp)
        z_cell = self.cell_encoder(cell_expr)
        z_resistance = self.resistance_encoder(resistance_expr)

        z = torch.cat([z_drug, z_cell, z_resistance], dim=1)
        return self.predictor(z)