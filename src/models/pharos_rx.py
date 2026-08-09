import torch
import torch.nn as nn

from src.models.cell_encoder import ExpressionEncoder
from src.models.drug_encoder import MorganFingerprintEncoder
from src.models.resistance_film import ResistanceFiLM

class PharosRXModel(nn.Module):
    def __init__(
            self,
            drug_dim: int,
            cell_dim: int,
            resistance_dim: int,
            hidden_dim: int = 512,
            drug_out_dim: int = 256,
            cell_out_dim: int = 256,
            resistance_out_dim: int = 64,
            droupout: float = 0.2,
    ):
        super().__init__()

        self.drug_encoder = MorganFingerprintEncoder(
            drug_dim=drug_dim,
            hidden_dim=hidden_dim,
            out_dim=drug_out_dim,
            dropout=droupout,
        )

        self.cell_encoder = ExpressionEncoder(
            cell_dim=cell_dim,
            hidden_dim=hidden_dim,
            out_dim=cell_out_dim,
            dropout=droupout
        )

        self.resistance_film = ResistanceFiLM(
            drug_dim=drug_out_dim,
            resistance_dim=resistance_dim,
            hidden_dim=128,
            output_dim=resistance_out_dim,
            dropout=droupout
        )

        fusion_dim = (
            drug_out_dim + cell_out_dim + resistance_out_dim
        )

        self.bioactivity_bridge = nn.Sequential(
            nn.Liner(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(droupout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        self.response_head = nn.Linear(
            hidden_dim // 2,
            1
        )

    def forward(
            self,
            drug_fp,
            cell_expr,
            resistance_expr
    ):
        z_drug = self.drug_encoder(drug_fp)

        z_cell = self.cell_encoder(cell_expr)

        z_resistance = self.resistance_film(
            resistance_features = resistance_expr,
            drug_embedding = z_drug,
        )

        z = torch.cat(
            [
                z_drug,
                z_cell,
                z_resistance
            ],
            dim = 1,
        )

        z_bioactivity = self.bioactivity_bridge(z)

        response = self.response_head(z_bioactivity)

        return response