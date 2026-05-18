import torch
import torch.nn as nn

from src.models.cell_encoder import PretrainedCellEmbeddingEncoder, ResistanceEncoder
from src.models.drug_encoder import MorganFingerprintEncoder


class PharosEmbeddingModel(nn.Module):
    """
    PHAROS variant that uses precomputed cell embeddings.

    Current use:
    - PCA cell embeddings

    Future use:
    - scGPT / Geneformer / scFoundation embeddings
    """

    def __init__(
        self,
        drug_dim: int,
        cell_embedding_dim: int,
        resistance_dim: int,
        hidden_dim: int = 512,
        drug_out_dim: int = 256,
        cell_out_dim: int = 256,
        resistance_out_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.drug_encoder = MorganFingerprintEncoder(
            drug_dim=drug_dim,
            hidden_dim=hidden_dim,
            out_dim=drug_out_dim,
            dropout=dropout,
        )

        self.cell_encoder = PretrainedCellEmbeddingEncoder(
            embedding_dim=cell_embedding_dim,
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

    def forward(self, drug_fp, cell_embedding, resistance_expr):
        z_drug = self.drug_encoder(drug_fp)
        z_cell = self.cell_encoder(cell_embedding)
        z_resistance = self.resistance_encoder(resistance_expr)

        z = torch.cat([z_drug, z_cell, z_resistance], dim=1)
        z_bioactivity = self.bioactivity_bridge(z)

        return self.response_head(z_bioactivity)