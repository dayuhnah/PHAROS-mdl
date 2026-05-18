import torch.nn as nn


class ExpressionEncoder(nn.Module):
    def __init__(
        self,
        cell_dim: int,
        hidden_dim: int = 512,
        out_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(cell_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, cell_expr):
        return self.encoder(cell_expr)


class ResistanceEncoder(nn.Module):
    def __init__(
        self,
        resistance_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(resistance_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, resistance_expr):
        return self.encoder(resistance_expr)


class PretrainedCellEmbeddingEncoder(nn.Module):
    """
    Placeholder encoder for future scGPT/Geneformer/scFoundation embeddings.

    Expected future input:
    - precomputed cell embedding vector instead of raw expression
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 512,
        out_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, cell_embedding):
        return self.encoder(cell_embedding)