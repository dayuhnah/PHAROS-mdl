import torch
import torch.nn as nn 

class ResistanceFiLM(nn.Module):

    def __init__(
            self,
            drug_dim,
            resistance_dim=29,
            hidden_dim=128,
            output_dim=128,
            dropout=0.2,
    ):
        super().__init__()

        self.resistance_encoder = nn.Sequential(
            nn.Linear(resistance_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

        self.film_generator = nn.Sequential(
            nn.Linear(drug_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim * 2),
        )

    def forward(self, resistance_features, drug_embedding):
        r = self.resistance_encoder(resistance_features)

        flim_params = self.film_generator(drug_embedding)

        gamma, beta = torch.chunk(
            flim_params,
            chunks=2,
            dim=-1,
        )

        gamma = 1.0 + gamma

        conditioned_resistance = gamma * r + beta
        
        return conditioned_resistance