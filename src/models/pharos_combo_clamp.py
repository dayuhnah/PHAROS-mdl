import torch
import torch.nn as nn


class PharosComboCLAMP(nn.Module):
    """
    PHAROS-Combo using frozen pretrained CLAMP drug embeddings.

    Drug:
        Frozen 768-dimensional CLAMP embedding
        -> shared trainable projection

    Drug pair:
        zA + zB
        |zA - zB|
        zA * zB

    Cell:
        Full gene-expression MLP

    Prediction:
        Continuous ZIP synergy score

    OFF:
        Molecular GNN
        PPI-GNN
        Resistance FiLM
    """

    def __init__(
        self,
        clamp_dim=768,
        cell_dim=19176,
        drug_out_dim=256,
        pair_out_dim=256,
        cell_out_dim=256,
        hidden_dim=512,
        dropout=0.2,
    ):
        super().__init__()

        # ====================================================
        # Shared projection for pretrained CLAMP embeddings
        # ====================================================

        self.drug_encoder = nn.Sequential(
            nn.Linear(
                clamp_dim,
                hidden_dim,
            ),
            nn.LayerNorm(
                hidden_dim
            ),
            nn.ReLU(),
            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim,
                drug_out_dim,
            ),
            nn.LayerNorm(
                drug_out_dim
            ),
            nn.ReLU(),
        )

        # ====================================================
        # Permutation-invariant drug-pair encoder
        # ====================================================

        self.pair_encoder = nn.Sequential(
            nn.Linear(
                drug_out_dim * 3,
                hidden_dim,
            ),
            nn.LayerNorm(
                hidden_dim
            ),
            nn.ReLU(),
            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim,
                pair_out_dim,
            ),
            nn.LayerNorm(
                pair_out_dim
            ),
            nn.ReLU(),
        )

        # ====================================================
        # Full-expression cell encoder
        # ====================================================

        self.cell_encoder = nn.Sequential(
            nn.Linear(
                cell_dim,
                hidden_dim,
            ),
            nn.LayerNorm(
                hidden_dim
            ),
            nn.ReLU(),
            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim,
                cell_out_dim,
            ),
            nn.LayerNorm(
                cell_out_dim
            ),
            nn.ReLU(),
        )

        # ====================================================
        # Final prediction head
        # ====================================================

        fusion_dim = (
            pair_out_dim
            + cell_out_dim
        )

        self.response_head = nn.Sequential(
            nn.Linear(
                fusion_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim,
                hidden_dim // 2,
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dim // 2,
                1,
            ),
        )

    def forward(
        self,
        drug_a_clamp,
        drug_b_clamp,
        cell_expr,
        return_details=False,
    ):

        # ====================================================
        # Shared pretrained-drug projection
        # ====================================================

        z_a = self.drug_encoder(
            drug_a_clamp
        )

        z_b = self.drug_encoder(
            drug_b_clamp
        )

        # ====================================================
        # Symmetric pair representation
        # ====================================================

        z_sum = (
            z_a
            + z_b
        )

        z_absdiff = torch.abs(
            z_a
            - z_b
        )

        z_product = (
            z_a
            * z_b
        )

        pair_features = torch.cat(
            [
                z_sum,
                z_absdiff,
                z_product,
            ],
            dim=-1,
        )

        z_pair = self.pair_encoder(
            pair_features
        )

        # ====================================================
        # Cell representation
        # ====================================================

        z_cell = self.cell_encoder(
            cell_expr
        )

        # ====================================================
        # Prediction
        # ====================================================

        fused = torch.cat(
            [
                z_pair,
                z_cell,
            ],
            dim=-1,
        )

        prediction = self.response_head(
            fused
        )

        if return_details:

            return {
                "prediction": prediction,
                "z_a": z_a,
                "z_b": z_b,
                "z_pair": z_pair,
                "z_cell": z_cell,
            }

        return prediction