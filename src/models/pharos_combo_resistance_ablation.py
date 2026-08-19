import torch
import torch.nn as nn

from src.models.resistance_film import ResistanceFiLM


class PharosComboResistanceAblation(nn.Module):
    """
    PHAROS-Combo resistance ablation.

    Drug representation:
        Morgan fingerprint only
        Shared encoder for Drug A and Drug B

    Pair representation:
        A + B
        |A - B|
        A * B

    Cell representation:
        Full gene-expression MLP only

    Resistance:
        29 resistance genes
        FiLM conditioned on drug-pair embedding

    Removed:
        - Molecular GNN
        - PPI-GNN
    """

    def __init__(
        self,
        drug_fp_dim=2048,
        cell_dim=19176,
        resistance_dim=29,
        drug_out_dim=256,
        pair_out_dim=256,
        cell_out_dim=256,
        resistance_out_dim=64,
        hidden_dim=512,
        dropout=0.2,
    ):
        super().__init__()

        # ====================================================
        # Shared Morgan encoder
        # ====================================================

        self.drug_encoder = nn.Sequential(
            nn.Linear(
                drug_fp_dim,
                512,
            ),
            nn.LayerNorm(
                512
            ),
            nn.ReLU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                512,
                drug_out_dim,
            ),
            nn.LayerNorm(
                drug_out_dim
            ),
            nn.ReLU(),
        )

        # ====================================================
        # Symmetric pair encoder
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
        # Full expression encoder
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
        # Pair-conditioned resistance branch
        # ====================================================

        self.resistance_film = ResistanceFiLM(
            drug_dim=pair_out_dim,
            resistance_dim=resistance_dim,
            hidden_dim=128,
            output_dim=resistance_out_dim,
            dropout=dropout,
        )

        # ====================================================
        # Final fusion
        # ====================================================

        fusion_dim = (
            pair_out_dim
            + cell_out_dim
            + resistance_out_dim
        )

        self.bioactivity_bridge = nn.Sequential(
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
        )

        self.response_head = nn.Linear(
            hidden_dim // 2,
            1,
        )

    def forward(
        self,
        drug_a_fp,
        drug_b_fp,
        cell_expr,
        resistance_expr,
        return_details=False,
    ):

        # ====================================================
        # Drug A / Drug B
        # ====================================================

        z_a = self.drug_encoder(
            drug_a_fp
        )

        z_b = self.drug_encoder(
            drug_b_fp
        )

        # ====================================================
        # Symmetric pair representation
        # ====================================================

        z_sum = (
            z_a
            + z_b
        )

        z_diff = torch.abs(
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
                z_diff,
                z_product,
            ],
            dim=-1,
        )

        z_pair = self.pair_encoder(
            pair_features
        )

        # ====================================================
        # Cell expression
        # ====================================================

        z_cell = self.cell_encoder(
            cell_expr
        )

        # ====================================================
        # Pair-conditioned resistance
        # ====================================================

        z_resistance = self.resistance_film(
            resistance_features=resistance_expr,
            drug_embedding=z_pair,
        )

        # ====================================================
        # Final prediction
        # ====================================================

        fused = torch.cat(
            [
                z_pair,
                z_cell,
                z_resistance,
            ],
            dim=-1,
        )

        hidden = self.bioactivity_bridge(
            fused
        )

        prediction = self.response_head(
            hidden
        )

        if return_details:

            return {
                "prediction": prediction,
                "z_a": z_a,
                "z_b": z_b,
                "z_pair": z_pair,
                "z_cell": z_cell,
                "z_resistance": z_resistance,
            }

        return prediction