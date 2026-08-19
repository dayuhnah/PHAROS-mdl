import torch
import torch.nn as nn


class PharosComboBaseline(nn.Module):
    """
    Simple two-drug baseline.

    Drug:
        Morgan fingerprint only.

    Cell:
        Full gene-expression MLP only.

    Drug pair:
        Symmetric interaction:
            A + B
            |A - B|
            A * B

    No:
        - Molecular GNN
        - PPI-GNN
        - Resistance FiLM
    """

    def __init__(
        self,
        drug_fp_dim=2048,
        cell_dim=19176,
        drug_out_dim=256,
        cell_out_dim=256,
        pair_out_dim=256,
        hidden_dim=512,
        dropout=0.2,
    ):
        super().__init__()

        # ====================================================
        # Shared Morgan drug encoder
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
        # Symmetric drug-pair encoder
        #
        # 256 * 3 = 768
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
        # Response predictor
        # ====================================================

        fusion_dim = (
            pair_out_dim
            +
            cell_out_dim
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
        drug_a_fp,
        drug_b_fp,
        cell_expr,
    ):

        # ====================================================
        # Shared drug encoder
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
            +
            z_b
        )

        z_diff = torch.abs(
            z_a
            -
            z_b
        )

        z_product = (
            z_a
            *
            z_b
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
        # Cell representation
        # ====================================================

        z_cell = self.cell_encoder(
            cell_expr
        )

        # ====================================================
        # Final prediction
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

        return prediction