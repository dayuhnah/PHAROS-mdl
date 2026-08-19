import torch
import torch.nn as nn

from src.models.pharos_combo_clamp_hybrid import (
    MorganCLAMPHybridEncoder,
)

from src.models.resistance_film import (
    ResistanceFiLM,
)


class PharosComboCLAMPResistance(nn.Module):
    """
    PHAROS-Combo pretrained + resistance architecture.

    Drug A / Drug B:
        Morgan fingerprint
            +
        Frozen cached CLAMP embedding
            ↓
        Learned Morgan/CLAMP gate
            ↓
        shared drug representation

    Drug pair:
        zA + zB
        |zA - zB|
        zA * zB
            ↓
        pair embedding

    Cell:
        19,176 general expression genes
            ↓
        cell embedding

    Resistance:
        29 dedicated resistance genes
            ↓
        FiLM conditioned by drug-pair embedding
            ↓
        resistance embedding

    Final:
        pair + cell + resistance
            ↓
        ZIP synergy
    """

    def __init__(
        self,
        fingerprint_dim=2048,
        clamp_dim=768,
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
        # Shared Morgan + pretrained CLAMP drug encoder
        # ====================================================

        self.drug_encoder = MorganCLAMPHybridEncoder(
            fingerprint_dim=fingerprint_dim,
            clamp_dim=clamp_dim,
            out_dim=drug_out_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # ====================================================
        # Symmetric drug-pair encoder
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
        # General cancer-cell encoder
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

        self.fusion = nn.Sequential(
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

            nn.Dropout(
                dropout
            ),
        )

        self.response_head = nn.Linear(
            hidden_dim // 2,
            1,
        )

    def forward(
        self,

        drug_a_fp,
        drug_a_clamp,

        drug_b_fp,
        drug_b_clamp,

        cell_expr,
        resistance_expr,

        return_details=False,
    ):

        # ====================================================
        # Drug A
        # ====================================================

        if return_details:

            (
                z_a,
                gate_a,
                z_morgan_a,
                z_clamp_a,
            ) = self.drug_encoder(
                drug_a_fp,
                drug_a_clamp,
                return_gate=True,
            )

        else:

            z_a = self.drug_encoder(
                drug_a_fp,
                drug_a_clamp,
            )

        # ====================================================
        # Drug B
        # ====================================================

        if return_details:

            (
                z_b,
                gate_b,
                z_morgan_b,
                z_clamp_b,
            ) = self.drug_encoder(
                drug_b_fp,
                drug_b_clamp,
                return_gate=True,
            )

        else:

            z_b = self.drug_encoder(
                drug_b_fp,
                drug_b_clamp,
            )

        # ====================================================
        # Permutation-invariant pair
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
        # General cell state
        # ====================================================

        z_cell = self.cell_encoder(
            cell_expr
        )

        # ====================================================
        # Pair-conditioned resistance biology
        #
        # Important:
        # Resistance is conditioned on the TWO-DRUG PAIR,
        # not on Drug A or Drug B independently.
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

        hidden = self.fusion(
            fused
        )

        prediction = self.response_head(
            hidden
        )

        if return_details:

            return {
                "prediction":
                    prediction,

                "z_a":
                    z_a,

                "z_b":
                    z_b,

                "z_pair":
                    z_pair,

                "z_cell":
                    z_cell,

                "z_resistance":
                    z_resistance,

                "drug_a_clamp_gate":
                    gate_a,

                "drug_b_clamp_gate":
                    gate_b,

                "z_morgan_a":
                    z_morgan_a,

                "z_clamp_a":
                    z_clamp_a,

                "z_morgan_b":
                    z_morgan_b,

                "z_clamp_b":
                    z_clamp_b,
            }

        return prediction