import torch
import torch.nn as nn

from src.models.pharos_combo_clamp_hybrid import (
    MorganCLAMPHybridEncoder,
)

from src.models.resistance_film import (
    ResistanceFiLM,
)


class ScDrugActPairResistanceEncoder(nn.Module):
    """
    Encode drug-specific scDrugAct resistance-expression profiles.

    For each drug/cell pair:
        r_(d,c) = x_c * m_d

    where x_c is DepMap expression over the usable scDrugAct genes
    and m_d is the binary scDrugAct resistance mask for that drug.
    """

    def __init__(
        self,
        input_dim=2399,
        per_drug_hidden_dim=512,
        per_drug_out_dim=128,
        pair_out_dim=64,
        dropout=0.2,
    ):
        super().__init__()

        self.shared_drug_resistance_encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(
                input_dim,
                per_drug_hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                per_drug_hidden_dim,
                per_drug_out_dim,
            ),
            nn.LayerNorm(
                per_drug_out_dim
            ),
            nn.GELU(),
        )

        pair_input_dim = (
            per_drug_out_dim * 3
            + 2
        )

        self.pair_encoder = nn.Sequential(
            nn.Linear(
                pair_input_dim,
                256,
            ),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                256,
                pair_out_dim,
            ),
            nn.LayerNorm(
                pair_out_dim
            ),
            nn.GELU(),
        )

    def forward(
        self,
        drug_a_scdrugact_expr,
        drug_b_scdrugact_expr,
        drug_a_available,
        drug_b_available,
        return_details=False,
    ):

        if drug_a_available.dim() == 1:
            drug_a_available = (
                drug_a_available
                .unsqueeze(-1)
            )

        if drug_b_available.dim() == 1:
            drug_b_available = (
                drug_b_available
                .unsqueeze(-1)
            )

        drug_a_available = (
            drug_a_available.to(
                dtype=drug_a_scdrugact_expr.dtype,
                device=drug_a_scdrugact_expr.device,
            )
        )

        drug_b_available = (
            drug_b_available.to(
                dtype=drug_b_scdrugact_expr.dtype,
                device=drug_b_scdrugact_expr.device,
            )
        )

        z_sc_a = (
            self.shared_drug_resistance_encoder(
                drug_a_scdrugact_expr
            )
        )

        z_sc_b = (
            self.shared_drug_resistance_encoder(
                drug_b_scdrugact_expr
            )
        )

        # Important: Linear layers have biases, so an all-zero
        # unmatched input could otherwise become non-zero.
        z_sc_a = (
            z_sc_a
            * drug_a_available
        )

        z_sc_b = (
            z_sc_b
            * drug_b_available
        )

        z_sum = (
            z_sc_a
            + z_sc_b
        )

        z_absdiff = torch.abs(
            z_sc_a
            - z_sc_b
        )

        z_product = (
            z_sc_a
            * z_sc_b
        )

        availability_sum = (
            drug_a_available
            + drug_b_available
        )

        availability_both = (
            drug_a_available
            * drug_b_available
        )

        pair_features = torch.cat(
            [
                z_sum,
                z_absdiff,
                z_product,
                availability_sum,
                availability_both,
            ],
            dim=-1,
        )

        z_sc_pair = (
            self.pair_encoder(
                pair_features
            )
        )

        pair_available = torch.clamp(
            availability_sum,
            min=0.0,
            max=1.0,
        )

        # If neither drug has database evidence, force the
        # database-derived branch to exactly zero.
        z_sc_pair = (
            z_sc_pair
            * pair_available
        )

        if return_details:
            return {
                "z_scdrugact_a":
                    z_sc_a,

                "z_scdrugact_b":
                    z_sc_b,

                "z_scdrugact_pair":
                    z_sc_pair,

                "scdrugact_pair_available":
                    pair_available,

                "scdrugact_availability_sum":
                    availability_sum,

                "scdrugact_both_available":
                    availability_both,
            }

        return z_sc_pair


class PharosComboCLAMPResistanceV2(nn.Module):
    """
    PHAROS-Combo Resistance V2.

    Drug representation:
        Morgan + frozen cached CLAMP -> learned hybrid drug embedding

    Pair representation:
        zA + zB
        |zA - zB|
        zA * zB
        -> symmetric pair embedding

    Cell representation:
        19,176 general DepMap expression genes -> cell embedding

    Core resistance:
        29 curated resistance genes
        -> pair-conditioned ResistanceFiLM

    Database resistance:
        Drug A scDrugAct-masked DepMap expression
        Drug B scDrugAct-masked DepMap expression
        -> shared per-drug encoder
        -> symmetric pair resistance encoder

    Final resistance:
        core resistance + scDrugAct resistance
        -> fused resistance embedding

    Prediction:
        pair + cell + fused resistance -> ZIP synergy
    """

    def __init__(
        self,
        fingerprint_dim=2048,
        clamp_dim=768,
        cell_dim=19176,
        resistance_dim=29,
        scdrugact_dim=2399,

        drug_out_dim=256,
        pair_out_dim=256,
        cell_out_dim=256,

        core_resistance_out_dim=64,
        scdrugact_out_dim=64,
        fused_resistance_out_dim=64,

        scdrugact_hidden_dim=512,
        scdrugact_per_drug_dim=128,

        hidden_dim=512,
        dropout=0.2,
    ):
        super().__init__()

        self.drug_encoder = (
            MorganCLAMPHybridEncoder(
                fingerprint_dim=fingerprint_dim,
                clamp_dim=clamp_dim,
                out_dim=drug_out_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
        )

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

        self.core_resistance_film = (
            ResistanceFiLM(
                drug_dim=pair_out_dim,
                resistance_dim=resistance_dim,
                hidden_dim=128,
                output_dim=core_resistance_out_dim,
                dropout=dropout,
            )
        )

        # Alias retained for compatibility with existing code.
        self.resistance_film = (
            self.core_resistance_film
        )

        self.scdrugact_resistance_encoder = (
            ScDrugActPairResistanceEncoder(
                input_dim=scdrugact_dim,
                per_drug_hidden_dim=scdrugact_hidden_dim,
                per_drug_out_dim=scdrugact_per_drug_dim,
                pair_out_dim=scdrugact_out_dim,
                dropout=dropout,
            )
        )

        resistance_fusion_input_dim = (
            core_resistance_out_dim
            + scdrugact_out_dim
        )

        self.resistance_fusion = nn.Sequential(
            nn.Linear(
                resistance_fusion_input_dim,
                128,
            ),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                128,
                fused_resistance_out_dim,
            ),
            nn.LayerNorm(
                fused_resistance_out_dim
            ),
            nn.GELU(),
        )

        fusion_dim = (
            pair_out_dim
            + cell_out_dim
            + fused_resistance_out_dim
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
        drug_a_scdrugact_expr,
        drug_b_scdrugact_expr,
        drug_a_scdrugact_available,
        drug_b_scdrugact_available,
        return_details=False,
    ):

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

        z_pair = (
            self.pair_encoder(
                pair_features
            )
        )

        z_cell = (
            self.cell_encoder(
                cell_expr
            )
        )

        z_core_resistance = (
            self.core_resistance_film(
                resistance_features=resistance_expr,
                drug_embedding=z_pair,
            )
        )

        if return_details:
            sc_details = (
                self.scdrugact_resistance_encoder(
                    drug_a_scdrugact_expr,
                    drug_b_scdrugact_expr,
                    drug_a_scdrugact_available,
                    drug_b_scdrugact_available,
                    return_details=True,
                )
            )

            z_scdrugact_resistance = (
                sc_details[
                    "z_scdrugact_pair"
                ]
            )
        else:
            z_scdrugact_resistance = (
                self.scdrugact_resistance_encoder(
                    drug_a_scdrugact_expr,
                    drug_b_scdrugact_expr,
                    drug_a_scdrugact_available,
                    drug_b_scdrugact_available,
                )
            )

        resistance_features = torch.cat(
            [
                z_core_resistance,
                z_scdrugact_resistance,
            ],
            dim=-1,
        )

        z_resistance = (
            self.resistance_fusion(
                resistance_features
            )
        )

        fused = torch.cat(
            [
                z_pair,
                z_cell,
                z_resistance,
            ],
            dim=-1,
        )

        hidden = (
            self.fusion(
                fused
            )
        )

        prediction = (
            self.response_head(
                hidden
            )
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

                "z_core_resistance":
                    z_core_resistance,

                "z_scdrugact_resistance":
                    z_scdrugact_resistance,

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

                "z_scdrugact_a":
                    sc_details[
                        "z_scdrugact_a"
                    ],

                "z_scdrugact_b":
                    sc_details[
                        "z_scdrugact_b"
                    ],

                "scdrugact_pair_available":
                    sc_details[
                        "scdrugact_pair_available"
                    ],

                "scdrugact_availability_sum":
                    sc_details[
                        "scdrugact_availability_sum"
                    ],

                "scdrugact_both_available":
                    sc_details[
                        "scdrugact_both_available"
                    ],
            }

        return prediction
