import torch
import torch.nn as nn


class PartialCLAMPEncoder(nn.Module):
    """
    Partially fine-tuned CLAMP compound encoder.

    Input:
        Cached 2048-D hidden state from frozen CLAMP layers.

    Trainable:
        Original pretrained CLAMP final layer:
            2048 -> 768

    The weights are initialized from the actual pretrained
    CLAMP checkpoint.
    """

    def __init__(
        self,
        pretrained_weight,
        pretrained_bias,
        hidden_dim=2048,
        clamp_dim=768,
    ):
        super().__init__()

        self.linear_output = nn.Linear(
            hidden_dim,
            clamp_dim,
        )

        # Load actual pretrained CLAMP weights
        with torch.no_grad():

            self.linear_output.weight.copy_(
                pretrained_weight
            )

            self.linear_output.bias.copy_(
                pretrained_bias
            )

    def forward(
        self,
        clamp_hidden,
    ):

        return self.linear_output(
            clamp_hidden
        )


class PharosComboCLAMPFineTuned(nn.Module):
    """
    PHAROS-Combo with partially fine-tuned CLAMP.

    CLAMP:
        frozen 8192 -> 4096 -> 2048
        cached 2048-D representation

        pretrained 2048 -> 768
        FINE-TUNED with low LR

    Drug adaptation:
        768 -> 512 -> 256

    Pair representation:
        A+B
        |A-B|
        A*B

    Cell:
        full gene-expression MLP

    Molecular GNN: OFF
    PPI-GNN: OFF
    Resistance FiLM: OFF
    """

    def __init__(
        self,

        pretrained_weight,
        pretrained_bias,

        clamp_hidden_dim=2048,
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
        # Actual pretrained CLAMP layer
        # ====================================================

        self.clamp_encoder = PartialCLAMPEncoder(
            pretrained_weight=pretrained_weight,
            pretrained_bias=pretrained_bias,
            hidden_dim=clamp_hidden_dim,
            clamp_dim=clamp_dim,
        )

        # ====================================================
        # Task-specific drug projection
        # ====================================================

        self.drug_projection = nn.Sequential(

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
        # Cell-expression encoder
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
        # ZIP prediction head
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

    def encode_drug(
        self,
        hidden_features,
    ):

        z_clamp = self.clamp_encoder(
            hidden_features
        )

        z_drug = self.drug_projection(
            z_clamp
        )

        return (
            z_drug,
            z_clamp,
        )

    def forward(
        self,

        drug_a_hidden,
        drug_b_hidden,
        cell_expr,

        return_details=False,
    ):

        # ====================================================
        # Fine-tuned pretrained drug embeddings
        # ====================================================

        (
            z_a,
            z_clamp_a,
        ) = self.encode_drug(
            drug_a_hidden
        )

        (
            z_b,
            z_clamp_b,
        ) = self.encode_drug(
            drug_b_hidden
        )

        # ====================================================
        # Permutation-invariant pair representation
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
        # Cell
        # ====================================================

        z_cell = self.cell_encoder(
            cell_expr
        )

        # ====================================================
        # Fusion
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

                "z_clamp_a": z_clamp_a,
                "z_clamp_b": z_clamp_b,

                "z_a": z_a,
                "z_b": z_b,

                "z_pair": z_pair,
                "z_cell": z_cell,
            }

        return prediction