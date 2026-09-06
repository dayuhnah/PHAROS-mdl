import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityEncoder(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        output_dim=128,
        dropout=0.2,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class MultiOmicsCellEncoder(nn.Module):

    def __init__(
        self,
        expression_dim=19205,
        cnv_dim=19955,
        mutation_dim=15631,
        crispr_dim=18531,
        modality_dim=128,
        output_dim=256,
        dropout=0.2,
    ):
        super().__init__()

        self.expression_encoder = ModalityEncoder(
            expression_dim,
            output_dim=modality_dim,
            dropout=dropout,
        )

        self.cnv_encoder = ModalityEncoder(
            cnv_dim,
            output_dim=modality_dim,
            dropout=dropout,
        )

        self.mutation_encoder = ModalityEncoder(
            mutation_dim,
            output_dim=modality_dim,
            dropout=dropout,
        )

        self.crispr_encoder = ModalityEncoder(
            crispr_dim,
            output_dim=modality_dim,
            dropout=dropout,
        )

        self.attention = nn.Sequential(
            nn.Linear(modality_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.fusion = nn.Sequential(
            nn.Linear(modality_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )


    def forward(
        self,
        expression,
        cnv,
        mutation,
        crispr,
        masks,
        return_weights=False,
    ):

        z_expr = self.expression_encoder(expression)
        z_cnv = self.cnv_encoder(cnv)
        z_mut = self.mutation_encoder(mutation)
        z_crispr = self.crispr_encoder(crispr)

        modalities = torch.stack(
            [
                z_expr,
                z_cnv,
                z_mut,
                z_crispr,
            ],
            dim=1,
        )

        scores = self.attention(
            modalities
        ).squeeze(-1)

        mask_bool = masks.bool()
        mask_float = masks.float()

        scores = scores.masked_fill(
            ~mask_bool,
            -1e9,
        )

        weights = F.softmax(
            scores,
            dim=1,
        )

        # ----------------------------------------------------
        # Critical:
        # force unavailable modalities to exact zero.
        # Also handles rows where ALL modalities are missing.
        # ----------------------------------------------------

        weights = (
            weights
            * mask_float
        )

        denominator = weights.sum(
            dim=1,
            keepdim=True,
        )

        weights = torch.where(
            denominator > 0,
            weights / denominator.clamp_min(1e-12),
            torch.zeros_like(weights),
        )

        fused = (
            modalities
            * weights.unsqueeze(-1)
        ).sum(dim=1)

        cell_embedding = self.fusion(
            fused
        )

        # If no modality exists, cell contribution must
        # remain exactly zero, including Linear biases.
        any_available = (
            mask_float.sum(
                dim=1,
                keepdim=True,
            ) > 0
        ).float()

        cell_embedding = (
            cell_embedding
            * any_available
        )

        if return_weights:

            return {
                "cell_embedding":
                    cell_embedding,

                "modality_weights":
                    weights,

                "expression_embedding":
                    z_expr,

                "cnv_embedding":
                    z_cnv,

                "mutation_embedding":
                    z_mut,

                "crispr_embedding":
                    z_crispr,
            }

        return cell_embedding
