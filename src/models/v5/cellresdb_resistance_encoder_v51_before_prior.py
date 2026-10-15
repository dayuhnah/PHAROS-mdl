from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class CellResDBResistanceEncoderV51(nn.Module):
    """
    Evidence-aware CellResDB encoder.

    Improvements over V5:
      1. Query uses drug-pair + cell embedding.
      2. Adds a learnable NULL/no-evidence token.
      3. Returns NULL attention for interpretability.
    """

    def __init__(
        self,
        cache_path,
        pair_dim=256,
        cell_dim=256,
        hidden_dim=128,
        out_dim=64,
        num_heads=4,
        dropout=0.1,
    ):
        super().__init__()

        cache = torch.load(
            Path(cache_path),
            map_location="cpu",
            weights_only=True,
        )

        features = cache["features"].float()
        pathway_mask = cache["pathway_mask"].bool()
        cohort_ids = cache["cohort_ids"].long()
        cell_type_ids = cache["cell_type_ids"].long()
        pathway_ids = cache["pathway_ids"].long()

        T, P, feature_dim = features.shape

        self.register_buffer(
            "bank_features",
            features,
        )

        self.register_buffer(
            "pathway_mask",
            pathway_mask,
        )

        self.register_buffer(
            "cohort_ids",
            cohort_ids,
        )

        self.register_buffer(
            "cell_type_ids",
            cell_type_ids,
        )

        self.register_buffer(
            "pathway_ids",
            pathway_ids,
        )

        self.num_real_tokens = T
        self.num_pathways = P
        self.feature_dim = feature_dim

        num_cohorts = (
            int(cohort_ids.max().item()) + 1
        )

        num_cell_types = (
            int(cell_type_ids.max().item()) + 1
        )

        # ====================================================
        # PATHWAY ENCODING
        # ====================================================

        self.feature_encoder = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(
                feature_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
        )

        self.pathway_embedding = nn.Embedding(
            P,
            hidden_dim,
        )

        self.pathway_attention = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim // 2,
            ),
            nn.Tanh(),
            nn.Linear(
                hidden_dim // 2,
                1,
            ),
        )

        # ====================================================
        # BIOLOGICAL TOKEN IDENTITY
        # ====================================================

        self.cohort_embedding = nn.Embedding(
            num_cohorts,
            hidden_dim,
        )

        self.cell_type_embedding = nn.Embedding(
            num_cell_types,
            hidden_dim,
        )

        self.token_norm = nn.LayerNorm(
            hidden_dim
        )

        # ====================================================
        # NULL / NO-EVIDENCE TOKEN
        # ====================================================

        self.null_token = nn.Parameter(
            torch.zeros(
                1,
                hidden_dim,
            )
        )

        nn.init.normal_(
            self.null_token,
            mean=0.0,
            std=0.02,
        )

        # ====================================================
        # PAIR + CELL CONDITIONED QUERY
        # ====================================================

        self.query_encoder = nn.Sequential(
            nn.Linear(
                pair_dim + cell_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            # Keep attention weights normalized because the
            # NULL-token probability is used as evidence relevance.
            dropout=0.0,
            batch_first=True,
        )

        self.cross_norm = nn.LayerNorm(
            hidden_dim
        )

        # ====================================================
        # OUTPUT
        # ====================================================

        self.output_head = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dim,
                out_dim,
            ),
        )

    def build_token_bank(self):

        x = self.feature_encoder(
            self.bank_features
        )

        pathway_emb = self.pathway_embedding(
            self.pathway_ids
        )

        x = (
            x
            + pathway_emb.unsqueeze(0)
        )

        logits = (
            self.pathway_attention(x)
            .squeeze(-1)
        )

        logits = logits.masked_fill(
            ~self.pathway_mask,
            float("-inf"),
        )

        pathway_weights = F.softmax(
            logits,
            dim=-1,
        )

        token_bank = torch.sum(
            x
            * pathway_weights.unsqueeze(-1),
            dim=1,
        )

        token_bank = (
            token_bank
            + self.cohort_embedding(
                self.cohort_ids
            )
            + self.cell_type_embedding(
                self.cell_type_ids
            )
        )

        token_bank = self.token_norm(
            token_bank
        )

        # Append NULL/no-evidence token.
        token_bank = torch.cat(
            [
                token_bank,
                self.null_token,
            ],
            dim=0,
        )

        return (
            token_bank,
            pathway_weights,
        )

    def forward(
        self,
        drug_pair_embedding,
        cell_embedding,
        return_attention=False,
    ):

        if drug_pair_embedding.ndim != 2:
            raise ValueError(
                "drug_pair_embedding must be [B,D]"
            )

        if cell_embedding.ndim != 2:
            raise ValueError(
                "cell_embedding must be [B,D]"
            )

        B = drug_pair_embedding.size(0)

        token_bank, pathway_weights = (
            self.build_token_bank()
        )

        # [21,H] → [B,21,H]
        keys = (
            token_bank
            .unsqueeze(0)
            .expand(
                B,
                -1,
                -1,
            )
        )

        query_input = torch.cat(
            [
                drug_pair_embedding,
                cell_embedding,
            ],
            dim=-1,
        )

        query = (
            self.query_encoder(
                query_input
            )
            .unsqueeze(1)
        )

        context, token_attention = (
            self.cross_attention(
                query=query,
                key=keys,
                value=keys,
                need_weights=True,
                average_attn_weights=False,
            )
        )

        context = self.cross_norm(
            context + query
        )

        context = context.squeeze(1)

        embedding = self.output_head(
            context
        )

        if not return_attention:
            return embedding

        token_attention = (
            token_attention
            .mean(dim=1)
            .squeeze(1)
        )

        # Last token is NULL.
        real_token_attention = (
            token_attention[
                :,
                :-1
            ]
        )

        null_attention = (
            token_attention[
                :,
                -1:
            ]
        )

        return {
            "embedding":
                embedding,

            "token_attention":
                real_token_attention,

            "null_attention":
                null_attention,

            "pathway_attention":
                pathway_weights,
        }
