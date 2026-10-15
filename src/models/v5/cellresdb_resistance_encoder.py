from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class CellResDBResistanceEncoder(nn.Module):
    """
    Patient-derived CellResDB resistance encoder.

    Input cache:
        pathway_features : [T, P, F]
            T = cohort × cell-type tokens (20)
            P = Hallmark pathways (50)
            F = pathway/evidence features (10)

    Forward input:
        drug_pair_embedding : [B, drug_pair_dim]

    Output:
        resistance_embedding : [B, out_dim]

    Architecture:
        pathway features
            ↓
        pathway projection + identity embedding
            ↓
        interpretable pathway attention
            ↓
        20 cohort/cell-type tokens
            ↓
        cohort + cell-type embeddings
            ↓
        drug-pair cross-attention
            ↓
        CellResDB resistance representation
    """

    def __init__(
        self,
        cache_path,
        drug_pair_dim=256,
        hidden_dim=128,
        out_dim=64,
        num_heads=4,
        dropout=0.1,
    ):
        super().__init__()

        cache_path = Path(cache_path)

        cache = torch.load(
            cache_path,
            map_location="cpu",
            weights_only=True,
        )

        features = cache["features"].float()
        pathway_mask = cache["pathway_mask"].bool()
        cohort_ids = cache["cohort_ids"].long()
        cell_type_ids = cache["cell_type_ids"].long()
        pathway_ids = cache["pathway_ids"].long()

        if features.ndim != 3:
            raise ValueError(
                f"Expected [T,P,F] features, got {features.shape}"
            )

        T, P, feature_dim = features.shape

        if pathway_mask.shape != (T, P):
            raise ValueError(
                "pathway_mask shape does not match features"
            )

        # --------------------------------------------------
        # Static biological evidence bank
        # --------------------------------------------------

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

        self.num_tokens = T
        self.num_pathways = P
        self.feature_dim = feature_dim

        num_cohorts = int(cohort_ids.max().item()) + 1
        num_cell_types = int(cell_type_ids.max().item()) + 1

        # --------------------------------------------------
        # Pathway-level encoding
        # --------------------------------------------------

        self.feature_encoder = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.pathway_embedding = nn.Embedding(
            P,
            hidden_dim,
        )

        # Learns which pathways matter within each
        # cohort × cell-type token.
        self.pathway_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # --------------------------------------------------
        # Cohort / cell-type identity
        # --------------------------------------------------

        self.cohort_embedding = nn.Embedding(
            num_cohorts,
            hidden_dim,
        )

        self.cell_type_embedding = nn.Embedding(
            num_cell_types,
            hidden_dim,
        )

        self.token_norm = nn.LayerNorm(hidden_dim)

        # --------------------------------------------------
        # Drug-conditioned retrieval
        # --------------------------------------------------

        self.drug_query = nn.Sequential(
            nn.Linear(drug_pair_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.cross_norm = nn.LayerNorm(hidden_dim)

        # --------------------------------------------------
        # Final resistance representation
        # --------------------------------------------------

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def build_token_bank(self):
        """
        Convert the fixed [T,P,F] biological evidence
        into T cohort×cell-type embeddings.

        Returns
        -------
        token_bank:
            [T, hidden_dim]

        pathway_weights:
            [T, P]
        """

        x = self.feature_encoder(
            self.bank_features
        )

        # Add Hallmark pathway identity.
        p_emb = self.pathway_embedding(
            self.pathway_ids
        )

        x = x + p_emb.unsqueeze(0)

        # ----------------------------------------------
        # Pathway attention
        # ----------------------------------------------

        logits = self.pathway_attention(
            x
        ).squeeze(-1)

        logits = logits.masked_fill(
            ~self.pathway_mask,
            float("-inf"),
        )

        pathway_weights = F.softmax(
            logits,
            dim=-1,
        )

        token_bank = torch.sum(
            x * pathway_weights.unsqueeze(-1),
            dim=1,
        )

        # ----------------------------------------------
        # Biological token identity
        # ----------------------------------------------

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

        return token_bank, pathway_weights

    def forward(
        self,
        drug_pair_embedding,
        return_attention=False,
    ):
        """
        Parameters
        ----------
        drug_pair_embedding:
            [B, drug_pair_dim]

        return_attention:
            If True, also return interpretability
            information.

        Returns
        -------
        resistance_embedding:
            [B, out_dim]
        """

        if drug_pair_embedding.ndim != 2:
            raise ValueError(
                "drug_pair_embedding must be [B,D]"
            )

        batch_size = drug_pair_embedding.size(0)

        token_bank, pathway_weights = (
            self.build_token_bank()
        )

        # [T,H] → [B,T,H]
        keys = token_bank.unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )

        # Drug pair acts as query.
        query = self.drug_query(
            drug_pair_embedding
        ).unsqueeze(1)

        context, token_attention = (
            self.cross_attention(
                query=query,
                key=keys,
                value=keys,
                need_weights=True,
                average_attn_weights=False,
            )
        )

        # Residual connection from drug query.
        context = self.cross_norm(
            context + query
        )

        context = context.squeeze(1)

        resistance_embedding = (
            self.output_head(context)
        )

        if return_attention:

            # MHA:
            # [B, heads, 1, T]
            token_attention_mean = (
                token_attention
                .mean(dim=1)
                .squeeze(1)
            )

            return {
                "embedding": resistance_embedding,

                # Which patient-derived
                # cohort/cell-type evidence mattered
                # for this drug pair?
                "token_attention":
                    token_attention_mean,

                # Which Hallmark pathways mattered
                # inside each biological token?
                "pathway_attention":
                    pathway_weights,
            }

        return resistance_embedding
