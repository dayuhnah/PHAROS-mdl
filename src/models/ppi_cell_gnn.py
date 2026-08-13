import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    GCNConv,
    global_mean_pool,
    global_max_pool,
)


class PPICellGNNEncoder(nn.Module):
    """
    Cancer cell encoder using a fixed
    protein-protein interaction topology.

    Every sample uses the same graph topology.

    Sample-specific node feature:
        gene expression

    Static node feature:
        learnable gene identity embedding
    """

    def __init__(
        self,
        edge_index,
        edge_weight,
        num_genes,
        gene_embedding_dim=16,
        hidden_dim=64,
        out_dim=256,
        dropout=0.2,
    ):
        super().__init__()

        self.num_genes = num_genes
        self.dropout = dropout

        # Fixed PPI topology
        self.register_buffer(
            "edge_index",
            edge_index.long(),
        )

        self.register_buffer(
            "edge_weight",
            edge_weight.float(),
        )

        # ---------------------------------
        # Gene identity
        # ---------------------------------

        self.gene_embedding = nn.Embedding(
            num_genes,
            gene_embedding_dim,
        )

        # expression scalar + gene identity
        input_dim = (
            1
            + gene_embedding_dim
        )

        self.input_projection = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.ReLU(),
        )

        # ---------------------------------
        # PPI message passing
        # ---------------------------------

        self.conv1 = GCNConv(
            hidden_dim,
            hidden_dim,
        )

        self.norm1 = nn.LayerNorm(
            hidden_dim
        )

        self.conv2 = GCNConv(
            hidden_dim,
            hidden_dim,
        )

        self.norm2 = nn.LayerNorm(
            hidden_dim
        )

        # ---------------------------------
        # Cell-level embedding
        # ---------------------------------

        self.output_projection = nn.Sequential(
            nn.Linear(
                hidden_dim * 2,
                out_dim,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def _make_batched_graph(
        self,
        batch_size,
        device,
    ):
        """
        Repeat the same PPI graph for every
        cell line in the mini-batch.
        """

        num_edges = (
            self.edge_index
            .shape[1]
        )

        offsets = (
            torch.arange(
                batch_size,
                device=device,
            )
            * self.num_genes
        )

        offsets = offsets.view(
            batch_size,
            1,
            1,
        )

        edge_index = (
            self.edge_index
            .unsqueeze(0)
            .expand(
                batch_size,
                -1,
                -1,
            )
        )

        edge_index = (
            edge_index
            + offsets
        )

        edge_index = (
            edge_index
            .permute(
                1,
                0,
                2,
            )
            .reshape(
                2,
                batch_size
                * num_edges,
            )
        )

        edge_weight = (
            self.edge_weight
            .repeat(batch_size)
        )

        graph_batch = (
            torch.arange(
                batch_size,
                device=device,
            )
            .repeat_interleave(
                self.num_genes
            )
        )

        return (
            edge_index,
            edge_weight,
            graph_batch,
        )

    def forward(
        self,
        expression,
    ):
        """
        expression:
            [batch_size, num_genes]

        return:
            [batch_size, out_dim]
        """

        batch_size, num_genes = (
            expression.shape
        )

        if (
            num_genes
            != self.num_genes
        ):
            raise ValueError(
                "PPI expression dimension "
                f"{num_genes} does not match "
                f"graph genes "
                f"{self.num_genes}."
            )

        device = expression.device

        # ---------------------------------
        # Gene identity embeddings
        # ---------------------------------

        gene_ids = torch.arange(
            self.num_genes,
            device=device,
        )

        gene_identity = (
            self.gene_embedding(
                gene_ids
            )
        )

        gene_identity = (
            gene_identity
            .unsqueeze(0)
            .expand(
                batch_size,
                -1,
                -1,
            )
        )

        # ---------------------------------
        # Sample-specific expression
        # ---------------------------------

        expression = (
            expression
            .unsqueeze(-1)
        )

        x = torch.cat(
            [
                expression,
                gene_identity,
            ],
            dim=-1,
        )

        x = x.reshape(
            batch_size
            * self.num_genes,
            -1,
        )

        x = self.input_projection(
            x
        )

        # ---------------------------------
        # Repeat PPI topology
        # ---------------------------------

        (
            edge_index,
            edge_weight,
            graph_batch,
        ) = self._make_batched_graph(
            batch_size,
            device,
        )

        # ---------------------------------
        # GCN 1
        # ---------------------------------

        residual = x

        x = self.conv1(
            x,
            edge_index,
            edge_weight=edge_weight,
        )

        x = self.norm1(
            x + residual
        )

        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        # ---------------------------------
        # GCN 2
        # ---------------------------------

        residual = x

        x = self.conv2(
            x,
            edge_index,
            edge_weight=edge_weight,
        )

        x = self.norm2(
            x + residual
        )

        x = F.relu(x)

        # ---------------------------------
        # Pool genes -> cell
        # ---------------------------------

        mean_embedding = (
            global_mean_pool(
                x,
                graph_batch,
            )
        )

        max_embedding = (
            global_max_pool(
                x,
                graph_batch,
            )
        )

        cell_embedding = torch.cat(
            [
                mean_embedding,
                max_embedding,
            ],
            dim=-1,
        )

        return self.output_projection(
            cell_embedding
        )