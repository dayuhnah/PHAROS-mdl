import torch

from src.models.ppi_cell_gnn import (
    PPICellGNNEncoder,
)


def main():

    ppi = torch.load(
        "data/processed/pharos_ppi_graph.pt",
        map_location="cpu",
        weights_only=False,
    )

    num_genes = ppi[
        "num_genes"
    ]

    model = PPICellGNNEncoder(
        edge_index=ppi[
            "edge_index"
        ],
        edge_weight=ppi[
            "edge_weight"
        ],
        num_genes=num_genes,
        gene_embedding_dim=16,
        hidden_dim=64,
        out_dim=256,
        dropout=0.2,
    )

    # Four fake cell lines
    expression = torch.randn(
        4,
        num_genes,
    )

    model.eval()

    with torch.no_grad():

        output = model(
            expression
        )

    print(
        "Genes:",
        num_genes,
    )

    print(
        "Edges:",
        ppi[
            "edge_index"
        ].shape,
    )

    print(
        "Input:",
        expression.shape,
    )

    print(
        "Cell embedding:",
        output.shape,
    )


if __name__ == "__main__":
    main()