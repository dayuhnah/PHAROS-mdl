import torch

from src.models.hybrid_cell_encoder import (
    HybridCellEncoder,
)


def main():

    ppi = torch.load(
        "data/processed/pharos_ppi_graph.pt",
        map_location="cpu",
        weights_only=False,
    )

    num_ppi_genes = ppi[
        "num_genes"
    ]

    print(
        "PPI genes:",
        num_ppi_genes,
    )

    print(
        "PPI edges:",
        ppi["edge_index"].shape,
    )

    # ==========================================
    # Fake batch
    # ==========================================

    batch_size = 4

    full_expression = torch.randn(
        batch_size,
        19176,
    )

    ppi_expression = torch.randn(
        batch_size,
        num_ppi_genes,
    )

    # ==========================================
    # Model
    # ==========================================

    model = HybridCellEncoder(
        full_expression_dim=19176,

        ppi_edge_index=ppi[
            "edge_index"
        ],

        ppi_edge_weight=ppi[
            "edge_weight"
        ],

        num_ppi_genes=num_ppi_genes,

        expression_hidden_dim=512,
        ppi_hidden_dim=64,

        out_dim=256,
        dropout=0.2,
    )

    model.eval()

    # ==========================================
    # Forward pass
    # ==========================================

    with torch.no_grad():

        (
            z_cell,
            gate,
            z_expression,
            z_ppi,
        ) = model(
            full_expression,
            ppi_expression,
            return_gate=True,
        )

    print(
        "\n================================"
    )

    print(
        "HYBRID CELL ENCODER TEST"
    )

    print(
        "================================"
    )

    print(
        "Full expression:",
        full_expression.shape,
    )

    print(
        "PPI expression:",
        ppi_expression.shape,
    )

    print(
        "Expression embedding:",
        z_expression.shape,
    )

    print(
        "PPI embedding:",
        z_ppi.shape,
    )

    print(
        "Gate:",
        gate.shape,
    )

    print(
        "Final cell embedding:",
        z_cell.shape,
    )

    print(
        "\nMean PPI gate:",
        gate.mean().item(),
    )

    print(
        "Mean expression gate:",
        (
            1.0 - gate
        ).mean().item(),
    )


if __name__ == "__main__":
    main()