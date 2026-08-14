import torch
from torch.utils.data import DataLoader

from src.data.dataset import PharosDepMapDataset
from src.data.graph_collate import pharos_graph_collate
from src.models.pharos_rx_dual_hybrid import PharosRXDualHybridModel


def main():

    # ==========================================
    # Dataset
    # ==========================================

    dataset = PharosDepMapDataset(
        max_rows=10,
        use_drug_graphs=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=pharos_graph_collate,
    )

    batch = next(
        iter(loader)
    )

    # ==========================================
    # Load PPI graph
    # ==========================================

    ppi = torch.load(
        "data/processed/pharos_ppi_graph.pt",
        map_location="cpu",
        weights_only=False,
    )

    # ==========================================
    # Dimensions
    # ==========================================

    drug_fp_dim = batch[
        "drug_fp"
    ].shape[1]

    cell_dim = batch[
        "cell_expr"
    ].shape[1]

    resistance_dim = batch[
        "resistance_expr"
    ].shape[1]

    num_ppi_genes = batch[
        "ppi_expression"
    ].shape[1]

    node_dim = batch[
        "drug_graph"
    ].x.shape[1]

    edge_dim = batch[
        "drug_graph"
    ].edge_attr.shape[1]

    # ==========================================
    # Model
    # ==========================================

    model = PharosRXDualHybridModel(
        drug_fp_dim=drug_fp_dim,
        cell_dim=cell_dim,
        resistance_dim=resistance_dim,

        ppi_edge_index=ppi[
            "edge_index"
        ],

        ppi_edge_weight=ppi[
            "edge_weight"
        ],

        num_ppi_genes=num_ppi_genes,

        node_dim=node_dim,
        edge_dim=edge_dim,

        hidden_dim=512,

        drug_gnn_hidden_dim=128,
        ppi_hidden_dim=64,

        drug_out_dim=256,
        cell_out_dim=256,
        resistance_out_dim=64,

        dropout=0.2,
    )

    model.eval()

    # ==========================================
    # Forward
    # ==========================================

    with torch.no_grad():

        output = model(
            drug_graph=batch[
                "drug_graph"
            ],

            drug_fp=batch[
                "drug_fp"
            ],

            cell_expr=batch[
                "cell_expr"
            ],

            ppi_expression=batch[
                "ppi_expression"
            ],

            resistance_expr=batch[
                "resistance_expr"
            ],

            return_gates=True,
        )

    # ==========================================
    # Print results
    # ==========================================

    print(
        "\n================================"
    )

    print(
        "PHAROS V2B DUAL-HYBRID TEST"
    )

    print(
        "================================"
    )

    print(
        "Prediction:",
        output[
            "prediction"
        ].shape,
    )

    print(
        "Drug:",
        output[
            "z_drug"
        ].shape,
    )

    print(
        "Graph:",
        output[
            "z_graph"
        ].shape,
    )

    print(
        "Morgan:",
        output[
            "z_fp"
        ].shape,
    )

    print(
        "Cell:",
        output[
            "z_cell"
        ].shape,
    )

    print(
        "Expression:",
        output[
            "z_expression"
        ].shape,
    )

    print(
        "PPI:",
        output[
            "z_ppi"
        ].shape,
    )

    print(
        "Resistance:",
        output[
            "z_resistance"
        ].shape,
    )

    print(
        "Drug gate:",
        output[
            "drug_gate"
        ].shape,
    )

    print(
        "Cell gate:",
        output[
            "cell_gate"
        ].shape,
    )

    print(
        "\nMean GNN drug gate:",
        output[
            "drug_gate"
        ].mean().item(),
    )

    print(
        "Mean Morgan gate:",
        (
            1.0
            - output[
                "drug_gate"
            ]
        ).mean().item(),
    )

    print(
        "\nMean PPI cell gate:",
        output[
            "cell_gate"
        ].mean().item(),
    )

    print(
        "Mean expression gate:",
        (
            1.0
            - output[
                "cell_gate"
            ]
        ).mean().item(),
    )


if __name__ == "__main__":
    main()