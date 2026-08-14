import torch

from torch_geometric.data import Batch

from src.data.dataset import (
    PharosDepMapDataset,
)

from src.models.pharos_rx_hybrid import (
    PharosRXHybridModel,
)


def main():

    print(
        "Loading PHAROS graph dataset..."
    )

    dataset = PharosDepMapDataset(
        max_rows=10,
        use_drug_graphs=True,
    )

    samples = [
        dataset[i]
        for i in range(4)
    ]

    # ==========================================
    # Build graph batch
    # ==========================================

    drug_graph = Batch.from_data_list(
        [
            sample["drug_graph"]
            for sample in samples
        ]
    )

    drug_fp = torch.stack(
        [
            sample["drug_fp"]
            for sample in samples
        ]
    )

    cell_expr = torch.stack(
        [
            sample["cell_expr"]
            for sample in samples
        ]
    )

    resistance_expr = torch.stack(
        [
            sample["resistance_expr"]
            for sample in samples
        ]
    )

    # ==========================================
    # Dimensions
    # ==========================================

    drug_fp_dim = (
        drug_fp.shape[1]
    )

    cell_dim = (
        cell_expr.shape[1]
    )

    resistance_dim = (
        resistance_expr.shape[1]
    )

    node_dim = (
        drug_graph.x.shape[1]
    )

    edge_dim = (
        drug_graph.edge_attr.shape[1]
    )

    # ==========================================
    # Model
    # ==========================================

    model = PharosRXHybridModel(
        drug_fp_dim=drug_fp_dim,
        cell_dim=cell_dim,
        resistance_dim=resistance_dim,

        node_dim=node_dim,
        edge_dim=edge_dim,

        hidden_dim=512,

        drug_gnn_hidden_dim=128,

        drug_out_dim=256,
        cell_out_dim=256,
        resistance_out_dim=64,

        dropout=0.2,
    )

    model.eval()

    with torch.no_grad():

        outputs = model(
            drug_graph=drug_graph,
            drug_fp=drug_fp,
            cell_expr=cell_expr,
            resistance_expr=resistance_expr,
            return_gate=True,
        )

    print(
        "\n================================"
    )

    print(
        "PHAROS-RX-HYBRID TEST"
    )

    print(
        "================================"
    )

    print(
        "Prediction:",
        outputs["prediction"].shape,
    )

    print(
        "Drug:",
        outputs["z_drug"].shape,
    )

    print(
        "Graph:",
        outputs["z_graph"].shape,
    )

    print(
        "Morgan:",
        outputs["z_fp"].shape,
    )

    print(
        "Cell:",
        outputs["z_cell"].shape,
    )

    print(
        "Resistance:",
        outputs[
            "z_resistance"
        ].shape,
    )

    print(
        "Gate:",
        outputs["gate"].shape,
    )

    print(
        "\nMean GNN gate:",
        outputs[
            "gate"
        ].mean().item(),
    )

    print(
        "Mean Morgan gate:",
        (
            1.0
            - outputs["gate"]
        ).mean().item(),
    )


if __name__ == "__main__":
    main()