import torch

from torch_geometric.data import Batch

from src.data.dataset import (
    PharosDepMapDataset,
)

from src.models.hybrid_drug_encoder import (
    HybridDrugEncoder,
)


def main():

    print(
        "Loading graph dataset..."
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
    # Molecular graphs
    # ==========================================

    drug_graph = Batch.from_data_list(
        [
            sample["drug_graph"]
            for sample in samples
        ]
    )

    # ==========================================
    # Morgan fingerprints
    # ==========================================

    drug_fp = torch.stack(
        [
            sample["drug_fp"]
            for sample in samples
        ]
    )

    # ==========================================
    # Determine dimensions
    # ==========================================

    node_dim = (
        drug_graph.x.shape[1]
    )

    edge_dim = (
        drug_graph.edge_attr.shape[1]
    )

    fingerprint_dim = (
        drug_fp.shape[1]
    )

    # ==========================================
    # Model
    # ==========================================

    model = HybridDrugEncoder(
        node_dim=node_dim,
        edge_dim=edge_dim,
        fingerprint_dim=fingerprint_dim,
        gnn_hidden_dim=128,
        out_dim=256,
        dropout=0.2,
    )

    model.eval()

    # ==========================================
    # Forward pass
    # ==========================================

    with torch.no_grad():

        (
            z_drug,
            gate,
            z_graph,
            z_fp,
        ) = model(
            drug_graph,
            drug_fp,
            return_gate=True,
        )

    print(
        "\n================================"
    )

    print(
        "HYBRID DRUG ENCODER TEST"
    )

    print(
        "================================"
    )

    print(
        "Drug graph embedding:",
        z_graph.shape,
    )

    print(
        "Morgan embedding:",
        z_fp.shape,
    )

    print(
        "Gate:",
        gate.shape,
    )

    print(
        "Final drug embedding:",
        z_drug.shape,
    )

    print(
        "\nMean GNN gate:",
        gate.mean().item(),
    )

    print(
        "Mean Morgan gate:",
        (
            1.0 - gate
        ).mean().item(),
    )


if __name__ == "__main__":

    main()