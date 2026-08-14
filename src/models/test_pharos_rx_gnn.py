import torch
from torch.utils.data import DataLoader

from src.data.dataset import PharosDepMapDataset
from src.data.graph_collate import pharos_graph_collate
from src.models.pharos_rx_gnn import PharosRXGNNModel


def main():

    dataset = PharosDepMapDataset(
        max_rows=100,
        use_drug_graphs=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=pharos_graph_collate,
    )

    batch = next(iter(loader))

    drug_graph = batch["drug_graph"]
    cell_expr = batch["cell_expr"]
    resistance_expr = batch[
        "resistance_expr"
    ]

    print("\nInput:")
    print(
        "Molecules:",
        drug_graph.num_graphs,
    )

    print(
        "Atoms:",
        drug_graph.x.shape,
    )

    print(
        "Edges:",
        drug_graph.edge_index.shape,
    )

    print(
        "Cell:",
        cell_expr.shape,
    )

    print(
        "Resistance:",
        resistance_expr.shape,
    )

    model = PharosRXGNNModel(
        cell_dim=cell_expr.shape[1],
        resistance_dim=resistance_expr.shape[1],
        node_dim=drug_graph.x.shape[1],
        edge_dim=drug_graph.edge_attr.shape[1],
        hidden_dim=512,
        drug_gnn_hidden_dim=128,
        drug_out_dim=256,
        cell_out_dim=256,
        resistance_out_dim=64,
        gnn_heads=4,
        dropout=0.2,
    )

    model.eval()

    with torch.no_grad():

        prediction = model(
            drug_graph=drug_graph,
            cell_expr=cell_expr,
            resistance_expr=resistance_expr,
        )

    print("\nOutput:")

    print(
        "Prediction:",
        prediction.shape,
    )

    print(
        prediction
    )


if __name__ == "__main__":
    main()