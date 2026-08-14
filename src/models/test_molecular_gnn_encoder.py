import torch

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.models.molecular_gnn_encoder import (
    MolecularGNNEncoder,
)


def main():

    # ---------------------------------
    # Fake molecule 1
    # 3 atoms, 4 directed edges
    # ---------------------------------

    graph1 = Data(
        x=torch.randn(3, 18),

        edge_index=torch.tensor(
            [
                [0, 1, 1, 2],
                [1, 0, 2, 1],
            ],
            dtype=torch.long,
        ),

        edge_attr=torch.randn(4, 6),
    )

    # ---------------------------------
    # Fake molecule 2
    # ---------------------------------

    graph2 = Data(
        x=torch.randn(4, 18),

        edge_index=torch.tensor(
            [
                [0, 1, 1, 2, 2, 3],
                [1, 0, 2, 1, 3, 2],
            ],
            dtype=torch.long,
        ),

        edge_attr=torch.randn(6, 6),
    )

    loader = DataLoader(
        [graph1, graph2],
        batch_size=2,
    )

    model = MolecularGNNEncoder(
        node_dim=18,
        edge_dim=6,
        hidden_dim=128,
        out_dim=256,
        heads=4,
        dropout=0.2,
    )

    for batch in loader:

        output = model(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            batch=batch.batch,
        )

        print(
            "Node features:",
            batch.x.shape,
        )

        print(
            "Edge index:",
            batch.edge_index.shape,
        )

        print(
            "Edge features:",
            batch.edge_attr.shape,
        )

        print(
            "Drug embedding:",
            output.shape,
        )


if __name__ == "__main__":
    main()