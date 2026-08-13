from torch.utils.data import DataLoader

from src.data.dataset import (
    PharosDepMapDataset,
)

from src.data.graph_collate import (
    pharos_graph_collate,
)


def main():

    dataset = PharosDepMapDataset(
        max_rows=100,
        use_drug_graphs=True,
    )

    print("\nDataset length:")
    print(len(dataset))

    # ---------------------------------
    # Inspect one sample
    # ---------------------------------

    sample = dataset[0]

    graph = sample["drug_graph"]

    print("\nSingle sample:")
    print(
        "Broad ID:",
        sample["broad_id"],
    )

    print(
        "Drug FP:",
        sample["drug_fp"].shape,
    )

    print(
        "Cell:",
        sample["cell_expr"].shape,
    )

    print(
        "Resistance:",
        sample["resistance_expr"].shape,
    )

    print(
        "Graph x:",
        graph.x.shape,
    )

    print(
        "Graph edge_index:",
        graph.edge_index.shape,
    )

    print(
        "Graph edge_attr:",
        graph.edge_attr.shape,
    )

    # ---------------------------------
    # Batch several real molecules
    # ---------------------------------

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=pharos_graph_collate,
    )

    batch = next(
        iter(loader)
    )

    drug_graph = batch[
        "drug_graph"
    ]

    print("\nBatched data:")

    print(
        "Drug FP:",
        batch["drug_fp"].shape,
    )

    print(
        "Cell:",
        batch["cell_expr"].shape,
    )

    print(
        "Resistance:",
        batch[
            "resistance_expr"
        ].shape,
    )

    print(
        "Labels:",
        batch["label"].shape,
    )

    print(
        "Graph nodes:",
        drug_graph.x.shape,
    )

    print(
        "Graph edges:",
        drug_graph.edge_index.shape,
    )

    print(
        "Graph edge features:",
        drug_graph.edge_attr.shape,
    )

    print(
        "Graph batch vector:",
        drug_graph.batch.shape,
    )

    print(
        "Number of molecules:",
        drug_graph.num_graphs,
    )

    print(
        "Broad IDs:",
        batch["broad_id"],
    )


if __name__ == "__main__":
    main()