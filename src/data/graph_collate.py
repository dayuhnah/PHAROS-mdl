import torch

from torch_geometric.data import Batch


def pharos_graph_collate(batch):
    """
    Collate PHAROS samples containing variable-sized
    PyTorch Geometric molecular graphs.
    """

    # ---------------------------------
    # Molecular graphs
    # ---------------------------------

    drug_graphs = [
        item["drug_graph"]
        for item in batch
    ]

    drug_graph_batch = Batch.from_data_list(
        drug_graphs
    )

    # ---------------------------------
    # Regular fixed-size tensors
    # ---------------------------------

    drug_fp = torch.stack(
        [
            item["drug_fp"]
            for item in batch
        ]
    )

    cell_expr = torch.stack(
        [
            item["cell_expr"]
            for item in batch
        ]
    )

    # ---------------------------------
    # PPI-aligned expression
    # ---------------------------------

    ppi_expression = torch.stack(
        [
            item["ppi_expression"]
            for item in batch
        ]
    )

    resistance_expr = torch.stack(
        [
            item["resistance_expr"]
            for item in batch
        ]
    )

    labels = torch.stack(
        [
            item["label"]
            for item in batch
        ]
    )

    cell_embedding = torch.stack(
        [
            item["cell_embedding"]
            for item in batch
        ]
    )

    drug_embedding = torch.stack(
        [
            item["drug_embedding"]
            for item in batch
        ]
    )

    # ---------------------------------
    # Metadata
    # ---------------------------------

    broad_ids = [
        item["broad_id"]
        for item in batch
    ]

    depmap_ids = [
        item["depmap_id"]
        for item in batch
    ]

    # ---------------------------------
    # Output batch
    # ---------------------------------

    return {
        "drug_graph": drug_graph_batch,
        "drug_fp": drug_fp,
        "cell_expr": cell_expr,
        "ppi_expression": ppi_expression,
        "resistance_expr": resistance_expr,
        "label": labels,
        "cell_embedding": cell_embedding,
        "drug_embedding": drug_embedding,
        "broad_id": broad_ids,
        "depmap_id": depmap_ids,
    }