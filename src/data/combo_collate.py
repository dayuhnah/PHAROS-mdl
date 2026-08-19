import torch

from torch_geometric.data import Batch


def pharos_combo_collate(batch):

    # ========================================================
    # Drug A graphs
    # ========================================================

    drug_a_graphs = [
        item["drug_a_graph"]
        for item in batch
    ]

    drug_a_graph = (
        Batch.from_data_list(
            drug_a_graphs
        )
    )

    # ========================================================
    # Drug B graphs
    # ========================================================

    drug_b_graphs = [
        item["drug_b_graph"]
        for item in batch
    ]

    drug_b_graph = (
        Batch.from_data_list(
            drug_b_graphs
        )
    )

    # ========================================================
    # Fingerprints
    # ========================================================

    drug_a_fp = torch.stack(
        [
            item["drug_a_fp"]
            for item in batch
        ]
    )

    drug_b_fp = torch.stack(
        [
            item["drug_b_fp"]
            for item in batch
        ]
    )

    # ========================================================
    # Cell features
    # ========================================================

    cell_expr = torch.stack(
        [
            item["cell_expr"]
            for item in batch
        ]
    )

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

    # ========================================================
    # Labels
    # ========================================================

    labels = torch.stack(
        [
            item["label"]
            for item in batch
        ]
    )

    # ========================================================
    # Metadata
    # ========================================================

    drug_a_names = [
        item["drug_a"]
        for item in batch
    ]

    drug_b_names = [
        item["drug_b"]
        for item in batch
    ]

    depmap_ids = [
        item["depmap_id"]
        for item in batch
    ]

    # ========================================================
    # Return
    # ========================================================

    return {

        "drug_a_graph":
            drug_a_graph,

        "drug_a_fp":
            drug_a_fp,

        "drug_b_graph":
            drug_b_graph,

        "drug_b_fp":
            drug_b_fp,

        "cell_expr":
            cell_expr,

        "ppi_expression":
            ppi_expression,

        "resistance_expr":
            resistance_expr,

        "label":
            labels,

        "drug_a":
            drug_a_names,

        "drug_b":
            drug_b_names,

        "depmap_id":
            depmap_ids,
    }