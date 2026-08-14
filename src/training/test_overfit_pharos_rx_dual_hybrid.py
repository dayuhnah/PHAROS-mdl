import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import PharosDepMapDataset
from src.data.graph_collate import pharos_graph_collate
from src.evaluation.metrics import regression_metrics
from src.models.pharos_rx_dual_hybrid import (
    PharosRXDualHybridModel,
)


# =========================================================
# Reproducibility
# =========================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# =========================================================
# Evaluation
# =========================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    loss_fn,
    device,
):

    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    drug_gates = []
    cell_gates = []

    for batch in loader:

        drug_graph = batch[
            "drug_graph"
        ].to(device)

        drug_fp = batch[
            "drug_fp"
        ].to(device)

        cell_expr = batch[
            "cell_expr"
        ].to(device)

        ppi_expression = batch[
            "ppi_expression"
        ].to(device)

        resistance_expr = batch[
            "resistance_expr"
        ].to(device)

        labels = batch[
            "label"
        ].to(device)

        output = model(
            drug_graph=drug_graph,
            drug_fp=drug_fp,
            cell_expr=cell_expr,
            ppi_expression=ppi_expression,
            resistance_expr=resistance_expr,
            return_gates=True,
        )

        predictions = output[
            "prediction"
        ]

        loss = loss_fn(
            predictions,
            labels,
        )

        total_loss += (
            loss.item()
            * labels.size(0)
        )

        all_preds.extend(
            predictions
            .cpu()
            .numpy()
            .reshape(-1)
        )

        all_labels.extend(
            labels
            .cpu()
            .numpy()
            .reshape(-1)
        )

        drug_gates.append(
            output[
                "drug_gate"
            ].cpu()
        )

        cell_gates.append(
            output[
                "cell_gate"
            ].cpu()
        )

    metrics = regression_metrics(
        all_labels,
        all_preds,
    )

    metrics["loss"] = (
        total_loss
        / len(loader.dataset)
    )

    drug_gate = torch.cat(
        drug_gates,
        dim=0,
    )

    cell_gate = torch.cat(
        cell_gates,
        dim=0,
    )

    metrics[
        "gnn_drug_gate"
    ] = drug_gate.mean().item()

    metrics[
        "morgan_gate"
    ] = (
        1.0 - drug_gate
    ).mean().item()

    metrics[
        "ppi_gate"
    ] = cell_gate.mean().item()

    metrics[
        "expression_gate"
    ] = (
        1.0 - cell_gate
    ).mean().item()

    return metrics


# =========================================================
# Main
# =========================================================

def main():

    seed = 42
    set_seed(seed)

    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------

    if torch.backends.mps.is_available():
        device = torch.device("mps")

    elif torch.cuda.is_available():
        device = torch.device("cuda")

    else:
        device = torch.device("cpu")

    print(
        f"Using device: {device}"
    )

    # -----------------------------------------------------
    # Tiny dataset
    # -----------------------------------------------------

    dataset = PharosDepMapDataset(
        max_rows=256,
        use_drug_graphs=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        collate_fn=pharos_graph_collate,
    )

    eval_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=pharos_graph_collate,
    )

    sample = dataset[0]

    # -----------------------------------------------------
    # PPI graph
    # -----------------------------------------------------

    ppi = torch.load(
        "data/processed/pharos_ppi_graph.pt",
        map_location="cpu",
        weights_only=False,
    )

    # -----------------------------------------------------
    # Dimensions
    # -----------------------------------------------------

    drug_fp_dim = (
        sample[
            "drug_fp"
        ].shape[0]
    )

    cell_dim = (
        sample[
            "cell_expr"
        ].shape[0]
    )

    resistance_dim = (
        sample[
            "resistance_expr"
        ].shape[0]
    )

    num_ppi_genes = (
        sample[
            "ppi_expression"
        ].shape[0]
    )

    node_dim = (
        sample[
            "drug_graph"
        ].x.shape[1]
    )

    edge_dim = (
        sample[
            "drug_graph"
        ].edge_attr.shape[1]
    )

    print(
        "\nDimensions:"
    )

    print(
        "Rows:",
        len(dataset),
    )

    print(
        "Drug FP:",
        drug_fp_dim,
    )

    print(
        "Cell:",
        cell_dim,
    )

    print(
        "PPI:",
        num_ppi_genes,
    )

    print(
        "Resistance:",
        resistance_dim,
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

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

        # IMPORTANT:
        # dropout off for memorisation test
        dropout=0.0,
    ).to(device)

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.0,
    )

    epochs = 120

    print(
        "\n================================"
    )

    print(
        "V2B OVERFIT SANITY TEST"
    )

    print(
        "================================"
    )

    for epoch in range(
        1,
        epochs + 1,
    ):

        model.train()

        train_loss = 0.0

        for batch in loader:

            drug_graph = batch[
                "drug_graph"
            ].to(device)

            drug_fp = batch[
                "drug_fp"
            ].to(device)

            cell_expr = batch[
                "cell_expr"
            ].to(device)

            ppi_expression = batch[
                "ppi_expression"
            ].to(device)

            resistance_expr = batch[
                "resistance_expr"
            ].to(device)

            labels = batch[
                "label"
            ].to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            predictions = model(
                drug_graph=drug_graph,
                drug_fp=drug_fp,
                cell_expr=cell_expr,
                ppi_expression=ppi_expression,
                resistance_expr=resistance_expr,
            )

            loss = loss_fn(
                predictions,
                labels,
            )

            loss.backward()
            optimizer.step()

            train_loss += (
                loss.item()
                * labels.size(0)
            )

        train_loss /= len(
            dataset
        )

        # ---------------------------------------------
        # Evaluate on the SAME 256 examples
        # ---------------------------------------------

        metrics = evaluate(
            model,
            eval_loader,
            loss_fn,
            device,
        )

        if (
            epoch == 1
            or epoch % 5 == 0
        ):

            print(
                f"Epoch {epoch:03d} | "
                f"Train {train_loss:.6f} | "
                f"RMSE {metrics['rmse']:.6f} | "
                f"MAE {metrics['mae']:.6f} | "
                f"R2 {metrics['r2']:.6f} | "
                f"Pearson "
                f"{metrics['pearson']:.6f} | "
                f"Spearman "
                f"{metrics['spearman']:.6f} | "
                f"Drug GNN "
                f"{metrics['gnn_drug_gate']:.3f} | "
                f"PPI "
                f"{metrics['ppi_gate']:.3f}"
            )

        # ---------------------------------------------
        # Strong memorisation achieved
        # ---------------------------------------------

        if (
            metrics["rmse"] < 0.08
            and metrics["r2"] > 0.98
        ):

            print(
                "\n🔥 OVERFIT TEST PASSED"
            )

            print(
                f"Epoch: {epoch}"
            )

            print(
                f"RMSE: "
                f"{metrics['rmse']:.6f}"
            )

            print(
                f"R2: "
                f"{metrics['r2']:.6f}"
            )

            print(
                f"Pearson: "
                f"{metrics['pearson']:.6f}"
            )

            print(
                f"Spearman: "
                f"{metrics['spearman']:.6f}"
            )

            print(
                f"Drug GNN gate: "
                f"{metrics['gnn_drug_gate']:.3f}"
            )

            print(
                f"Morgan gate: "
                f"{metrics['morgan_gate']:.3f}"
            )

            print(
                f"PPI gate: "
                f"{metrics['ppi_gate']:.3f}"
            )

            print(
                f"Expression gate: "
                f"{metrics['expression_gate']:.3f}"
            )

            return

    # -----------------------------------------------------
    # Failed to strongly memorise
    # -----------------------------------------------------

    print(
        "\n🚨 OVERFIT TARGET NOT REACHED"
    )

    print(
        "Final metrics:"
    )

    print(
        metrics
    )


if __name__ == "__main__":
    main()