import random

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from src.data.combo_dataset import (
    PharosComboDataset,
)

from src.data.combo_collate import (
    pharos_combo_collate,
)

from src.models.pharos_combo import (
    PharosComboModel,
)

from src.evaluation.metrics import (
    regression_metrics,
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    loss_fn,
    device,
):

    model.eval()

    all_predictions = []
    all_labels = []

    total_loss = 0.0

    drug_a_gates = []
    drug_b_gates = []
    cell_gates = []

    for batch in loader:

        drug_a_graph = (
            batch["drug_a_graph"]
            .to(device)
        )

        drug_a_fp = (
            batch["drug_a_fp"]
            .to(device)
        )

        drug_b_graph = (
            batch["drug_b_graph"]
            .to(device)
        )

        drug_b_fp = (
            batch["drug_b_fp"]
            .to(device)
        )

        cell_expr = (
            batch["cell_expr"]
            .to(device)
        )

        ppi_expression = (
            batch["ppi_expression"]
            .to(device)
        )

        resistance_expr = (
            batch["resistance_expr"]
            .to(device)
        )

        labels = (
            batch["label"]
            .to(device)
        )

        output = model(

            drug_a_graph=drug_a_graph,
            drug_a_fp=drug_a_fp,

            drug_b_graph=drug_b_graph,
            drug_b_fp=drug_b_fp,

            cell_expr=cell_expr,
            ppi_expression=ppi_expression,

            resistance_expr=resistance_expr,

            return_details=True,
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

        all_predictions.extend(
            predictions
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )

        all_labels.extend(
            labels
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )

        drug_a_gates.append(
            output[
                "drug_a_gate"
            ]
            .detach()
            .cpu()
        )

        drug_b_gates.append(
            output[
                "drug_b_gate"
            ]
            .detach()
            .cpu()
        )

        cell_gates.append(
            output[
                "cell_gate"
            ]
            .detach()
            .cpu()
        )

    metrics = regression_metrics(
        all_labels,
        all_predictions,
    )

    metrics["loss"] = (
        total_loss
        / len(loader.dataset)
    )

    drug_a_gate = torch.cat(
        drug_a_gates,
        dim=0,
    )

    drug_b_gate = torch.cat(
        drug_b_gates,
        dim=0,
    )

    cell_gate = torch.cat(
        cell_gates,
        dim=0,
    )

    metrics[
        "drug_a_gnn_gate"
    ] = (
        drug_a_gate
        .mean()
        .item()
    )

    metrics[
        "drug_b_gnn_gate"
    ] = (
        drug_b_gate
        .mean()
        .item()
    )

    metrics[
        "ppi_gate"
    ] = (
        cell_gate
        .mean()
        .item()
    )

    metrics[
        "expression_gate"
    ] = (
        1.0
        -
        cell_gate
    ).mean().item()

    return metrics


# ============================================================
# Main
# ============================================================

def main():

    seed = 42

    set_seed(seed)

    # ========================================================
    # Device
    # ========================================================

    if torch.backends.mps.is_available():

        device = torch.device(
            "mps"
        )

    elif torch.cuda.is_available():

        device = torch.device(
            "cuda"
        )

    else:

        device = torch.device(
            "cpu"
        )

    print(
        f"Using device: {device}"
    )

    # ========================================================
    # Same 256 samples used for training AND evaluation
    # ========================================================

    dataset = PharosComboDataset(

        max_rows=256,

        seed=seed,
    )

    train_loader = DataLoader(

        dataset,

        batch_size=32,

        shuffle=True,

        num_workers=0,

        collate_fn=(
            pharos_combo_collate
        ),
    )

    eval_loader = DataLoader(

        dataset,

        batch_size=32,

        shuffle=False,

        num_workers=0,

        collate_fn=(
            pharos_combo_collate
        ),
    )

    sample = dataset[0]

    # ========================================================
    # PPI graph
    # ========================================================

    ppi = torch.load(

        "data/processed/"
        "pharos_ppi_graph.pt",

        map_location="cpu",

        weights_only=False,
    )

    # ========================================================
    # Dimensions
    # ========================================================

    drug_fp_dim = (
        sample[
            "drug_a_fp"
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
            "drug_a_graph"
        ].x.shape[1]
    )

    edge_dim = (
        sample[
            "drug_a_graph"
        ].edge_attr.shape[1]
    )

    print(
        "\n================================"
    )

    print(
        "PHAROS-COMBO OVERFIT TEST"
    )

    print(
        "================================"
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

    # ========================================================
    # Model
    #
    # dropout = 0 because this is deliberately
    # a memorisation test.
    # ========================================================

    model = PharosComboModel(

        drug_fp_dim=drug_fp_dim,

        cell_dim=cell_dim,

        resistance_dim=(
            resistance_dim
        ),

        ppi_edge_index=(
            ppi[
                "edge_index"
            ]
        ),

        ppi_edge_weight=(
            ppi[
                "edge_weight"
            ]
        ),

        num_ppi_genes=(
            num_ppi_genes
        ),

        node_dim=node_dim,

        edge_dim=edge_dim,

        hidden_dim=512,

        drug_gnn_hidden_dim=128,

        ppi_hidden_dim=64,

        drug_out_dim=256,

        pair_out_dim=256,

        cell_out_dim=256,

        resistance_out_dim=64,

        dropout=0.0,
    ).to(
        device
    )

    # ========================================================
    # Optimisation
    # ========================================================

    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=1e-3,

        weight_decay=0.0,
    )

    epochs = 200

    # ========================================================
    # Train
    # ========================================================

    for epoch in range(
        1,
        epochs + 1,
    ):

        model.train()

        total_train_loss = 0.0

        for batch in train_loader:

            drug_a_graph = (
                batch[
                    "drug_a_graph"
                ].to(device)
            )

            drug_a_fp = (
                batch[
                    "drug_a_fp"
                ].to(device)
            )

            drug_b_graph = (
                batch[
                    "drug_b_graph"
                ].to(device)
            )

            drug_b_fp = (
                batch[
                    "drug_b_fp"
                ].to(device)
            )

            cell_expr = (
                batch[
                    "cell_expr"
                ].to(device)
            )

            ppi_expression = (
                batch[
                    "ppi_expression"
                ].to(device)
            )

            resistance_expr = (
                batch[
                    "resistance_expr"
                ].to(device)
            )

            labels = (
                batch[
                    "label"
                ].to(device)
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            predictions = model(

                drug_a_graph=(
                    drug_a_graph
                ),

                drug_a_fp=(
                    drug_a_fp
                ),

                drug_b_graph=(
                    drug_b_graph
                ),

                drug_b_fp=(
                    drug_b_fp
                ),

                cell_expr=(
                    cell_expr
                ),

                ppi_expression=(
                    ppi_expression
                ),

                resistance_expr=(
                    resistance_expr
                ),
            )

            loss = loss_fn(
                predictions,
                labels,
            )

            loss.backward()

            optimizer.step()

            total_train_loss += (
                loss.item()
                * labels.size(0)
            )

        train_loss = (
            total_train_loss
            / len(dataset)
        )

        # ====================================================
        # Evaluate on SAME 256 rows
        # ====================================================

        metrics = evaluate(

            model=model,

            loader=eval_loader,

            loss_fn=loss_fn,

            device=device,
        )

        if (
            epoch == 1
            or epoch % 5 == 0
        ):

            print(

                f"Epoch {epoch:03d} | "

                f"Train "
                f"{train_loss:.6f} | "

                f"RMSE "
                f"{metrics['rmse']:.6f} | "

                f"MAE "
                f"{metrics['mae']:.6f} | "

                f"R2 "
                f"{metrics['r2']:.6f} | "

                f"Pearson "
                f"{metrics['pearson']:.6f} | "

                f"Spearman "
                f"{metrics['spearman']:.6f} | "

                f"A-GNN "
                f"{metrics['drug_a_gnn_gate']:.3f} | "

                f"B-GNN "
                f"{metrics['drug_b_gnn_gate']:.3f} | "

                f"PPI "
                f"{metrics['ppi_gate']:.3f}"
            )

        # ====================================================
        # Strong memorisation criterion
        # ====================================================

        if (
            metrics[
                "r2"
            ] > 0.98
            and
            metrics[
                "rmse"
            ] < 0.8
        ):

            print(
                "\n🔥 PHAROS-COMBO "
                "OVERFIT TEST PASSED"
            )

            print(
                f"Epoch: {epoch}"
            )

            print(
                f"RMSE: "
                f"{metrics['rmse']:.6f}"
            )

            print(
                f"MAE: "
                f"{metrics['mae']:.6f}"
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
                f"Drug A GNN gate: "
                f"{metrics['drug_a_gnn_gate']:.3f}"
            )

            print(
                f"Drug B GNN gate: "
                f"{metrics['drug_b_gnn_gate']:.3f}"
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

    # ========================================================
    # If target not reached
    # ========================================================

    print(
        "\n🚨 OVERFIT TARGET "
        "NOT REACHED"
    )

    print(
        metrics
    )


if __name__ == "__main__":

    main()