import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import (
    Dataset,
    DataLoader,
    Subset,
)

from src.data.combo_dataset import (
    PharosComboDataset,
)

from src.evaluation.metrics import (
    regression_metrics,
)

from src.models.pharos_combo_clamp_finetuned import (
    PharosComboCLAMPFineTuned,
)


# ============================================================
# Paths
# ============================================================

CLAMP_HIDDEN_PATH = Path(
    "data/processed/combination/"
    "pharos_combo_clamp_hidden_2048.npz"
)

CLAMP_FROZEN_PATH = Path(
    "data/processed/combination/"
    "pharos_combo_clamp_embeddings.npz"
)


# ============================================================
# Seed
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# Dataset
# ============================================================

class FineTunedCLAMPDataset(Dataset):

    def __init__(
        self,
        base_dataset,
    ):

        self.base_dataset = base_dataset

        self.metadata = (
            base_dataset
            .get_metadata()
            .reset_index(drop=True)
        )

        print(
            "\nLoading CLAMP hidden cache..."
        )

        cache = np.load(
            CLAMP_HIDDEN_PATH
        )

        drug_names = (
            cache[
                "drug_names"
            ]
            .astype(str)
        )

        hidden_features = (
            cache[
                "hidden_features"
            ]
            .astype(np.float32)
        )

        self.pretrained_weight = torch.from_numpy(
            cache[
                "linear_output_weight"
            ].astype(np.float32)
        )

        self.pretrained_bias = torch.from_numpy(
            cache[
                "linear_output_bias"
            ].astype(np.float32)
        )

        self.hidden_dim = (
            hidden_features.shape[1]
        )

        self.hidden_map = {
            str(drug):
                torch.from_numpy(
                    hidden_features[i]
                )

            for i, drug
            in enumerate(drug_names)
        }

        # ====================================================
        # Coverage check
        # ====================================================

        combo_drugs = set(
            self.metadata[
                "drug_a"
            ].astype(str)
        )

        combo_drugs.update(
            self.metadata[
                "drug_b"
            ].astype(str)
        )

        missing = (
            combo_drugs
            - set(
                self.hidden_map.keys()
            )
        )

        print(
            f"CLAMP hidden drugs: "
            f"{len(self.hidden_map):,}"
        )

        print(
            f"Hidden dimension: "
            f"{self.hidden_dim}"
        )

        print(
            f"Dataset drugs: "
            f"{len(combo_drugs):,}"
        )

        print(
            f"Missing drugs: "
            f"{len(missing)}"
        )

        if missing:

            raise RuntimeError(
                "Missing CLAMP hidden features."
            )

        print(
            "✅ CLAMP hidden coverage: 100%"
        )

    def __len__(self):

        return len(
            self.base_dataset
        )

    def __getitem__(
        self,
        idx,
    ):

        base_item = (
            self.base_dataset[
                idx
            ]
        )

        row = (
            self.metadata
            .iloc[idx]
        )

        drug_a = str(
            row[
                "drug_a"
            ]
        )

        drug_b = str(
            row[
                "drug_b"
            ]
        )

        return {
            "drug_a_hidden":
                self.hidden_map[
                    drug_a
                ],

            "drug_b_hidden":
                self.hidden_map[
                    drug_b
                ],

            "cell_expr":
                base_item[
                    "cell_expr"
                ],

            "label":
                base_item[
                    "label"
                ],
        }

    def get_metadata(self):

        return self.metadata


# ============================================================
# Collate
# ============================================================

def finetune_collate(batch):

    return {

        "drug_a_hidden":
            torch.stack(
                [
                    x[
                        "drug_a_hidden"
                    ]
                    for x in batch
                ]
            ),

        "drug_b_hidden":
            torch.stack(
                [
                    x[
                        "drug_b_hidden"
                    ]
                    for x in batch
                ]
            ),

        "cell_expr":
            torch.stack(
                [
                    x[
                        "cell_expr"
                    ]
                    for x in batch
                ]
            ),

        "label":
            torch.stack(
                [
                    x[
                        "label"
                    ]
                    for x in batch
                ]
            ),
    }


# ============================================================
# Saved held-out pair split
# ============================================================

def load_saved_split(
    dataset,
    seed,
):

    train_path = Path(
        f"outputs/"
        f"pharos_combo_seed_{seed}_train_pairs.csv"
    )

    val_path = Path(
        f"outputs/"
        f"pharos_combo_seed_{seed}_val_pairs.csv"
    )

    if not train_path.exists():

        raise FileNotFoundError(
            train_path
        )

    if not val_path.exists():

        raise FileNotFoundError(
            val_path
        )

    train_df = pd.read_csv(
        train_path
    )

    val_df = pd.read_csv(
        val_path
    )

    train_pairs = set(
        zip(
            train_df[
                "drug_a"
            ].astype(str),

            train_df[
                "drug_b"
            ].astype(str),
        )
    )

    val_pairs = set(
        zip(
            val_df[
                "drug_a"
            ].astype(str),

            val_df[
                "drug_b"
            ].astype(str),
        )
    )

    overlap = (
        train_pairs
        &
        val_pairs
    )

    if overlap:

        raise RuntimeError(
            "🚨 Pair leakage detected."
        )

    metadata = (
        dataset
        .get_metadata()
        .reset_index(drop=True)
    )

    row_pairs = list(
        zip(
            metadata[
                "drug_a"
            ].astype(str),

            metadata[
                "drug_b"
            ].astype(str),
        )
    )

    train_indices = [
        i
        for i, pair
        in enumerate(row_pairs)
        if pair in train_pairs
    ]

    val_indices = [
        i
        for i, pair
        in enumerate(row_pairs)
        if pair in val_pairs
    ]

    if (
        len(train_indices)
        + len(val_indices)
        != len(dataset)
    ):

        raise RuntimeError(
            "Some dataset rows were not "
            "assigned to the saved split."
        )

    print(
        "\n======================================"
    )

    print(
        "PAIR-LEVEL SPLIT"
    )

    print(
        "======================================"
    )

    print(
        f"Train pairs: "
        f"{len(train_pairs):,}"
    )

    print(
        f"Validation pairs: "
        f"{len(val_pairs):,}"
    )

    print(
        f"Train rows: "
        f"{len(train_indices):,}"
    )

    print(
        f"Validation rows: "
        f"{len(val_indices):,}"
    )

    print(
        f"Exact pair overlap: "
        f"{len(overlap)}"
    )

    return (

        Subset(
            dataset,
            train_indices,
        ),

        Subset(
            dataset,
            val_indices,
        ),
    )


# ============================================================
# Verify cache reconstruction
# ============================================================

def verify_pretrained_reconstruction(
    dataset,
):

    """
    Before fine-tuning, verify that:

        cached 2048-D hidden
             +
        cached pretrained W,b

    reconstructs the original frozen
    768-D CLAMP embedding.
    """

    print(
        "\n======================================"
    )

    print(
        "VERIFYING PRETRAINED CLAMP"
    )

    print(
        "======================================"
    )

    frozen = np.load(
        CLAMP_FROZEN_PATH
    )

    frozen_names = (
        frozen[
            "drug_names"
        ]
        .astype(str)
    )

    frozen_embeddings = (
        frozen[
            "embeddings"
        ]
        .astype(np.float32)
    )

    frozen_map = {
        str(drug):
            frozen_embeddings[i]

        for i, drug
        in enumerate(frozen_names)
    }

    hidden_cache = np.load(
        CLAMP_HIDDEN_PATH
    )

    names = (
        hidden_cache[
            "drug_names"
        ]
        .astype(str)
    )

    hidden = (
        hidden_cache[
            "hidden_features"
        ]
        .astype(np.float32)
    )

    weight = (
        hidden_cache[
            "linear_output_weight"
        ]
        .astype(np.float32)
    )

    bias = (
        hidden_cache[
            "linear_output_bias"
        ]
        .astype(np.float32)
    )

    # Check first 50 compounds
    errors = []

    check_n = min(
        50,
        len(names),
    )

    for i in range(
        check_n
    ):

        drug = str(
            names[i]
        )

        reconstructed = (
            hidden[i]
            @ weight.T
            + bias
        )

        original = (
            frozen_map[
                drug
            ]
        )

        errors.append(
            np.max(
                np.abs(
                    reconstructed
                    - original
                )
            )
        )

    max_error = float(
        np.max(errors)
    )

    print(
        f"Checked compounds: "
        f"{check_n}"
    )

    print(
        f"Maximum reconstruction error: "
        f"{max_error:.8f}"
    )

    # float32 tolerance
    if max_error > 1e-4:

        raise RuntimeError(
            "🚨 Pretrained CLAMP reconstruction "
            "does not match frozen embeddings."
        )

    print(
        "✅ Pretrained CLAMP reconstruction PASS"
    )


# ============================================================
# Training
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    device,
):

    model.train()

    total_loss = 0.0

    for batch in loader:

        drug_a = (
            batch[
                "drug_a_hidden"
            ]
            .to(device)
        )

        drug_b = (
            batch[
                "drug_b_hidden"
            ]
            .to(device)
        )

        cell_expr = (
            batch[
                "cell_expr"
            ]
            .to(device)
        )

        labels = (
            batch[
                "label"
            ]
            .to(device)
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        predictions = model(
            drug_a_hidden=drug_a,
            drug_b_hidden=drug_b,
            cell_expr=cell_expr,
        )

        loss = loss_fn(
            predictions,
            labels,
        )

        loss.backward()

        # Helps keep pretrained adaptation stable
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        total_loss += (
            loss.item()
            * labels.size(0)
        )

    return (
        total_loss
        / len(loader.dataset)
    )


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

    total_loss = 0.0

    predictions_all = []
    labels_all = []

    for batch in loader:

        drug_a = (
            batch[
                "drug_a_hidden"
            ]
            .to(device)
        )

        drug_b = (
            batch[
                "drug_b_hidden"
            ]
            .to(device)
        )

        cell_expr = (
            batch[
                "cell_expr"
            ]
            .to(device)
        )

        labels = (
            batch[
                "label"
            ]
            .to(device)
        )

        predictions = model(
            drug_a_hidden=drug_a,
            drug_b_hidden=drug_b,
            cell_expr=cell_expr,
        )

        loss = loss_fn(
            predictions,
            labels,
        )

        total_loss += (
            loss.item()
            * labels.size(0)
        )

        predictions_all.extend(
            predictions
            .cpu()
            .numpy()
            .reshape(-1)
        )

        labels_all.extend(
            labels
            .cpu()
            .numpy()
            .reshape(-1)
        )

    metrics = regression_metrics(
        labels_all,
        predictions_all,
    )

    metrics[
        "loss"
    ] = (
        total_loss
        / len(loader.dataset)
    )

    return metrics


# ============================================================
# Main
# ============================================================

def main(seed=42):

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
        f"Random seed: {seed}"
    )

    print(
        f"Using device: {device}"
    )

    # ========================================================
    # Verify frozen CLAMP reconstruction FIRST
    # ========================================================

    verify_pretrained_reconstruction(
        None
    )

    # ========================================================
    # Dataset
    # ========================================================

    print(
        "\nLoading PHAROS-Combo dataset..."
    )

    base_dataset = PharosComboDataset(
        max_rows=None,
        target="zip_score",
        seed=seed,
    )

    dataset = FineTunedCLAMPDataset(
        base_dataset
    )

    sample = dataset[0]

    (
        train_dataset,
        val_dataset,
    ) = load_saved_split(
        dataset,
        seed,
    )

    # ========================================================
    # Model
    # ========================================================

    model = PharosComboCLAMPFineTuned(

        pretrained_weight=(
            dataset
            .pretrained_weight
        ),

        pretrained_bias=(
            dataset
            .pretrained_bias
        ),

        clamp_hidden_dim=2048,
        clamp_dim=768,

        cell_dim=(
            sample[
                "cell_expr"
            ].shape[0]
        ),

        drug_out_dim=256,
        pair_out_dim=256,
        cell_out_dim=256,

        hidden_dim=512,
        dropout=0.2,

    ).to(
        device
    )

    # ========================================================
    # Parameter counts
    # ========================================================

    clamp_parameters = sum(
        p.numel()
        for p in (
            model
            .clamp_encoder
            .parameters()
        )
    )

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "\n======================================"
    )

    print(
        "PHAROS-COMBO PARTIAL CLAMP FINE-TUNE"
    )

    print(
        "======================================"
    )

    print(
        "Frozen CLAMP layers: "
        "8192→4096→2048"
    )

    print(
        "Fine-tuned CLAMP layer: "
        "2048→768"
    )

    print(
        f"Fine-tuned CLAMP parameters: "
        f"{clamp_parameters:,}"
    )

    print(
        f"Total trainable parameters: "
        f"{total_parameters:,}"
    )

    print(
        "Molecular GNN: OFF"
    )

    print(
        "PPI-GNN: OFF"
    )

    print(
        "Resistance FiLM: OFF"
    )

    print(
        "Target: ZIP synergy"
    )

    print(
        "Split: held-out drug pairs"
    )

    # ========================================================
    # Loaders
    # ========================================================

    generator = (
        torch.Generator()
        .manual_seed(seed)
    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=64,

        shuffle=True,

        generator=generator,

        num_workers=0,

        collate_fn=finetune_collate,
    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=64,

        shuffle=False,

        num_workers=0,

        collate_fn=finetune_collate,
    )

    # ========================================================
    # Different LRs
    # ========================================================

    clamp_params = list(
        model
        .clamp_encoder
        .parameters()
    )

    clamp_param_ids = {
        id(p)
        for p in clamp_params
    }

    pharos_params = [
        p
        for p in model.parameters()
        if id(p)
        not in clamp_param_ids
    ]

    optimizer = torch.optim.AdamW(

        [
            {
                "params":
                    pharos_params,

                "lr":
                    1e-3,
            },

            {
                "params":
                    clamp_params,

                "lr":
                    1e-5,
            },
        ],

        weight_decay=1e-5,
    )

    print(
        "\nLearning rates:"
    )

    print(
        "PHAROS layers: 1.00e-03"
    )

    print(
        "Pretrained CLAMP layer: 1.00e-05"
    )

    loss_fn = nn.MSELoss()

    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(

            optimizer,

            mode="min",

            factor=0.5,

            patience=5,

            min_lr=[
                1e-5,
                1e-7,
            ],
        )
    )

    # ========================================================
    # Training settings
    # ========================================================

    max_epochs = 80

    early_stopping_patience = 10

    best_loss = float(
        "inf"
    )

    best_epoch = 0

    best_metrics = None

    patience_counter = 0

    history = []

    # ========================================================
    # Outputs
    # ========================================================

    output_dir = Path(
        "outputs"
    )

    output_dir.mkdir(
        exist_ok=True
    )

    best_model_path = (
        output_dir
        /
        (
            f"pharos_combo_clamp_finetuned_"
            f"seed_{seed}_best_model.pt"
        )
    )

    history_path = (
        output_dir
        /
        (
            f"pharos_combo_clamp_finetuned_"
            f"seed_{seed}_training_history.csv"
        )
    )

    metrics_path = (
        output_dir
        /
        (
            f"pharos_combo_clamp_finetuned_"
            f"seed_{seed}_best_metrics.csv"
        )
    )

    # ========================================================
    # Training loop
    # ========================================================

    print(
        "\n======================================"
    )

    print(
        "STARTING PARTIAL CLAMP FINE-TUNING"
    )

    print(
        "======================================\n"
    )

    for epoch in range(
        1,
        max_epochs + 1,
    ):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
        )

        metrics = evaluate(
            model,
            val_loader,
            loss_fn,
            device,
        )

        scheduler.step(
            metrics[
                "loss"
            ]
        )

        pharos_lr = (
            optimizer
            .param_groups[0][
                "lr"
            ]
        )

        clamp_lr = (
            optimizer
            .param_groups[1][
                "lr"
            ]
        )

        history.append(
            {
                "epoch":
                    epoch,

                "seed":
                    seed,

                "train_loss":
                    train_loss,

                "pharos_lr":
                    pharos_lr,

                "clamp_lr":
                    clamp_lr,

                **metrics,
            }
        )

        print(
            f"Epoch {epoch:02d} | "

            f"Train: "
            f"{train_loss:.4f} | "

            f"Val: "
            f"{metrics['loss']:.4f} | "

            f"RMSE: "
            f"{metrics['rmse']:.4f} | "

            f"MAE: "
            f"{metrics['mae']:.4f} | "

            f"R2: "
            f"{metrics['r2']:.4f} | "

            f"Pearson: "
            f"{metrics['pearson']:.4f} | "

            f"Spearman: "
            f"{metrics['spearman']:.4f} | "

            f"PHAROS LR: "
            f"{pharos_lr:.1e} | "

            f"CLAMP LR: "
            f"{clamp_lr:.1e}"
        )

        # ====================================================
        # Best
        # ====================================================

        if (
            metrics[
                "loss"
            ]
            < best_loss
        ):

            best_loss = (
                metrics[
                    "loss"
                ]
            )

            best_epoch = epoch

            best_metrics = dict(
                metrics
            )

            best_metrics[
                "train_loss"
            ] = train_loss

            best_metrics[
                "best_epoch"
            ] = best_epoch

            best_metrics[
                "seed"
            ] = seed

            best_metrics[
                "pharos_lr"
            ] = pharos_lr

            best_metrics[
                "clamp_lr"
            ] = clamp_lr

            best_metrics[
                "model"
            ] = (
                "partial_finetuned_clamp_expression"
            )

            best_metrics[
                "drug_representation"
            ] = (
                "CLAMP_pretrained_"
                "2048to768_finetuned"
            )

            best_metrics[
                "split_type"
            ] = (
                "held_out_drug_pairs"
            )

            best_metrics[
                "train_rows"
            ] = (
                len(train_dataset)
            )

            best_metrics[
                "val_rows"
            ] = (
                len(val_dataset)
            )

            patience_counter = 0

            torch.save(

                {
                    "model_state_dict":
                        {
                            key:
                                value
                                .detach()
                                .cpu()

                            for key, value
                            in model
                            .state_dict()
                            .items()
                        },

                    "seed":
                        seed,

                    "best_epoch":
                        best_epoch,

                    "metrics":
                        best_metrics,
                },

                best_model_path,
            )

            pd.DataFrame(
                [
                    best_metrics
                ]
            ).to_csv(
                metrics_path,
                index=False,
            )

            print(
                "🔥 Saved new best "
                "fine-tuned CLAMP checkpoint"
            )

        else:

            patience_counter += 1

        # ====================================================
        # History
        # ====================================================

        pd.DataFrame(
            history
        ).to_csv(
            history_path,
            index=False,
        )

        # ====================================================
        # Early stopping
        # ====================================================

        if (
            patience_counter
            >= early_stopping_patience
        ):

            print(
                "\nEarly stopping triggered."
            )

            break

    # ========================================================
    # Complete
    # ========================================================

    print(
        "\n======================================"
    )

    print(
        "PARTIAL CLAMP FINE-TUNING COMPLETE"
    )

    print(
        "======================================"
    )

    print(
        f"\nBest epoch: "
        f"{best_epoch}"
    )

    print(
        "\nBest metrics:"
    )

    print(
        best_metrics
    )

    print(
        "\nBest model:"
    )

    print(
        best_model_path
    )

    print(
        "\nTraining history:"
    )

    print(
        history_path
    )

    print(
        "\nBest metrics:"
    )

    print(
        metrics_path
    )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    main(
        seed=args.seed
    )