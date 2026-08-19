import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import (
    DataLoader,
    Dataset,
    Subset,
)

from src.data.combo_dataset import (
    PharosComboDataset,
)

from src.evaluation.metrics import (
    regression_metrics,
)

from src.models.pharos_combo_clamp_resistance import (
    PharosComboCLAMPResistance,
)


# ============================================================
# Paths
# ============================================================

CLAMP_CACHE_PATH = Path(
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
# Dataset wrapper
# ============================================================

class HybridComboDataset(Dataset):

    def __init__(
        self,
        base_dataset,
        clamp_cache_path=CLAMP_CACHE_PATH,
    ):
        self.base_dataset = base_dataset

        self.metadata = (
            base_dataset
            .get_metadata()
            .reset_index(drop=True)
        )

        print("\nLoading cached CLAMP embeddings...")

        cache = np.load(clamp_cache_path)

        drug_names = cache["drug_names"].astype(str)
        embeddings = cache["embeddings"].astype(np.float32)

        self.clamp_dim = embeddings.shape[1]

        self.embedding_map = {
            str(drug): torch.from_numpy(embeddings[i])
            for i, drug in enumerate(drug_names)
        }

        # ====================================================
        # Validate CLAMP coverage
        # ====================================================

        combo_drugs = set(
            self.metadata["drug_a"].astype(str)
        )

        combo_drugs.update(
            self.metadata["drug_b"].astype(str)
        )

        missing = sorted(
            combo_drugs - set(self.embedding_map.keys())
        )

        print(f"CLAMP embeddings: {len(self.embedding_map):,}")
        print(f"CLAMP dimension: {self.clamp_dim}")
        print(f"Dataset drugs: {len(combo_drugs):,}")
        print(f"Missing CLAMP drugs: {len(missing)}")

        if missing:
            print("First missing drugs:")
            for drug in missing[:20]:
                print(f"  - {drug}")

            raise RuntimeError(
                "Missing CLAMP embeddings for dataset drugs."
            )

        print("✅ CLAMP coverage: 100%")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        base_item = self.base_dataset[idx]
        row = self.metadata.iloc[idx]

        drug_a = str(row["drug_a"])
        drug_b = str(row["drug_b"])

        # Dedicated 29-gene resistance tensor is deliberately
        # kept separate from the general-expression tensor.
        if "resistance_expr" not in base_item:
            raise KeyError(
                "Expected base dataset item to contain 'resistance_expr'. "
                "Check PharosComboDataset before training."
            )

        return {
            "drug_a_fp": base_item["drug_a_fp"],
            "drug_a_clamp": self.embedding_map[drug_a],
            "drug_b_fp": base_item["drug_b_fp"],
            "drug_b_clamp": self.embedding_map[drug_b],
            "cell_expr": base_item["cell_expr"],
            "resistance_expr": base_item["resistance_expr"],
            "label": base_item["label"],
        }

    def get_metadata(self):
        return self.metadata


# ============================================================
# Collate
# ============================================================

def hybrid_collate(batch):
    return {
        "drug_a_fp": torch.stack(
            [item["drug_a_fp"] for item in batch]
        ),
        "drug_a_clamp": torch.stack(
            [item["drug_a_clamp"] for item in batch]
        ),
        "drug_b_fp": torch.stack(
            [item["drug_b_fp"] for item in batch]
        ),
        "drug_b_clamp": torch.stack(
            [item["drug_b_clamp"] for item in batch]
        ),
        "cell_expr": torch.stack(
            [item["cell_expr"] for item in batch]
        ),
        "resistance_expr": torch.stack(
            [item["resistance_expr"] for item in batch]
        ),
        "label": torch.stack(
            [item["label"] for item in batch]
        ),
    }


# ============================================================
# Load exact same saved pair split
# ============================================================

def load_saved_split(dataset, seed):
    output_dir = Path("outputs")

    train_path = (
        output_dir
        / f"pharos_combo_seed_{seed}_train_pairs.csv"
    )

    val_path = (
        output_dir
        / f"pharos_combo_seed_{seed}_val_pairs.csv"
    )

    if not train_path.exists():
        raise FileNotFoundError(f"Missing saved train split: {train_path}")

    if not val_path.exists():
        raise FileNotFoundError(f"Missing saved validation split: {val_path}")

    train_pairs = pd.read_csv(train_path)
    val_pairs = pd.read_csv(val_path)

    train_pair_set = set(
        zip(
            train_pairs["drug_a"].astype(str),
            train_pairs["drug_b"].astype(str),
        )
    )

    val_pair_set = set(
        zip(
            val_pairs["drug_a"].astype(str),
            val_pairs["drug_b"].astype(str),
        )
    )

    overlap = train_pair_set & val_pair_set

    if overlap:
        raise RuntimeError(
            "🚨 Drug-pair leakage detected."
        )

    metadata = (
        dataset
        .get_metadata()
        .reset_index(drop=True)
    )

    row_pairs = list(
        zip(
            metadata["drug_a"].astype(str),
            metadata["drug_b"].astype(str),
        )
    )

    train_indices = [
        i
        for i, pair in enumerate(row_pairs)
        if pair in train_pair_set
    ]

    val_indices = [
        i
        for i, pair in enumerate(row_pairs)
        if pair in val_pair_set
    ]

    if len(train_indices) + len(val_indices) != len(dataset):
        raise RuntimeError(
            "Some rows were not assigned to the saved split."
        )

    print("\n======================================")
    print("PAIR-LEVEL SPLIT")
    print("======================================")
    print(f"Train pairs: {len(train_pair_set):,}")
    print(f"Validation pairs: {len(val_pair_set):,}")
    print(f"Train rows: {len(train_indices):,}")
    print(f"Validation rows: {len(val_indices):,}")
    print(f"Exact pair overlap: {len(overlap)}")

    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
    )


# ============================================================
# Train epoch
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
        drug_a_fp = batch["drug_a_fp"].to(device)
        drug_a_clamp = batch["drug_a_clamp"].to(device)
        drug_b_fp = batch["drug_b_fp"].to(device)
        drug_b_clamp = batch["drug_b_clamp"].to(device)
        cell_expr = batch["cell_expr"].to(device)
        resistance_expr = batch["resistance_expr"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)

        predictions = model(
            drug_a_fp=drug_a_fp,
            drug_a_clamp=drug_a_clamp,
            drug_b_fp=drug_b_fp,
            drug_b_clamp=drug_b_clamp,
            cell_expr=cell_expr,
            resistance_expr=resistance_expr,
        )

        loss = loss_fn(
            predictions,
            labels,
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

    return total_loss / len(loader.dataset)


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

    all_predictions = []
    all_labels = []

    clamp_gates_a = []
    clamp_gates_b = []

    for batch in loader:
        drug_a_fp = batch["drug_a_fp"].to(device)
        drug_a_clamp = batch["drug_a_clamp"].to(device)
        drug_b_fp = batch["drug_b_fp"].to(device)
        drug_b_clamp = batch["drug_b_clamp"].to(device)
        cell_expr = batch["cell_expr"].to(device)
        resistance_expr = batch["resistance_expr"].to(device)
        labels = batch["label"].to(device)

        output = model(
            drug_a_fp=drug_a_fp,
            drug_a_clamp=drug_a_clamp,
            drug_b_fp=drug_b_fp,
            drug_b_clamp=drug_b_clamp,
            cell_expr=cell_expr,
            resistance_expr=resistance_expr,
            return_details=True,
        )

        predictions = output["prediction"]

        loss = loss_fn(
            predictions,
            labels,
        )

        total_loss += loss.item() * labels.size(0)

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

        clamp_gates_a.append(
            output["drug_a_clamp_gate"]
            .detach()
            .cpu()
        )

        clamp_gates_b.append(
            output["drug_b_clamp_gate"]
            .detach()
            .cpu()
        )

    metrics = regression_metrics(
        all_labels,
        all_predictions,
    )

    metrics["loss"] = (
        total_loss / len(loader.dataset)
    )

    # ========================================================
    # Gate diagnostics
    # ========================================================

    gate_a = torch.cat(clamp_gates_a, dim=0)
    gate_b = torch.cat(clamp_gates_b, dim=0)

    combined_gate = torch.cat(
        [gate_a, gate_b],
        dim=0,
    )

    mean_clamp = combined_gate.mean().item()

    metrics["mean_drug_a_clamp_gate"] = (
        gate_a.mean().item()
    )

    metrics["mean_drug_b_clamp_gate"] = (
        gate_b.mean().item()
    )

    metrics["mean_clamp_gate"] = mean_clamp
    metrics["mean_morgan_gate"] = 1.0 - mean_clamp

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
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Random seed: {seed}")
    print(f"Using device: {device}")

    # ========================================================
    # Dataset
    # ========================================================

    print("\nLoading PHAROS-Combo dataset...")

    base_dataset = PharosComboDataset(
        max_rows=None,
        target="zip_score",
        seed=seed,
    )

    dataset = HybridComboDataset(
        base_dataset
    )

    sample = dataset[0]

    if sample["resistance_expr"].shape[0] != 29:
        raise RuntimeError(
            "🚨 Expected 29 resistance genes, got "
            f"{sample['resistance_expr'].shape[0]}."
        )

    (
        train_dataset,
        val_dataset,
    ) = load_saved_split(
        dataset,
        seed,
    )

    # ========================================================
    # Summary
    # ========================================================

    print("\n======================================")
    print("PHAROS-COMBO PRETRAINED + RESISTANCE")
    print("======================================")
    print(f"Morgan dimension: {sample['drug_a_fp'].shape[0]}")
    print(f"CLAMP dimension: {dataset.clamp_dim}")
    print(f"General expression dimension: {sample['cell_expr'].shape[0]}")
    print(f"Resistance genes: {sample['resistance_expr'].shape[0]}")
    print("Morgan: ON")
    print("Frozen pretrained CLAMP: ON")
    print("General expression: ON")
    print("Resistance FiLM: ON")
    print("Molecular GNN: OFF")
    print("PPI-GNN: OFF")
    print("Drug fusion: learned feature-wise Morgan/CLAMP gate")
    print("Pair representation: A+B / |A-B| / A*B")
    print("Resistance conditioning: drug-pair embedding")
    print("Target: ZIP synergy")
    print("Split: held-out exact drug pairs")

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
        collate_fn=hybrid_collate,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        collate_fn=hybrid_collate,
    )

    # ========================================================
    # Model
    # ========================================================

    set_seed(seed)

    model = PharosComboCLAMPResistance(
        fingerprint_dim=sample["drug_a_fp"].shape[0],
        clamp_dim=dataset.clamp_dim,
        cell_dim=sample["cell_expr"].shape[0],
        resistance_dim=sample["resistance_expr"].shape[0],
        drug_out_dim=256,
        pair_out_dim=256,
        cell_out_dim=256,
        resistance_out_dim=64,
        hidden_dim=512,
        dropout=0.2,
    ).to(device)

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"\nTrainable parameters: {total_parameters:,}")

    # ========================================================
    # Optimisation
    # ========================================================

    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-5,
    )

    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
        )
    )

    max_epochs = 80
    early_stopping_patience = 10

    best_loss = float("inf")
    best_epoch = 0
    best_metrics = None
    patience_counter = 0
    history = []

    # ========================================================
    # Outputs
    # ========================================================

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    best_model_path = (
        output_dir
        / (
            f"pharos_combo_clamp_resistance_"
            f"seed_{seed}_best_model.pt"
        )
    )

    history_path = (
        output_dir
        / (
            f"pharos_combo_clamp_resistance_"
            f"seed_{seed}_training_history.csv"
        )
    )

    metrics_path = (
        output_dir
        / (
            f"pharos_combo_clamp_resistance_"
            f"seed_{seed}_best_metrics.csv"
        )
    )

    # ========================================================
    # Training
    # ========================================================

    print("\n======================================")
    print("STARTING MORGAN + CLAMP + RESISTANCE")
    print("======================================\n")

    for epoch in range(1, max_epochs + 1):
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

        scheduler.step(metrics["loss"])

        current_lr = optimizer.param_groups[0]["lr"]

        history.append(
            {
                "epoch": epoch,
                "seed": seed,
                "train_loss": train_loss,
                "lr": current_lr,
                **metrics,
            }
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {metrics['loss']:.4f} | "
            f"RMSE: {metrics['rmse']:.4f} | "
            f"MAE: {metrics['mae']:.4f} | "
            f"R2: {metrics['r2']:.4f} | "
            f"Pearson: {metrics['pearson']:.4f} | "
            f"Spearman: {metrics['spearman']:.4f} | "
            f"CLAMP: {metrics['mean_clamp_gate']:.3f} | "
            f"Morgan: {metrics['mean_morgan_gate']:.3f} | "
            f"LR: {current_lr:.2e}"
        )

        # ====================================================
        # Best checkpoint
        # ====================================================

        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            best_epoch = epoch
            best_metrics = dict(metrics)

            best_metrics["train_loss"] = train_loss
            best_metrics["best_epoch"] = best_epoch
            best_metrics["seed"] = seed
            best_metrics["lr"] = current_lr
            best_metrics["model"] = (
                "morgan_pretrained_clamp_resistance_film"
            )
            best_metrics["drug_representation"] = (
                "Morgan_2048+CLAMP_frozen_768"
            )
            best_metrics["resistance_representation"] = (
                "29_gene_pair_conditioned_FiLM"
            )
            best_metrics["split_type"] = (
                "held_out_drug_pairs"
            )
            best_metrics["train_rows"] = len(train_dataset)
            best_metrics["val_rows"] = len(val_dataset)

            patience_counter = 0

            torch.save(
                {
                    key: value.detach().cpu()
                    for key, value
                    in model.state_dict().items()
                },
                best_model_path,
            )

            pd.DataFrame(
                [best_metrics]
            ).to_csv(
                metrics_path,
                index=False,
            )

            print(
                "🔥 Saved new best "
                "Morgan + CLAMP + Resistance checkpoint"
            )

        else:
            patience_counter += 1

        # ====================================================
        # Save history
        # ====================================================

        pd.DataFrame(
            history
        ).to_csv(
            history_path,
            index=False,
        )

        # ====================================================
        # Early stop
        # ====================================================

        if patience_counter >= early_stopping_patience:
            print("\nEarly stopping triggered.")
            break

    # ========================================================
    # Complete
    # ========================================================

    print("\n======================================")
    print("MORGAN + CLAMP + RESISTANCE COMPLETE")
    print("======================================")
    print(f"\nBest epoch: {best_epoch}")
    print("\nBest metrics:")
    print(best_metrics)
    print("\nBest model:")
    print(best_model_path)
    print("\nTraining history:")
    print(history_path)
    print("\nBest metrics:")
    print(metrics_path)


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

    main(seed=args.seed)