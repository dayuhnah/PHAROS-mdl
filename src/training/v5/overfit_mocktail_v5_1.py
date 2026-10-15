import random

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import (
    DataLoader,
    Subset,
)

from src.data.v4.mocktail_v4_dataset import (
    PharosMocktailV4Dataset,
)

from src.data.v5.cellresdb_relevance import (
    CellResDBRelevanceMapper,
)

from src.models.v5.pharos_mocktail_v5_1 import (
    PharosMocktailV51,
)


SEED = 42
N_ROWS = 256
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-3


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():

    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 65)
    print("PHAROS MOCKTAIL V5.1 RELEVANCE-ENRICHED OVERFIT SANITY")
    print("=" * 65)

    print("Device:", device)

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    full_dataset = PharosMocktailV4Dataset(
        split="train",
        seed=SEED,
    )

    # --------------------------------------------------------
    # CELLRESDB RELEVANCE MAPPER
    # --------------------------------------------------------

    cellresdb_mapper = CellResDBRelevanceMapper(
        metadata_path=(
            "data/processed/v5/cellresdb/"
            "resistance_bank_cohort/model_cache/"
            "cellresdb_token_metadata.json"
        ),
        pathway_tokens_path=(
            "data/processed/v5/cellresdb/"
            "resistance_bank_cohort/"
            "cellresdb_pathway_tokens.csv"
        ),
    )

    # --------------------------------------------------------
    # RELEVANCE-ENRICHED 256-ROW SANITY SET
    #
    # 64 unmatched
    # 64 Paclitaxel / TNBC
    # 64 Vismodegib / BCC
    # 64 MM-associated
    #
    # Categories are kept mutually exclusive so each group
    # tests one clear relevance pathway.
    # --------------------------------------------------------

    group_size = N_ROWS // 4

    if group_size * 4 != N_ROWS:
        raise ValueError(
            "N_ROWS must be divisible by 4."
        )

    mm_drugs = {
        "carfilzomib",
        "dexamethasone",
        "pomalidomide",
    }

    matched_drugs = set(
        cellresdb_mapper.drug_to_cohorts.keys()
    )

    selected = {
        "unmatched": [],
        "paclitaxel": [],
        "vismodegib": [],
        "mm": [],
    }

    for i in range(len(full_dataset)):

        item = full_dataset[i]

        a = str(
            item["drug_a"]
        ).strip().lower()

        b = str(
            item["drug_b"]
        ).strip().lower()

        pair = {a, b}

        has_paclitaxel = (
            "paclitaxel" in pair
        )

        has_vismodegib = (
            "vismodegib" in pair
        )

        has_mm = bool(
            pair & mm_drugs
        )

        # Pure Paclitaxel / TNBC example.
        if (
            has_paclitaxel
            and not has_vismodegib
            and not has_mm
            and len(
                selected["paclitaxel"]
            ) < group_size
        ):
            selected["paclitaxel"].append(i)

        # Pure Vismodegib / BCC example.
        elif (
            has_vismodegib
            and not has_paclitaxel
            and not has_mm
            and len(
                selected["vismodegib"]
            ) < group_size
        ):
            selected["vismodegib"].append(i)

        # Pure MM-evidence example.
        elif (
            has_mm
            and not has_paclitaxel
            and not has_vismodegib
            and len(
                selected["mm"]
            ) < group_size
        ):
            selected["mm"].append(i)

        # Completely unmatched example.
        elif (
            not (
                pair
                & matched_drugs
            )
            and len(
                selected["unmatched"]
            ) < group_size
        ):
            selected["unmatched"].append(i)

        if all(
            len(indices) == group_size
            for indices in selected.values()
        ):
            break

    print()
    print("=" * 65)
    print("OVERFIT SUBSET COMPOSITION")
    print("=" * 65)

    for name, indices in selected.items():
        print(
            f"{name:12s}:",
            len(indices),
        )

    for name, indices in selected.items():
        if len(indices) != group_size:
            raise RuntimeError(
                f"Could not collect {group_size} "
                f"rows for group '{name}'. "
                f"Found {len(indices)}."
            )

    selected_indices = (
        selected["unmatched"]
        + selected["paclitaxel"]
        + selected["vismodegib"]
        + selected["mm"]
    )

    if len(set(selected_indices)) != N_ROWS:
        raise RuntimeError(
            "Overfit subset contains duplicate rows."
        )

    subset = Subset(
        full_dataset,
        selected_indices,
    )

    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    eval_loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # --------------------------------------------------------
    # MODEL
    #
    # Dropout disabled ONLY for overfit sanity.
    # Full training still uses dropout=0.2.
    # --------------------------------------------------------

    model = PharosMocktailV51(
        expression_dim=full_dataset.expression_dim,
        cnv_dim=full_dataset.cnv_dim,
        mutation_dim=full_dataset.mutation_dim,
        crispr_dim=full_dataset.crispr_dim,
        dropout=0.0,
    ).to(device)

    print(
        "Parameters:",
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=0.0,
    )

    criterion = nn.MSELoss()

    # --------------------------------------------------------
    # FORWARD HELPER
    # --------------------------------------------------------

    def forward_batch(batch):

        def x(name):
            return batch[name].to(
                device,
                non_blocking=True,
            )

        cellresdb_token_relevance = (
            cellresdb_mapper.build(
                batch["drug_a"],
                batch["drug_b"],
                device=device,
            )
        )

        pred = model(
            x("drug_a_fp"),
            x("drug_a_clamp"),

            x("drug_b_fp"),
            x("drug_b_clamp"),

            x("expression"),
            x("cnv"),
            x("mutation"),
            x("crispr"),
            x("omics_mask"),

            x("drug_a_drmref_expr"),
            x("drug_b_drmref_expr"),

            x("drug_a_drmref_available"),
            x("drug_b_drmref_available"),
            cellresdb_token_relevance=(
                cellresdb_token_relevance
            ),
        )

        target = (
            x("label")
            .float()
            .view(-1)
        )

        return (
            pred.squeeze(-1),
            target,
        )

    # --------------------------------------------------------
    # EVALUATION ON SAME 256 ROWS
    # --------------------------------------------------------

    @torch.no_grad()
    def evaluate():

        model.eval()

        preds = []
        ys = []

        for batch in eval_loader:

            pred, target = forward_batch(
                batch
            )

            preds.append(
                pred.detach().cpu()
            )

            ys.append(
                target.detach().cpu()
            )

        pred = torch.cat(preds)
        y = torch.cat(ys)

        mse = torch.mean(
            (pred - y) ** 2
        )

        rmse = torch.sqrt(mse).item()

        ss_res = torch.sum(
            (y - pred) ** 2
        )

        ss_tot = torch.sum(
            (y - y.mean()) ** 2
        )

        r2 = (
            1.0
            - (ss_res / ss_tot).item()
            if ss_tot > 0
            else 0.0
        )

        return rmse, r2

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        total_loss = 0.0
        total_n = 0

        for batch in loader:

            optimizer.zero_grad(
                set_to_none=True
            )

            pred, target = forward_batch(
                batch
            )

            loss = criterion(
                pred,
                target,
            )

            loss.backward()

            optimizer.step()

            n = target.numel()

            total_loss += (
                loss.item() * n
            )

            total_n += n

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == EPOCHS
        ):

            train_rmse = (
                total_loss
                / total_n
            ) ** 0.5

            eval_rmse, r2 = evaluate()

            print(
                f"Epoch {epoch:3d} | "
                f"Train RMSE {train_rmse:.6f} | "
                f"Same-set RMSE {eval_rmse:.6f} | "
                f"R2 {r2:.6f}"
            )

    # --------------------------------------------------------
    # FINAL ATTENTION CHECK
    # --------------------------------------------------------

    model.eval()

    batch = next(iter(eval_loader))

    def x(name):
        return batch[name].to(device)

    with torch.no_grad():

        out = model(
            x("drug_a_fp"),
            x("drug_a_clamp"),

            x("drug_b_fp"),
            x("drug_b_clamp"),

            x("expression"),
            x("cnv"),
            x("mutation"),
            x("crispr"),
            x("omics_mask"),

            x("drug_a_drmref_expr"),
            x("drug_b_drmref_expr"),

            x("drug_a_drmref_available"),
            x("drug_b_drmref_available"),

            return_details=True,
        )

    print()
    print(
        "Final CellResDB attention sums:"
    )

    print(
        out[
            "cellresdb_token_attention"
        ]
        .sum(dim=1)
        .cpu()
    )

    print()
    print("DONE")


if __name__ == "__main__":
    main()
