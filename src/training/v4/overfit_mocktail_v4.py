import random
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset

from src.data.v4.mocktail_v4_dataset import (
    PharosMocktailV4Dataset,
)
from src.models.v4.pharos_mocktail_v4 import (
    PharosMocktailV4,
)


SEED = 42
N = 256
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-3


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


dataset = PharosMocktailV4Dataset(
    split="train",
    seed=SEED,
)

rng = np.random.default_rng(SEED)

indices = rng.choice(
    len(dataset),
    size=N,
    replace=False,
)

subset = Subset(
    dataset,
    indices.tolist(),
)

loader = DataLoader(
    subset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)


model = PharosMocktailV4(
    expression_dim=dataset.expression_dim,
    cnv_dim=dataset.cnv_dim,
    mutation_dim=dataset.mutation_dim,
    crispr_dim=dataset.crispr_dim,

    # Disable dropout for the memorization test
    dropout=0.0,
).to(device)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=0.0,
)

criterion = nn.MSELoss()


def gpu(batch, key):
    return batch[key].to(
        device,
        non_blocking=True,
    )


for epoch in range(1, EPOCHS + 1):

    model.train()

    squared_error = 0.0
    count = 0

    for batch in loader:

        optimizer.zero_grad(
            set_to_none=True
        )

        pred = model(
            gpu(batch, "drug_a_fp"),
            gpu(batch, "drug_a_clamp"),

            gpu(batch, "drug_b_fp"),
            gpu(batch, "drug_b_clamp"),

            gpu(batch, "expression"),
            gpu(batch, "cnv"),
            gpu(batch, "mutation"),
            gpu(batch, "crispr"),

            gpu(batch, "omics_mask"),

            gpu(batch, "resistance_expr"),
            gpu(batch, "core29_available"),

            gpu(batch, "drug_a_drmref_expr"),
            gpu(batch, "drug_b_drmref_expr"),

            gpu(batch, "drug_a_drmref_available"),
            gpu(batch, "drug_b_drmref_available"),
        )

        target = gpu(
            batch,
            "label",
        )

        loss = criterion(
            pred,
            target,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            5.0,
        )

        optimizer.step()

        squared_error += (
            (pred.detach() - target)
            .pow(2)
            .sum()
            .item()
        )

        count += target.numel()

    rmse = (
        squared_error / count
    ) ** 0.5

    if (
        epoch == 1
        or epoch % 5 == 0
    ):
        print(
            f"Epoch {epoch:03d} "
            f"| RMSE {rmse:.6f}"
        )


print("DONE")
