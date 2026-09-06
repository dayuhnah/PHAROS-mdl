from pathlib import Path
import pandas as pd

DATA = Path(
    "data/processed/v4/pharos_mocktail_v4_multiomics.parquet"
)

BASE = Path(
    "data/processed/v4/multiomics"
)

cells = sorted(
    pd.read_parquet(DATA)["depmap_id"]
    .astype(str)
    .unique()
)

modalities = [
    "expression",
    "cnv",
    "mutation",
    "crispr",
]

masks = pd.DataFrame(index=cells)


for name in modalities:

    df = pd.read_parquet(
        BASE / f"{name}.parquet"
    )

    df.index = df.index.astype(str)

    available = set(df.index)

    masks[f"has_{name}"] = [
        cell in available
        for cell in cells
    ]

    # Reindex to all 93 V4 cells.
    #
    # IMPORTANT:
    # Keep NaN values here.
    # Imputation will be fitted using TRAINING DATA ONLY.
    aligned = df.reindex(cells)

    aligned.index.name = "ModelID"

    aligned.to_parquet(
        BASE / f"{name}_aligned.parquet"
    )

    print(
        f"{name}:",
        aligned.shape,
        "available:",
        int(masks[f"has_{name}"].sum()),
        "NaN:",
        f"{aligned.isna().mean().mean()*100:.2f}%"
    )


masks.index.name = "ModelID"

masks.to_parquet(
    BASE / "modality_masks.parquet"
)

print("Masks:", masks.shape)
print("DONE")
