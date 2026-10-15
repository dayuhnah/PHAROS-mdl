from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch


INPUT = Path(
    "data/processed/v5/cellresdb/resistance_bank_cohort/"
    "cellresdb_cohort_resistance_bank.csv.gz"
)

OUT_DIR = Path(
    "data/processed/v5/cellresdb/resistance_bank_cohort/model_cache"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(INPUT)

# ---------------------------------------------------------
# Define biological tokens
#
# One token =
# independent patient cohort × cell type
#
# Each token contains 50 Hallmark pathways.
# ---------------------------------------------------------

tokens = (
    df[["cohort_id", "cell_type"]]
    .drop_duplicates()
    .sort_values(["cohort_id", "cell_type"])
    .reset_index(drop=True)
)

pathways = sorted(df["pathway"].unique())

cohorts = sorted(tokens["cohort_id"].unique())
cell_types = sorted(tokens["cell_type"].unique())

assert len(pathways) == 50, (
    f"Expected 50 Hallmark pathways, got {len(pathways)}"
)

cohort_to_id = {
    name: i for i, name in enumerate(cohorts)
}

celltype_to_id = {
    name: i for i, name in enumerate(cell_types)
}

pathway_to_id = {
    name: i for i, name in enumerate(pathways)
}


# ---------------------------------------------------------
# Pathway feature vector
# ---------------------------------------------------------

feature_names = [
    "resistance_score",
    "confidence",
    "median_logFC",
    "direction_agreement",
    "resistance_score_sd",
    "log1p_n_R",
    "log1p_n_NR",
    "available",
    "has_direct_monotherapy",
    "has_combination_context",
]

T = len(tokens)
P = len(pathways)
F = len(feature_names)

features = np.zeros(
    (T, P, F),
    dtype=np.float32
)

pathway_mask = np.zeros(
    (T, P),
    dtype=bool
)

cohort_ids = np.zeros(T, dtype=np.int64)
cell_type_ids = np.zeros(T, dtype=np.int64)


for token_idx, token in tokens.iterrows():

    cohort = token["cohort_id"]
    cell_type = token["cell_type"]

    cohort_ids[token_idx] = cohort_to_id[cohort]
    cell_type_ids[token_idx] = celltype_to_id[cell_type]

    g = df[
        (df["cohort_id"] == cohort)
        & (df["cell_type"] == cell_type)
    ].copy()

    if len(g) != 50:
        raise RuntimeError(
            f"{cohort} / {cell_type}: "
            f"expected 50 pathways, found {len(g)}"
        )

    g = g.set_index("pathway")

    for pathway in pathways:

        row = g.loc[pathway]

        p = pathway_to_id[pathway]

        values = [
            row["resistance_score"],
            row["confidence"],
            row["median_logFC"],
            row["direction_agreement"],
            row["resistance_score_sd"],
            np.log1p(row["n_R"]),
            np.log1p(row["n_NR"]),
            row["available"],
            row["has_direct_monotherapy"],
            row["has_combination_context"],
        ]

        values = np.asarray(
            values,
            dtype=np.float32
        )

        values = np.nan_to_num(
            values,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        features[token_idx, p] = values

        pathway_mask[token_idx, p] = (
            float(row["available"]) > 0
        )


# ---------------------------------------------------------
# Standardize continuous features only.
#
# Binary/evidence flags stay as 0/1.
# ---------------------------------------------------------

continuous_idx = [
    0,  # resistance_score
    1,  # confidence
    2,  # median_logFC
    4,  # resistance_score_sd
    5,  # log R
    6,  # log NR
]

flat = features.reshape(-1, F)

means = flat[:, continuous_idx].mean(axis=0)
stds = flat[:, continuous_idx].std(axis=0)

stds[stds < 1e-8] = 1.0

features[:, :, continuous_idx] = (
    features[:, :, continuous_idx]
    - means
) / stds


# ---------------------------------------------------------
# Save tensors
# ---------------------------------------------------------

tensor_file = OUT_DIR / "cellresdb_token_tensors.pt"

torch.save(
    {
        "features": torch.from_numpy(features),
        "pathway_mask": torch.from_numpy(pathway_mask),
        "cohort_ids": torch.from_numpy(cohort_ids),
        "cell_type_ids": torch.from_numpy(cell_type_ids),
        "pathway_ids": torch.arange(
            P,
            dtype=torch.long
        ),
    },
    tensor_file
)


# ---------------------------------------------------------
# Save human-readable metadata separately
# ---------------------------------------------------------

metadata = {
    "feature_names": feature_names,
    "continuous_feature_indices": continuous_idx,
    "continuous_means": means.tolist(),
    "continuous_stds": stds.tolist(),

    "cohorts": cohorts,
    "cell_types": cell_types,
    "pathways": pathways,

    "tokens": [
        {
            "token_id": int(i),
            "cohort_id": row["cohort_id"],
            "cell_type": row["cell_type"],
        }
        for i, row in tokens.iterrows()
    ],
}

metadata_file = (
    OUT_DIR / "cellresdb_token_metadata.json"
)

with open(metadata_file, "w") as f:
    json.dump(
        metadata,
        f,
        indent=2
    )


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("=" * 60)
print("CELLRESDB V5 TOKEN CACHE")
print("=" * 60)

print("Feature tensor:", features.shape)
print("Expected:       (20, 50, 10)")

print()
print("Tokens:", T)
print("Pathways:", P)
print("Features/pathway:", F)
print("Independent cohorts:", len(cohorts))
print("Unique cell types:", len(cell_types))

print()
print(
    "Available pathway measurements:",
    int(pathway_mask.sum()),
    "/",
    pathway_mask.size,
)

print()
print("Tensor cache:", tensor_file)
print("Metadata:", metadata_file)

print()
print("DONE")
