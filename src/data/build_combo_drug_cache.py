from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.data.featurize_smiles import (
    smiles_to_morgan_fp,
)

from src.data.featurize_graph import (
    smiles_to_graph,
)


# ============================================================
# Paths
# ============================================================

COMBO_PATH = Path(
    "data/processed/combination/"
    "pharos_combo_local_smiles.parquet"
)

OUTPUT_DIR = Path(
    "data/processed/combination"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FP_CACHE_PATH = (
    OUTPUT_DIR
    / "pharos_combo_drug_fingerprints.npz"
)

GRAPH_CACHE_PATH = (
    OUTPUT_DIR
    / "pharos_combo_drug_graphs.pt"
)

DRUG_TABLE_PATH = (
    OUTPUT_DIR
    / "pharos_combo_drug_table.csv"
)


# ============================================================
# Main
# ============================================================

def main():

    print(
        "========================================"
    )
    print(
        "PHAROS-COMBO DRUG CACHE BUILDER"
    )
    print(
        "========================================"
    )

    # --------------------------------------------------------
    # Load immediately usable combination dataset
    # --------------------------------------------------------

    combo = pd.read_parquet(
        COMBO_PATH
    )

    print(
        f"\nCombination rows: "
        f"{len(combo):,}"
    )

    # --------------------------------------------------------
    # Extract Drug A
    # --------------------------------------------------------

    drugs_a = (
        combo[
            [
                "drug_a",
                "smiles_a",
            ]
        ]
        .rename(
            columns={
                "drug_a":
                    "drug_name",

                "smiles_a":
                    "smiles",
            }
        )
    )

    # --------------------------------------------------------
    # Extract Drug B
    # --------------------------------------------------------

    drugs_b = (
        combo[
            [
                "drug_b",
                "smiles_b",
            ]
        ]
        .rename(
            columns={
                "drug_b":
                    "drug_name",

                "smiles_b":
                    "smiles",
            }
        )
    )

    # --------------------------------------------------------
    # Combine both sides
    # --------------------------------------------------------

    drugs = pd.concat(
        [
            drugs_a,
            drugs_b,
        ],
        ignore_index=True,
    )

    drugs = drugs.dropna(
        subset=[
            "drug_name",
            "smiles",
        ]
    ).copy()

    drugs[
        "drug_name"
    ] = (
        drugs[
            "drug_name"
        ]
        .astype(str)
        .str.strip()
    )

    drugs[
        "smiles"
    ] = (
        drugs[
            "smiles"
        ]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Check for name -> multiple SMILES conflicts
    # --------------------------------------------------------

    conflicts = (
        drugs
        .groupby(
            "drug_name"
        )[
            "smiles"
        ]
        .nunique()
    )

    conflicts = conflicts[
        conflicts > 1
    ]

    if len(conflicts) > 0:

        print(
            "\n🚨 Found drug names with "
            "multiple SMILES:"
        )

        print(
            conflicts.head(
                20
            )
        )

        raise ValueError(
            "Drug-name/SMILES conflicts "
            "must be resolved before caching."
        )

    # --------------------------------------------------------
    # One row per unique drug
    # --------------------------------------------------------

    drugs = (
        drugs
        .drop_duplicates(
            subset=[
                "drug_name",
            ]
        )
        .sort_values(
            "drug_name"
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"Unique usable drugs: "
        f"{len(drugs):,}"
    )

    # ========================================================
    # Build caches
    # ========================================================

    fingerprint_names = []

    fingerprints = []

    graph_cache = {}

    failed_graphs = []

    failed_fingerprints = []

    # --------------------------------------------------------
    # Featurize
    # --------------------------------------------------------

    for _, row in tqdm(
        drugs.iterrows(),
        total=len(drugs),
        desc="Featurizing combo drugs",
    ):

        drug_name = str(
            row["drug_name"]
        )

        smiles = str(
            row["smiles"]
        )

        # ----------------------------------------------------
        # Morgan fingerprint
        # ----------------------------------------------------

        fp = (
            smiles_to_morgan_fp(
                smiles,
                radius=2,
                n_bits=2048,
            )
        )

        # Existing PHAROS function returns all zeros
        # when parsing fails.
        if (
            fp is None
            or fp.shape != (2048,)
        ):

            failed_fingerprints.append(
                drug_name
            )

            continue

        # ----------------------------------------------------
        # Molecular graph
        # ----------------------------------------------------

        graph = (
            smiles_to_graph(
                smiles
            )
        )

        if graph is None:

            failed_graphs.append(
                drug_name
            )

            continue

        # ----------------------------------------------------
        # Safety checks
        # ----------------------------------------------------

        if graph.x.ndim != 2:

            raise ValueError(
                f"Invalid node tensor for "
                f"{drug_name}: "
                f"{graph.x.shape}"
            )

        if (
            graph.x.shape[1]
            != 18
        ):

            raise ValueError(
                f"Unexpected node feature "
                f"dimension for {drug_name}: "
                f"{graph.x.shape[1]}"
            )

        if (
            graph.edge_attr.ndim
            != 2
        ):

            raise ValueError(
                f"Invalid edge tensor for "
                f"{drug_name}: "
                f"{graph.edge_attr.shape}"
            )

        if (
            graph.edge_attr.shape[1]
            != 6
        ):

            raise ValueError(
                f"Unexpected edge feature "
                f"dimension for {drug_name}: "
                f"{graph.edge_attr.shape[1]}"
            )

        # ----------------------------------------------------
        # Add to caches only if BOTH representations work
        # ----------------------------------------------------

        fingerprint_names.append(
            drug_name
        )

        fingerprints.append(
            fp.astype(
                np.float32
            )
        )

        graph_cache[
            drug_name
        ] = graph

    # ========================================================
    # Convert fingerprint list
    # ========================================================

    if len(
        fingerprints
    ) == 0:

        raise RuntimeError(
            "No drug fingerprints were "
            "successfully generated."
        )

    fingerprint_matrix = np.stack(
        fingerprints
    ).astype(
        np.float32
    )

    drug_names_array = np.asarray(
        fingerprint_names,
        dtype=str,
    )

    # ========================================================
    # Cross-cache consistency
    # ========================================================

    graph_names = set(
        graph_cache.keys()
    )

    fp_names = set(
        fingerprint_names
    )

    if graph_names != fp_names:

        raise RuntimeError(
            "Fingerprint and graph caches "
            "contain different drugs."
        )

    # ========================================================
    # Save fingerprint cache
    # ========================================================

    np.savez_compressed(
        FP_CACHE_PATH,

        drug_names=(
            drug_names_array
        ),

        fingerprints=(
            fingerprint_matrix
        ),
    )

    # ========================================================
    # Save graph cache
    # ========================================================

    torch.save(
        graph_cache,
        GRAPH_CACHE_PATH,
    )

    # ========================================================
    # Save human-readable drug table
    # ========================================================

    successful_drugs = (
        drugs[
            drugs[
                "drug_name"
            ].isin(
                fingerprint_names
            )
        ]
        .copy()
    )

    successful_drugs.to_csv(
        DRUG_TABLE_PATH,
        index=False,
    )

    # ========================================================
    # Results
    # ========================================================

    print(
        "\n========================================"
    )
    print(
        "CACHE RESULTS"
    )
    print(
        "========================================"
    )

    print(
        f"Input drugs: "
        f"{len(drugs):,}"
    )

    print(
        f"Fingerprints saved: "
        f"{len(fingerprint_names):,}"
    )

    print(
        f"Graphs saved: "
        f"{len(graph_cache):,}"
    )

    print(
        f"Failed fingerprints: "
        f"{len(failed_fingerprints):,}"
    )

    print(
        f"Failed graphs: "
        f"{len(failed_graphs):,}"
    )

    print(
        "\nFingerprint matrix:"
    )

    print(
        fingerprint_matrix.shape
    )

    # --------------------------------------------------------
    # Example graph
    # --------------------------------------------------------

    first_name = (
        fingerprint_names[0]
    )

    first_graph = (
        graph_cache[
            first_name
        ]
    )

    print(
        f"\nExample drug: "
        f"{first_name}"
    )

    print(
        f"Nodes: "
        f"{first_graph.x.shape}"
    )

    print(
        f"Edges: "
        f"{first_graph.edge_index.shape}"
    )

    print(
        f"Edge attributes: "
        f"{first_graph.edge_attr.shape}"
    )

    # ========================================================
    # Save locations
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "SAVED"
    )

    print(
        "========================================"
    )

    print(
        f"Fingerprint cache:\n"
        f"{FP_CACHE_PATH}"
    )

    print(
        f"\nGraph cache:\n"
        f"{GRAPH_CACHE_PATH}"
    )

    print(
        f"\nDrug table:\n"
        f"{DRUG_TABLE_PATH}"
    )


if __name__ == "__main__":
    main()