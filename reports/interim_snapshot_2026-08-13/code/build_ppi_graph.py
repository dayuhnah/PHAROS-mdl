import argparse
from pathlib import Path

import pandas as pd
import torch

from src.data.dataset import (
    RESISTANCE_GENES,
    clean_gene_name,
)


def main(
    top_k: int = 2000,
    min_score: int = 700,
):
    expression_path = Path(
        "data/processed/pharos_depmap_expression.parquet"
    )

    info_path = Path(
        "data/raw/string/"
        "9606.protein.info.v12.0.txt"
    )

    links_path = Path(
        "data/raw/string/"
        "9606.protein.physical.links.v12.0.txt"
    )

    output_path = Path(
        "data/processed/pharos_ppi_graph.pt"
    )

    # =====================================
    # Check files
    # =====================================

    for path in [
        expression_path,
        info_path,
        links_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing file: {path}"
            )

    # =====================================
    # Expression genes
    # =====================================

    print("Loading expression matrix...")

    expression = pd.read_parquet(
        expression_path
    )

    gene_df = expression.drop(
        columns=["depmap_id"],
        errors="ignore",
    ).copy()

    gene_df.columns = [
        clean_gene_name(col)
        for col in gene_df.columns
    ]

    # Remove duplicate names after cleaning
    gene_df = gene_df.loc[
        :,
        ~gene_df.columns.duplicated(),
    ]

    # Keep resistance genes separate
    candidate_genes = [
        gene
        for gene in gene_df.columns
        if gene not in RESISTANCE_GENES
    ]

    print(
        f"Non-resistance expression genes: "
        f"{len(candidate_genes)}"
    )

    # =====================================
    # STRING protein -> gene mapping
    # =====================================

    print("Loading STRING protein information...")

    info = pd.read_csv(
        info_path,
        sep="\t",
        compression="infer",
    )

    if "#string_protein_id" in info.columns:
        string_id_col = "#string_protein_id"
    elif "string_protein_id" in info.columns:
        string_id_col = "string_protein_id"
    else:
        raise KeyError(
            "Could not find STRING protein ID column."
        )

    if "preferred_name" not in info.columns:
        raise KeyError(
            "Could not find preferred_name column."
        )

    info = info[
        [
            string_id_col,
            "preferred_name",
        ]
    ].dropna()

    # Avoid duplicate gene symbols
    info = info.drop_duplicates(
        subset=["preferred_name"]
    )

    string_gene_names = set(
        info["preferred_name"]
        .astype(str)
    )

    candidate_genes = [
        gene
        for gene in candidate_genes
        if gene in string_gene_names
    ]

    print(
        f"Expression genes mapped to STRING: "
        f"{len(candidate_genes)}"
    )

    # =====================================
    # Select variable genes
    # =====================================

    print(
        f"Selecting top {top_k} variable genes..."
    )

    variance = (
        gene_df[candidate_genes]
        .var(axis=0)
        .sort_values(
            ascending=False
        )
    )

    selected_genes = (
        variance
        .head(top_k)
        .index
        .tolist()
    )

    print(
        f"Selected genes: "
        f"{len(selected_genes)}"
    )

    # =====================================
    # Gene -> STRING ID
    # =====================================

    gene_to_string = dict(
        zip(
            info["preferred_name"]
            .astype(str),
            info[string_id_col]
            .astype(str),
        )
    )

    string_to_gene = {
        string_id: gene
        for gene, string_id
        in gene_to_string.items()
    }

    selected_string_ids = {
        gene_to_string[gene]
        for gene in selected_genes
        if gene in gene_to_string
    }

    # =====================================
    # Load physical interactions
    # =====================================

    print(
        "Loading STRING physical interactions..."
    )

    links = pd.read_csv(
        links_path,
        sep=r"\s+",
        compression="infer",
    )

    required_cols = {
        "protein1",
        "protein2",
        "combined_score",
    }

    missing = (
        required_cols
        - set(links.columns)
    )

    if missing:
        raise KeyError(
            f"Missing STRING columns: {missing}"
        )

    # Confidence threshold
    links = links[
        links["combined_score"]
        >= min_score
    ]

    # Restrict to selected genes
    links = links[
        links["protein1"].isin(
            selected_string_ids
        )
        &
        links["protein2"].isin(
            selected_string_ids
        )
    ].copy()

    print(
        f"Interactions after filtering: "
        f"{len(links)}"
    )

    # =====================================
    # Convert STRING IDs -> gene symbols
    # =====================================

    links["gene1"] = (
        links["protein1"]
        .map(string_to_gene)
    )

    links["gene2"] = (
        links["protein2"]
        .map(string_to_gene)
    )

    links = links.dropna(
        subset=[
            "gene1",
            "gene2",
        ]
    )

    links = links[
        links["gene1"]
        != links["gene2"]
    ]

    # =====================================
    # Remove genes disconnected by threshold
    # =====================================

    connected_genes = (
        set(links["gene1"])
        | set(links["gene2"])
    )

    final_genes = [
        gene
        for gene in selected_genes
        if gene in connected_genes
    ]

    gene_to_idx = {
        gene: idx
        for idx, gene
        in enumerate(final_genes)
    }

    print(
        f"Connected genes: "
        f"{len(final_genes)}"
    )

    # =====================================
    # Build undirected edge map
    # =====================================

    edge_scores = {}

    for row in links.itertuples(
        index=False
    ):
        gene1 = row.gene1
        gene2 = row.gene2

        if (
            gene1 not in gene_to_idx
            or gene2 not in gene_to_idx
        ):
            continue

        i = gene_to_idx[gene1]
        j = gene_to_idx[gene2]

        score = (
            float(row.combined_score)
            / 1000.0
        )

        # Add both directions
        for edge in [
            (i, j),
            (j, i),
        ]:
            # If duplicate, retain strongest score
            previous = edge_scores.get(
                edge,
                0.0,
            )

            edge_scores[edge] = max(
                previous,
                score,
            )

    edges = list(
        edge_scores.keys()
    )

    weights = [
        edge_scores[edge]
        for edge in edges
    ]

    edge_index = torch.tensor(
        edges,
        dtype=torch.long,
    ).t().contiguous()

    edge_weight = torch.tensor(
        weights,
        dtype=torch.float32,
    )

    print(
        f"Directed edges: "
        f"{edge_index.shape[1]}"
    )

    print(
        f"Edge index: "
        f"{edge_index.shape}"
    )

    print(
        f"Edge weights: "
        f"{edge_weight.shape}"
    )

    # =====================================
    # Save
    # =====================================

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "genes":
                final_genes,

            "gene_to_idx":
                gene_to_idx,

            "edge_index":
                edge_index,

            "edge_weight":
                edge_weight,

            "num_genes":
                len(final_genes),

            "top_k_requested":
                top_k,

            "min_score":
                min_score,

            "ppi_source":
                "STRING physical network",
        },
        output_path,
    )

    print(
        f"\nSaved PPI graph to: "
        f"{output_path}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--top-k",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--min-score",
        type=int,
        default=700,
    )

    args = parser.parse_args()

    main(
        top_k=args.top_k,
        min_score=args.min_score,
    )