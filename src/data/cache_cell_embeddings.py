from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


PROCESSED_DIR = Path("data/processed")
EXPRESSION_PATH = PROCESSED_DIR / "pharos_depmap_expression.parquet"
OUTPUT_PATH = PROCESSED_DIR / "pharos_cell_embeddings_pca.npz"


def clean_gene_name(column_name: str) -> str:
    return column_name.split(" (")[0]


def main():
    print("Loading expression matrix...")
    expression = pd.read_parquet(EXPRESSION_PATH)

    expression = expression.copy()
    expression.columns = [
        "depmap_id" if col == "depmap_id" else clean_gene_name(col)
        for col in expression.columns
    ]

    depmap_ids = expression["depmap_id"].values
    gene_matrix = expression.drop(columns=["depmap_id"]).values.astype(np.float32)

    print(f"Expression matrix shape: {gene_matrix.shape}")

    print("Standardizing expression...")
    scaler = StandardScaler()
    gene_matrix_scaled = scaler.fit_transform(gene_matrix)

    print("Fitting PCA cell embeddings...")
    pca = PCA(n_components=256, random_state=42)
    embeddings = pca.fit_transform(gene_matrix_scaled).astype(np.float32)

    explained = pca.explained_variance_ratio_.sum()

    np.savez_compressed(
        OUTPUT_PATH,
        depmap_ids=depmap_ids,
        embeddings=embeddings,
        explained_variance=explained,
    )

    print(f"Saved cell embeddings to: {OUTPUT_PATH}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Explained variance: {explained:.4f}")


if __name__ == "__main__":
    main()