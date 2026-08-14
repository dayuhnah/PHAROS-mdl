from src.data.dataset import PharosDepMapDataset


def main():
    dataset = PharosDepMapDataset(
        max_rows=1000,
        drug_embedding_cache_path="data/processed/pharos_drug_embeddings_chemberta.npz",
        use_drug_embeddings=True,
    )

    sample = dataset[0]

    print("drug_fp:", sample["drug_fp"].shape)
    print("drug_embedding:", sample["drug_embedding"].shape)
    print("cell_expr:", sample["cell_expr"].shape)
    print("resistance_expr:", sample["resistance_expr"].shape)
    print("label:", sample["label"].shape)


if __name__ == "__main__":
    main()