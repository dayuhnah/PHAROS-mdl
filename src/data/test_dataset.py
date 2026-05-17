from src.data.dataset import PharosDepMapDataset


def main():
    dataset = PharosDepMapDataset(max_rows=1000)

    print(f"Length: {len(dataset)}")

    sample = dataset[0]

    print("drug_fp:", sample["drug_fp"].shape)
    print("cell_expr:", sample["cell_expr"].shape)
    print("resistance_expr:", sample["resistance_expr"].shape)
    print("label:", sample["label"], sample["label"].shape)


if __name__ == "__main__":
    main()