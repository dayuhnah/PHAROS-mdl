from src.data.graph_dataset import PharosGraphDataset


def main():
    dataset = PharosGraphDataset(max_rows=1000)

    print(f"Length: {len(dataset)}")

    sample = dataset[0]

    print("x:", sample.x.shape)
    print("edge_index:", sample.edge_index.shape)
    print("edge_attr:", sample.edge_attr.shape)
    print("cell_expr:", sample.cell_expr.shape)
    print("resistance_expr:", sample.resistance_expr.shape)
    print("y:", sample.y.shape)


if __name__ == "__main__":
    main()