import torch

from src.models.pharos import PharosModel


def main():
    batch_size = 4

    drug_fp = torch.randn(batch_size, 2048)
    cell_expr = torch.randn(batch_size, 19176)
    resistance_expr = torch.randn(batch_size, 29)

    model = PharosModel(
        drug_dim=2048,
        cell_dim=19176,
        resistance_dim=29,
    )

    output = model(drug_fp, cell_expr, resistance_expr)

    print("Output shape:", output.shape)


if __name__ == "__main__":
    main()