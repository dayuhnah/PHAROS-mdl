import torch

from src.models.pharos_rx import PharosRXModel

def main():
    batch_size = 4

    drug_dim = 2048
    cell_dim = 19176
    resistance_dim = 29

    model = PharosRXModel(
        drug_dim=drug_dim,
        cell_dim=cell_dim,
        resistance_dim=resistance_dim,
    )

    drug_fp = torch.randn(batch_size, drug_dim)
    cell_expr = torch.randn(batch_size, cell_dim)
    resistance_expr = torch.randn(batch_size, resistance_dim)

    output = model(
        drug_fp,
        cell_expr,
        resistance_expr
    )

    print("Drug input:", drug_fp.shape)
    print("Cell input:", cell_expr.shape)
    print("Resistance input:", resistance_expr.shape)
    print("Prediction:", output.shape)

if __name__ == "__main__":
    main()