import torch

from src.models.v5.cellresdb_resistance_encoder import (
    CellResDBResistanceEncoder,
)


CACHE = (
    "data/processed/v5/cellresdb/"
    "resistance_bank_cohort/model_cache/"
    "cellresdb_token_tensors.pt"
)


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


model = CellResDBResistanceEncoder(
    cache_path=CACHE,
    drug_pair_dim=256,
    hidden_dim=128,
    out_dim=64,
    num_heads=4,
).to(device)


B = 4

fake_pair = torch.randn(
    B,
    256,
    device=device,
)


out = model(
    fake_pair,
    return_attention=True,
)


print("=" * 60)
print("CELLRESDB ENCODER SMOKE TEST")
print("=" * 60)

print(
    "Device:",
    device,
)

print(
    "Resistance embedding:",
    out["embedding"].shape,
)

print(
    "Token attention:",
    out["token_attention"].shape,
)

print(
    "Pathway attention:",
    out["pathway_attention"].shape,
)

print()
print(
    "Token attention sums:",
    out["token_attention"].sum(dim=1),
)

print(
    "Pathway attention sums:",
    out["pathway_attention"].sum(dim=1),
)

loss = out["embedding"].pow(2).mean()

loss.backward()

has_grad = any(
    p.grad is not None
    for p in model.parameters()
    if p.requires_grad
)

print()
print(
    "Backward pass:",
    "PASS" if has_grad else "FAIL"
)

print("DONE")
