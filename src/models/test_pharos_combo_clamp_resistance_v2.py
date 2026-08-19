import torch

from src.models.pharos_combo_clamp_resistance_v2 import (
    PharosComboCLAMPResistanceV2,
)


def main():

    print(
        "=" * 72
    )
    print(
        "TESTING PHAROS-COMBO RESISTANCE V2 MODEL"
    )
    print(
        "=" * 72
    )

    torch.manual_seed(42)

    batch_size = 4

    model = (
        PharosComboCLAMPResistanceV2()
    )

    model.eval()

    drug_a_fp = torch.randn(
        batch_size,
        2048,
    )

    drug_a_clamp = torch.randn(
        batch_size,
        768,
    )

    drug_b_fp = torch.randn(
        batch_size,
        2048,
    )

    drug_b_clamp = torch.randn(
        batch_size,
        768,
    )

    cell_expr = torch.randn(
        batch_size,
        19176,
    )

    resistance_expr = torch.randn(
        batch_size,
        29,
    )

    drug_a_scdrugact_expr = (
        torch.randn(
            batch_size,
            2399,
        )
    )

    drug_b_scdrugact_expr = (
        torch.randn(
            batch_size,
            2399,
        )
    )

    # row 0: both
    # row 1: only A
    # row 2: only B
    # row 3: neither
    drug_a_available = torch.tensor(
        [
            [1.0],
            [1.0],
            [0.0],
            [0.0],
        ]
    )

    drug_b_available = torch.tensor(
        [
            [1.0],
            [0.0],
            [1.0],
            [0.0],
        ]
    )

    drug_a_scdrugact_expr = (
        drug_a_scdrugact_expr
        * drug_a_available
    )

    drug_b_scdrugact_expr = (
        drug_b_scdrugact_expr
        * drug_b_available
    )

    with torch.no_grad():

        details = model(
            drug_a_fp,
            drug_a_clamp,
            drug_b_fp,
            drug_b_clamp,
            cell_expr,
            resistance_expr,
            drug_a_scdrugact_expr,
            drug_b_scdrugact_expr,
            drug_a_available,
            drug_b_available,
            return_details=True,
        )

    prediction = (
        details[
            "prediction"
        ]
    )

    print(
        "\nPrediction shape:",
        tuple(
            prediction.shape
        )
    )

    print(
        "Pair embedding:",
        tuple(
            details[
                "z_pair"
            ].shape
        )
    )

    print(
        "Cell embedding:",
        tuple(
            details[
                "z_cell"
            ].shape
        )
    )

    print(
        "Core resistance embedding:",
        tuple(
            details[
                "z_core_resistance"
            ].shape
        )
    )

    print(
        "scDrugAct resistance embedding:",
        tuple(
            details[
                "z_scdrugact_resistance"
            ].shape
        )
    )

    print(
        "Fused resistance embedding:",
        tuple(
            details[
                "z_resistance"
            ].shape
        )
    )

    assert prediction.shape == (
        batch_size,
        1,
    )

    assert details[
        "z_pair"
    ].shape == (
        batch_size,
        256,
    )

    assert details[
        "z_cell"
    ].shape == (
        batch_size,
        256,
    )

    assert details[
        "z_core_resistance"
    ].shape == (
        batch_size,
        64,
    )

    assert details[
        "z_scdrugact_resistance"
    ].shape == (
        batch_size,
        64,
    )

    assert details[
        "z_resistance"
    ].shape == (
        batch_size,
        64,
    )

    if not torch.isfinite(
        prediction
    ).all():
        raise AssertionError(
            "Prediction contains NaN/Inf."
        )

    # Neither-matched row must produce exactly zero
    # database-derived resistance embedding.
    if torch.count_nonzero(
        details[
            "z_scdrugact_resistance"
        ][3]
    ):
        raise AssertionError(
            "Neither-matched row must have "
            "zero scDrugAct resistance embedding."
        )

    if torch.count_nonzero(
        details[
            "z_scdrugact_a"
        ][2]
    ):
        raise AssertionError(
            "Unavailable Drug A embedding "
            "should be zero."
        )

    if torch.count_nonzero(
        details[
            "z_scdrugact_b"
        ][1]
    ):
        raise AssertionError(
            "Unavailable Drug B embedding "
            "should be zero."
        )

    print(
        "\nDatabase availability:"
    )

    print(
        "  pair_available:",
        details[
            "scdrugact_pair_available"
        ]
        .squeeze(-1)
        .tolist()
    )

    print(
        "  availability_sum:",
        details[
            "scdrugact_availability_sum"
        ]
        .squeeze(-1)
        .tolist()
    )

    print(
        "  both_available:",
        details[
            "scdrugact_both_available"
        ]
        .squeeze(-1)
        .tolist()
    )

    # Permutation invariance
    with torch.no_grad():

        pred_ab = model(
            drug_a_fp,
            drug_a_clamp,
            drug_b_fp,
            drug_b_clamp,
            cell_expr,
            resistance_expr,
            drug_a_scdrugact_expr,
            drug_b_scdrugact_expr,
            drug_a_available,
            drug_b_available,
        )

        pred_ba = model(
            drug_b_fp,
            drug_b_clamp,
            drug_a_fp,
            drug_a_clamp,
            cell_expr,
            resistance_expr,
            drug_b_scdrugact_expr,
            drug_a_scdrugact_expr,
            drug_b_available,
            drug_a_available,
        )

    max_swap_diff = torch.max(
        torch.abs(
            pred_ab
            - pred_ba
        )
    ).item()

    print(
        "\nMax A/B swap prediction difference:",
        max_swap_diff
    )

    if max_swap_diff > 1e-6:
        raise AssertionError(
            "Model is not permutation invariant."
        )

    print(
        "\n========================================"
    )
    print(
        "PASS: PHAROS-COMBO RESISTANCE V2 MODEL"
    )
    print(
        "========================================"
    )


if __name__ == "__main__":
    main()
