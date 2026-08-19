import torch

from src.models.pharos_combo_clamp_resistance_drmref import (
    PharosComboCLAMPResistanceDRMref,
)


def main():

    print(
        "=" * 72
    )

    print(
        "TESTING PHAROS-COMBO DRMref RESISTANCE V3"
    )

    print(
        "=" * 72
    )

    torch.manual_seed(
        42
    )

    batch_size = 4

    model = (
        PharosComboCLAMPResistanceDRMref()
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

    drug_a_drmref_expr = torch.randn(
        batch_size,
        6481,
    )

    drug_b_drmref_expr = torch.randn(
        batch_size,
        6481,
    )

    # Row 0: both
    # Row 1: only A
    # Row 2: only B
    # Row 3: neither
    a_available = torch.tensor(
        [
            [1.0],
            [1.0],
            [0.0],
            [0.0],
        ]
    )

    b_available = torch.tensor(
        [
            [1.0],
            [0.0],
            [1.0],
            [0.0],
        ]
    )

    # Match real dataset behavior.
    drug_a_drmref_expr = (
        drug_a_drmref_expr
        * a_available
    )

    drug_b_drmref_expr = (
        drug_b_drmref_expr
        * b_available
    )

    with torch.no_grad():

        details = model(
            drug_a_fp=drug_a_fp,
            drug_a_clamp=drug_a_clamp,
            drug_b_fp=drug_b_fp,
            drug_b_clamp=drug_b_clamp,
            cell_expr=cell_expr,
            resistance_expr=resistance_expr,
            drug_a_drmref_expr=drug_a_drmref_expr,
            drug_b_drmref_expr=drug_b_drmref_expr,
            drug_a_drmref_available=a_available,
            drug_b_drmref_available=b_available,
            return_details=True,
        )

    print(
        "\nPrediction:",
        tuple(
            details[
                "prediction"
            ].shape
        ),
    )

    print(
        "Pair:",
        tuple(
            details[
                "z_pair"
            ].shape
        ),
    )

    print(
        "Core resistance:",
        tuple(
            details[
                "z_core_resistance"
            ].shape
        ),
    )

    print(
        "DRMref resistance:",
        tuple(
            details[
                "z_drmref_resistance"
            ].shape
        ),
    )

    print(
        "Final resistance:",
        tuple(
            details[
                "z_resistance"
            ].shape
        ),
    )

    print(
        "DRMref alpha:",
        details[
            "drmref_alpha"
        ]
        .squeeze(-1)
        .tolist(),
    )

    assert details[
        "prediction"
    ].shape == (
        batch_size,
        1,
    )

    assert details[
        "z_core_resistance"
    ].shape == (
        batch_size,
        64,
    )

    assert details[
        "z_drmref_resistance"
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

    # --------------------------------------------------------
    # Critical property:
    # neither-matched row preserves V1 resistance exactly.
    # --------------------------------------------------------

    neither_idx = 3

    if not torch.equal(
        details[
            "z_resistance"
        ][
            neither_idx
        ],
        details[
            "z_core_resistance"
        ][
            neither_idx
        ],
    ):
        raise AssertionError(
            "Neither-matched row must preserve "
            "the core V1 resistance embedding exactly."
        )

    if details[
        "drmref_alpha"
    ][
        neither_idx
    ].item() != 0.0:
        raise AssertionError(
            "Neither-matched DRMref alpha must be exactly zero."
        )

    if torch.count_nonzero(
        details[
            "z_drmref_a"
        ][
            2
        ]
    ):
        raise AssertionError(
            "Unavailable Drug A DRMref embedding "
            "must be zero."
        )

    if torch.count_nonzero(
        details[
            "z_drmref_b"
        ][
            1
        ]
    ):
        raise AssertionError(
            "Unavailable Drug B DRMref embedding "
            "must be zero."
        )

    # Initial DRMref correction should be small but non-zero
    # where evidence is available.
    available_alpha = details[
        "drmref_alpha"
    ][
        :3
    ]

    if not torch.all(
        available_alpha
        > 0
    ):
        raise AssertionError(
            "Matched rows should start with a "
            "small positive DRMref gate."
        )

    if not torch.all(
        available_alpha
        < 0.25
    ):
        raise AssertionError(
            "Initial DRMref gate should be conservative."
        )

    # --------------------------------------------------------
    # Permutation invariance
    # --------------------------------------------------------

    with torch.no_grad():

        pred_ab = model(
            drug_a_fp=drug_a_fp,
            drug_a_clamp=drug_a_clamp,
            drug_b_fp=drug_b_fp,
            drug_b_clamp=drug_b_clamp,
            cell_expr=cell_expr,
            resistance_expr=resistance_expr,
            drug_a_drmref_expr=drug_a_drmref_expr,
            drug_b_drmref_expr=drug_b_drmref_expr,
            drug_a_drmref_available=a_available,
            drug_b_drmref_available=b_available,
        )

        pred_ba = model(
            drug_a_fp=drug_b_fp,
            drug_a_clamp=drug_b_clamp,
            drug_b_fp=drug_a_fp,
            drug_b_clamp=drug_a_clamp,
            cell_expr=cell_expr,
            resistance_expr=resistance_expr,
            drug_a_drmref_expr=drug_b_drmref_expr,
            drug_b_drmref_expr=drug_a_drmref_expr,
            drug_a_drmref_available=b_available,
            drug_b_drmref_available=a_available,
        )

    max_diff = torch.max(
        torch.abs(
            pred_ab
            - pred_ba
        )
    ).item()

    print(
        "\nMax A/B swap difference:",
        max_diff,
    )

    if max_diff > 1e-6:
        raise AssertionError(
            "DRMref V3 model is not permutation invariant."
        )

    print(
        "\n========================================"
    )

    print(
        "PASS: DRMref RESISTANCE V3 MODEL"
    )

    print(
        "========================================"
    )

    print(
        "V1 core pathway preserved when DRMref unavailable"
    )

    print(
        "DRMref residual gate: working"
    )

    print(
        "Permutation invariance: PASS"
    )


if __name__ == "__main__":
    main()
