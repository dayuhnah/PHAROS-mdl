import torch

from src.data.combo_dataset_drmref import (
    PharosComboDRMrefDataset,
)


def _available(
    item,
    key,
):
    return bool(
        item[
            key
        ].item()
        > 0.5
    )


def find_representative_cases(
    dataset,
):
    cases = {
        "both":
            None,

        "one":
            None,

        "neither":
            None,
    }

    for idx in range(
        len(dataset)
    ):
        item = dataset[
            idx
        ]

        a = _available(
            item,
            "drug_a_drmref_available",
        )

        b = _available(
            item,
            "drug_b_drmref_available",
        )

        if (
            a
            and b
            and cases[
                "both"
            ]
            is None
        ):
            cases[
                "both"
            ] = idx

        elif (
            a != b
            and cases[
                "one"
            ]
            is None
        ):
            cases[
                "one"
            ] = idx

        elif (
            not a
            and not b
            and cases[
                "neither"
            ]
            is None
        ):
            cases[
                "neither"
            ] = idx

        if all(
            value is not None
            for value
            in cases.values()
        ):
            break

    return cases


def validate_case(
    dataset,
    idx,
    name,
):
    item = dataset[
        idx
    ]

    dim = (
        dataset.drmref_gene_dim
    )

    a_available = _available(
        item,
        "drug_a_drmref_available",
    )

    b_available = _available(
        item,
        "drug_b_drmref_available",
    )

    a_count = int(
        item[
            "drug_a_drmref_gene_count"
        ].item()
    )

    b_count = int(
        item[
            "drug_b_drmref_gene_count"
        ].item()
    )

    print(
        "\n----------------------------------------"
    )

    print(
        f"CASE: {name.upper()}"
    )

    print(
        "----------------------------------------"
    )

    print(
        "Drug A:",
        item[
            "drug_a"
        ],
    )

    print(
        "Drug B:",
        item[
            "drug_b"
        ],
    )

    print(
        "Cell:",
        item[
            "depmap_id"
        ],
    )

    print(
        "cell_expr:",
        tuple(
            item[
                "cell_expr"
            ].shape
        ),
    )

    print(
        "resistance_expr:",
        tuple(
            item[
                "resistance_expr"
            ].shape
        ),
    )

    print(
        "drug_a_drmref_expr:",
        tuple(
            item[
                "drug_a_drmref_expr"
            ].shape
        ),
    )

    print(
        "drug_b_drmref_expr:",
        tuple(
            item[
                "drug_b_drmref_expr"
            ].shape
        ),
    )

    print(
        "A available:",
        int(
            a_available
        ),
        "genes:",
        a_count,
    )

    print(
        "B available:",
        int(
            b_available
        ),
        "genes:",
        b_count,
    )

    expected = (
        dim,
    )

    for key in [
        "drug_a_drmref_expr",
        "drug_b_drmref_expr",
        "drug_a_drmref_weights",
        "drug_b_drmref_weights",
        "drug_a_drmref_mask",
        "drug_b_drmref_mask",
    ]:
        if item[
            key
        ].shape != expected:
            raise AssertionError(
                f"{key} shape is "
                f"{tuple(item[key].shape)}, "
                f"expected {expected}."
            )

    for key in [
        "drug_a_drmref_expr",
        "drug_b_drmref_expr",
        "drug_a_drmref_weights",
        "drug_b_drmref_weights",
    ]:
        if not torch.isfinite(
            item[
                key
            ]
        ).all():
            raise AssertionError(
                f"{key} contains NaN/Inf."
            )

    if (
        a_available
        != (
            a_count > 0
        )
    ):
        raise AssertionError(
            "Drug A availability/count mismatch."
        )

    if (
        b_available
        != (
            b_count > 0
        )
    ):
        raise AssertionError(
            "Drug B availability/count mismatch."
        )

    for prefix, available in [
        (
            "drug_a",
            a_available,
        ),
        (
            "drug_b",
            b_available,
        ),
    ]:
        weights = item[
            f"{prefix}_drmref_weights"
        ]

        mask = item[
            f"{prefix}_drmref_mask"
        ]

        weighted_expr = item[
            f"{prefix}_drmref_expr"
        ]

        outside = (
            mask == 0
        )

        if torch.count_nonzero(
            weights[
                outside
            ]
        ):
            raise AssertionError(
                f"{prefix}: weights non-zero "
                "outside DRMref mask."
            )

        if torch.count_nonzero(
            weighted_expr[
                outside
            ]
        ):
            raise AssertionError(
                f"{prefix}: weighted expression non-zero "
                "outside DRMref mask."
            )

        if not available:
            if torch.count_nonzero(
                weights
            ):
                raise AssertionError(
                    f"{prefix}: unmatched weights "
                    "must be zero."
                )

            if torch.count_nonzero(
                weighted_expr
            ):
                raise AssertionError(
                    f"{prefix}: unmatched DRMref "
                    "expression must be zero."
                )

            if torch.count_nonzero(
                mask
            ):
                raise AssertionError(
                    f"{prefix}: unmatched mask "
                    "must be zero."
                )

        else:
            if not torch.count_nonzero(
                weights
            ):
                raise AssertionError(
                    f"{prefix}: available DRMref "
                    "signature is empty."
                )

            positive = int(
                torch.any(
                    weights > 0
                ).item()
            )

            negative = int(
                torch.any(
                    weights < 0
                ).item()
            )

            print(
                f"{prefix} signed DRMref weights: "
                f"positive={positive}, "
                f"negative={negative}"
            )


def main():

    print(
        "=" * 72
    )

    print(
        "TESTING PHAROS-COMBO DRMref DATASET"
    )

    print(
        "=" * 72
    )

    dataset = (
        PharosComboDRMrefDataset()
    )

    print(
        f"\nDataset rows: "
        f"{len(dataset):,}"
    )

    print(
        f"DRMref input dimension: "
        f"{dataset.drmref_gene_dim:,}"
    )

    if (
        dataset.drmref_gene_dim
        != 6481
    ):
        raise AssertionError(
            "Expected 6,481 DRMref/DepMap genes "
            f"from the current preprocessing run, got "
            f"{dataset.drmref_gene_dim:,}."
        )

    cases = (
        find_representative_cases(
            dataset
        )
    )

    print(
        "\nRepresentative indices:",
        cases,
    )

    missing = [
        name
        for name, idx
        in cases.items()
        if idx is None
    ]

    if missing:
        raise AssertionError(
            "Missing representative case(s): "
            + ", ".join(
                missing
            )
        )

    for name, idx in cases.items():
        validate_case(
            dataset,
            idx,
            name,
        )

    print(
        "\n========================================"
    )

    print(
        "PASS: DRMref DATASET INTEGRATION"
    )

    print(
        "========================================"
    )

    print(
        "Core resistance branch: retained"
    )

    print(
        "Real resistant-cell DRMref branch: aligned"
    )

    print(
        f"DRMref input dimension: "
        f"{dataset.drmref_gene_dim}"
    )


if __name__ == "__main__":
    main()
