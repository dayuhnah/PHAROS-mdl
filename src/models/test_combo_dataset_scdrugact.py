import torch

from src.data.combo_dataset_scdrugact import (
    PharosComboScDrugActDataset,
)


def assert_finite(
    tensor,
    name,
):
    if not torch.isfinite(
        tensor
    ).all():
        raise AssertionError(
            f"{name} contains NaN/Inf."
        )


def main():

    print(
        "=" * 72
    )
    print(
        "TESTING PHAROS-COMBO SCDRUGACT DATASET"
    )
    print(
        "=" * 72
    )

    dataset = (
        PharosComboScDrugActDataset()
    )

    print(
        f"\nDataset rows: "
        f"{len(dataset):,}"
    )

    # --------------------------------------------------------
    # Find representative examples:
    #   1) both drugs matched
    #   2) exactly one matched
    #   3) neither matched
    # --------------------------------------------------------

    example_indices = {
        "both": None,
        "one": None,
        "neither": None,
    }

    for idx, (
        drug_a,
        drug_b,
    ) in enumerate(
        zip(
            dataset.drug_a_names,
            dataset.drug_b_names,
        )
    ):

        a = (
            dataset
            .scdrugact_available_cache[
                drug_a
            ]
        )

        b = (
            dataset
            .scdrugact_available_cache[
                drug_b
            ]
        )

        if (
            a
            and b
            and example_indices[
                "both"
            ]
            is None
        ):
            example_indices[
                "both"
            ] = idx

        elif (
            a != b
            and example_indices[
                "one"
            ]
            is None
        ):
            example_indices[
                "one"
            ] = idx

        elif (
            not a
            and not b
            and example_indices[
                "neither"
            ]
            is None
        ):
            example_indices[
                "neither"
            ] = idx

        if all(
            value is not None
            for value
            in example_indices.values()
        ):
            break

    print(
        "\nRepresentative indices:",
        example_indices,
    )

    if any(
        value is None
        for value
        in example_indices.values()
    ):
        raise AssertionError(
            "Could not find all three expected "
            "coverage categories."
        )

    expected_sc_dim = (
        dataset.scdrugact_gene_dim
    )

    # --------------------------------------------------------
    # Validate all three cases
    # --------------------------------------------------------

    for category, idx in (
        example_indices.items()
    ):

        sample = dataset[
            idx
        ]

        print(
            "\n----------------------------------------"
        )
        print(
            f"CASE: {category.upper()}"
        )
        print(
            "----------------------------------------"
        )

        print(
            "Drug A:",
            sample[
                "drug_a"
            ],
        )

        print(
            "Drug B:",
            sample[
                "drug_b"
            ],
        )

        print(
            "Cell:",
            sample[
                "depmap_id"
            ],
        )

        print(
            "cell_expr:",
            tuple(
                sample[
                    "cell_expr"
                ].shape
            ),
        )

        print(
            "resistance_expr:",
            tuple(
                sample[
                    "resistance_expr"
                ].shape
            ),
        )

        print(
            "drug_a_scdrugact_expr:",
            tuple(
                sample[
                    "drug_a_scdrugact_expr"
                ].shape
            ),
        )

        print(
            "drug_b_scdrugact_expr:",
            tuple(
                sample[
                    "drug_b_scdrugact_expr"
                ].shape
            ),
        )

        print(
            "A available:",
            int(
                sample[
                    "drug_a_scdrugact_available"
                ].item()
            ),
            "genes:",
            int(
                sample[
                    "drug_a_scdrugact_gene_count"
                ].item()
            ),
        )

        print(
            "B available:",
            int(
                sample[
                    "drug_b_scdrugact_available"
                ].item()
            ),
            "genes:",
            int(
                sample[
                    "drug_b_scdrugact_gene_count"
                ].item()
            ),
        )

        assert (
            sample[
                "drug_a_scdrugact_expr"
            ].shape
            == (
                expected_sc_dim,
            )
        )

        assert (
            sample[
                "drug_b_scdrugact_expr"
            ].shape
            == (
                expected_sc_dim,
            )
        )

        assert (
            sample[
                "drug_a_scdrugact_mask"
            ].shape
            == (
                expected_sc_dim,
            )
        )

        assert (
            sample[
                "drug_b_scdrugact_mask"
            ].shape
            == (
                expected_sc_dim,
            )
        )

        assert_finite(
            sample[
                "drug_a_scdrugact_expr"
            ],
            "drug_a_scdrugact_expr",
        )

        assert_finite(
            sample[
                "drug_b_scdrugact_expr"
            ],
            "drug_b_scdrugact_expr",
        )

        # Masked-out positions MUST be zero.
        a_outside_mask = (
            sample[
                "drug_a_scdrugact_expr"
            ][
                sample[
                    "drug_a_scdrugact_mask"
                ]
                == 0
            ]
        )

        b_outside_mask = (
            sample[
                "drug_b_scdrugact_expr"
            ][
                sample[
                    "drug_b_scdrugact_mask"
                ]
                == 0
            ]
        )

        if torch.count_nonzero(
            a_outside_mask
        ):
            raise AssertionError(
                "Drug A expression is non-zero "
                "outside its scDrugAct mask."
            )

        if torch.count_nonzero(
            b_outside_mask
        ):
            raise AssertionError(
                "Drug B expression is non-zero "
                "outside its scDrugAct mask."
            )

        # Availability flag and mask count must agree.
        a_count = int(
            torch.count_nonzero(
                sample[
                    "drug_a_scdrugact_mask"
                ]
            ).item()
        )

        b_count = int(
            torch.count_nonzero(
                sample[
                    "drug_b_scdrugact_mask"
                ]
            ).item()
        )

        if a_count != int(
            sample[
                "drug_a_scdrugact_gene_count"
            ].item()
        ):
            raise AssertionError(
                "Drug A gene-count mismatch."
            )

        if b_count != int(
            sample[
                "drug_b_scdrugact_gene_count"
            ].item()
        ):
            raise AssertionError(
                "Drug B gene-count mismatch."
            )

        if (
            (a_count > 0)
            != bool(
                sample[
                    "drug_a_scdrugact_available"
                ].item()
            )
        ):
            raise AssertionError(
                "Drug A availability mismatch."
            )

        if (
            (b_count > 0)
            != bool(
                sample[
                    "drug_b_scdrugact_available"
                ].item()
            )
        ):
            raise AssertionError(
                "Drug B availability mismatch."
            )

        if category == "neither":

            if torch.count_nonzero(
                sample[
                    "drug_a_scdrugact_expr"
                ]
            ):
                raise AssertionError(
                    "Unmatched Drug A should have "
                    "an all-zero scDrugAct vector."
                )

            if torch.count_nonzero(
                sample[
                    "drug_b_scdrugact_expr"
                ]
            ):
                raise AssertionError(
                    "Unmatched Drug B should have "
                    "an all-zero scDrugAct vector."
                )

    print(
        "\n========================================"
    )
    print(
        "PASS: SCDRUGACT DATASET INTEGRATION"
    )
    print(
        "========================================"
    )

    print(
        "Core resistance branch: retained"
    )

    print(
        "Drug-specific scDrugAct branch: aligned"
    )

    print(
        f"scDrugAct input dimension: "
        f"{dataset.scdrugact_gene_dim}"
    )


if __name__ == "__main__":
    main()
