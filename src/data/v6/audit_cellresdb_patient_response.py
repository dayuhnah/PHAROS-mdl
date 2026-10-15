from pathlib import Path
import pandas as pd


SAMPLE_PATH = Path(
    "outputs/v6/cellresdb_expansion/"
    "cellresdb_response_sample_metadata.csv"
)

ELIGIBILITY_PATH = Path(
    "outputs/v6/cellresdb_expansion/"
    "cellresdb_response_eligibility_dataset_drug.csv"
)

OUT_DIR = Path(
    "outputs/v6/cellresdb_expansion"
)


def clean(x):
    if pd.isna(x):
        return ""

    return str(x).strip()


def response_class(x):
    x = clean(x).lower()

    if x == "r":
        return "R"

    if x == "nr":
        return "NR"

    return None


samples = pd.read_csv(
    SAMPLE_PATH
)

eligibility = pd.read_csv(
    ELIGIBILITY_PATH
)


samples[
    "response_class"
] = samples[
    "response"
].map(
    response_class
)


samples = samples[
    samples[
        "response_class"
    ].notna()
].copy()


records = []


for dataset_id, group in samples.groupby(
    "dataset_id"
):

    patient_map = {}

    for patient_id, pg in group.groupby(
        "patient_id"
    ):

        patient_id = clean(
            patient_id
        )

        if not patient_id:
            continue

        labels = sorted(
            set(
                pg[
                    "response_class"
                ]
                .dropna()
                .astype(str)
            )
        )

        patient_map[
            patient_id
        ] = labels


    r_only = 0
    nr_only = 0
    mixed = 0

    for labels in patient_map.values():

        if labels == [
            "R"
        ]:
            r_only += 1

        elif labels == [
            "NR"
        ]:
            nr_only += 1

        elif (
            "R" in labels
            and
            "NR" in labels
        ):
            mixed += 1


    clean_rnr_patients = (
        r_only
        +
        nr_only
    )


    records.append({
        "dataset_id":
            dataset_id,

        "response_labelled_patients":
            len(
                patient_map
            ),

        "R_only_patients":
            r_only,

        "NR_only_patients":
            nr_only,

        "mixed_R_NR_patients":
            mixed,

        "clean_R_NR_patients":
            clean_rnr_patients,

        "has_independent_R_and_NR":
            (
                r_only > 0
                and
                nr_only > 0
            ),

        "min_class_patients":
            min(
                r_only,
                nr_only,
            ),
    })


audit = pd.DataFrame(
    records
)


eligible_datasets = (
    eligibility[
        eligibility[
            "signature_eligible"
        ].astype(bool)
    ][
        [
            "dataset_id",
            "eligibility_grade",
        ]
    ]
    .drop_duplicates()
)


audit = eligible_datasets.merge(
    audit,
    on="dataset_id",
    how="left",
)


audit[
    "patient_quality"
] = "weak"


audit.loc[
    (
        audit[
            "has_independent_R_and_NR"
        ]
        &
        (
            audit[
                "min_class_patients"
            ]
            >= 2
        )
    ),
    "patient_quality",
] = "usable"


audit.loc[
    (
        audit[
            "has_independent_R_and_NR"
        ]
        &
        (
            audit[
                "min_class_patients"
            ]
            >= 5
        )
    ),
    "patient_quality",
] = "strong"


audit = audit.sort_values(
    [
        "patient_quality",
        "min_class_patients",
    ],
    ascending=[
        True,
        False,
    ],
)


out_path = (
    OUT_DIR
    / "cellresdb_patient_response_quality.csv"
)

audit.to_csv(
    out_path,
    index=False,
)


print("=" * 100)
print("CELLRESDB PATIENT-LEVEL RESPONSE AUDIT")
print("=" * 100)

print(
    audit.to_string(
        index=False
    )
)

print()
print("=" * 100)
print("QUALITY COUNTS")
print("=" * 100)

print(
    audit[
        "patient_quality"
    ]
    .value_counts()
    .to_string()
)

print()
print(
    "Datasets with independent R and NR patients:",
    int(
        audit[
            "has_independent_R_and_NR"
        ]
        .fillna(False)
        .sum()
    )
)

print()
print("Saved:")
print(out_path)
