from pathlib import Path
import pandas as pd


QUALITY_PATH = Path(
    "outputs/v6/cellresdb_expansion/"
    "cellresdb_patient_response_quality.csv"
)

EVIDENCE_PATH = Path(
    "outputs/v6/cellresdb_expansion/"
    "cellresdb_response_eligibility_dataset_drug.csv"
)

MAP_PATH = Path(
    "outputs/v6/cellresdb_expansion/"
    "cellresdb_pharos_drug_overlap_audited.csv"
)

RESPONSE_PATH = Path(
    "data/processed/v4/"
    "pharos_mocktail_v4_multiomics.parquet"
)

OUT_DIR = Path(
    "outputs/v6/cellresdb_expansion"
)


def norm_key(x):
    if pd.isna(x):
        return ""

    return str(x).strip().upper()


def parse_bool(x):
    return str(x).strip().lower() in {
        "true",
        "1",
        "yes",
    }


quality = pd.read_csv(
    QUALITY_PATH
)

evidence = pd.read_csv(
    EVIDENCE_PATH
)

mapping = pd.read_csv(
    MAP_PATH
)

response = pd.read_parquet(
    RESPONSE_PATH,
    columns=[
        "drug_a_key",
        "drug_b_key",
    ],
)


mapping["approved"] = (
    mapping["approved"]
    .map(parse_bool)
)

mapping = mapping[
    mapping["approved"]
].copy()


# ============================================================
# JOIN PATIENT QUALITY TO DATASET × DRUG EVIDENCE
# ============================================================

qualified = evidence.merge(
    quality[
        [
            "dataset_id",
            "patient_quality",
            "R_only_patients",
            "NR_only_patients",
            "mixed_R_NR_patients",
            "min_class_patients",
            "has_independent_R_and_NR",
        ]
    ],
    on="dataset_id",
    how="left",
)


qualified = qualified[
    qualified["signature_eligible"]
    .astype(bool)
].copy()


# ============================================================
# MAP PHAROS KEYS TO EACH DRUG
# ============================================================

drug_to_keys = {}

for _, row in mapping.iterrows():

    drug = str(
        row["cellres_drug"]
    )

    keys = {
        norm_key(x)
        for x in str(
            row["approved_pharos_keys"]
        ).split(";")
        if str(x).strip()
    }

    drug_to_keys[
        drug
    ] = keys


# ============================================================
# DETERMINE WHICH DRUGS HAVE EACH QUALITY LEVEL
# ============================================================

def drugs_for_quality(
    allowed,
):

    datasets = set(
        quality.loc[
            quality["patient_quality"]
            .isin(allowed),
            "dataset_id",
        ].astype(str)
    )

    return set(
        qualified.loc[
            qualified["dataset_id"]
            .astype(str)
            .isin(datasets),
            "cellres_drug",
        ].astype(str)
    )


strong_drugs = drugs_for_quality(
    {"strong"}
)

strong_usable_drugs = drugs_for_quality(
    {
        "strong",
        "usable",
    }
)

all_rnr_drugs = set(
    qualified[
        "cellres_drug"
    ].astype(str)
)


def keys_for_drugs(
    drugs,
):

    result = set()

    for drug in drugs:
        result.update(
            drug_to_keys.get(
                drug,
                set(),
            )
        )

    return result


strong_keys = keys_for_drugs(
    strong_drugs
)

strong_usable_keys = keys_for_drugs(
    strong_usable_drugs
)

all_rnr_keys = keys_for_drugs(
    all_rnr_drugs
)


# ============================================================
# DRUGCOMB COVERAGE
# ============================================================

a = response[
    "drug_a_key"
].map(
    norm_key
)

b = response[
    "drug_b_key"
].map(
    norm_key
)


def coverage(
    keys,
):

    any_mask = (
        a.isin(keys)
        |
        b.isin(keys)
    )

    both_mask = (
        a.isin(keys)
        &
        b.isin(keys)
    )

    return {
        "rows_any":
            int(any_mask.sum()),

        "pct_any":
            100.0
            * any_mask.sum()
            / len(response),

        "rows_both":
            int(both_mask.sum()),

        "pct_both":
            100.0
            * both_mask.sum()
            / len(response),
    }


rows = []


for name, drugs, keys in [
    (
        "strong_only",
        strong_drugs,
        strong_keys,
    ),
    (
        "strong_plus_usable",
        strong_usable_drugs,
        strong_usable_keys,
    ),
    (
        "all_R_NR_signature_eligible",
        all_rnr_drugs,
        all_rnr_keys,
    ),
]:

    stats = coverage(
        keys
    )

    rows.append({
        "scheme":
            name,

        "drug_count":
            len(drugs),

        "pharos_key_count":
            len(keys),

        **stats,
    })


summary = pd.DataFrame(
    rows
)


# ============================================================
# PER-DRUG QUALITY
# ============================================================

drug_rows = []


for drug in sorted(
    all_rnr_drugs
):

    g = qualified[
        qualified[
            "cellres_drug"
        ].astype(str)
        ==
        drug
    ]

    qualities = set(
        g[
            "patient_quality"
        ]
        .dropna()
        .astype(str)
    )

    if "strong" in qualities:
        best_quality = "strong"

    elif "usable" in qualities:
        best_quality = "usable"

    else:
        best_quality = "weak"


    keys = drug_to_keys.get(
        drug,
        set(),
    )

    mask = (
        a.isin(keys)
        |
        b.isin(keys)
    )


    drug_rows.append({
        "cellres_drug":
            drug,

        "best_patient_quality":
            best_quality,

        "eligible_dataset_count":
            g[
                "dataset_id"
            ].nunique(),

        "eligible_datasets":
            ";".join(
                sorted(
                    set(
                        g[
                            "dataset_id"
                        ].astype(str)
                    )
                )
            ),

        "pharos_keys":
            ";".join(
                sorted(keys)
            ),

        "drugcomb_rows":
            int(
                mask.sum()
            ),

        "drugcomb_pct":
            100.0
            * mask.sum()
            / len(response),
    })


drug_summary = (
    pd.DataFrame(
        drug_rows
    )
    .sort_values(
        [
            "best_patient_quality",
            "drugcomb_rows",
        ],
        ascending=[
            True,
            False,
        ],
    )
)


# ============================================================
# BUILD DOWNLOAD MANIFESTS
# ============================================================

candidate_manifest = pd.read_csv(
    "outputs/v6/cellresdb_expansion/"
    "cellresdb_rds_candidate_manifest.csv"
)


tier1_ids = set(
    quality.loc[
        quality[
            "patient_quality"
        ].isin(
            [
                "strong",
                "usable",
            ]
        ),
        "dataset_id",
    ].astype(str)
)


tier2_ids = set(
    quality.loc[
        quality[
            "patient_quality"
        ]
        ==
        "weak",
        "dataset_id",
    ].astype(str)
)


tier1 = candidate_manifest[
    candidate_manifest[
        "dataset_id"
    ].astype(str)
    .isin(tier1_ids)
].copy()


tier2 = candidate_manifest[
    candidate_manifest[
        "dataset_id"
    ].astype(str)
    .isin(tier2_ids)
].copy()


tier1 = tier1.merge(
    quality[
        [
            "dataset_id",
            "patient_quality",
            "R_only_patients",
            "NR_only_patients",
            "mixed_R_NR_patients",
        ]
    ],
    on="dataset_id",
    how="left",
)


tier2 = tier2.merge(
    quality[
        [
            "dataset_id",
            "patient_quality",
            "R_only_patients",
            "NR_only_patients",
            "mixed_R_NR_patients",
        ]
    ],
    on="dataset_id",
    how="left",
)


# ============================================================
# SAVE
# ============================================================

summary_path = (
    OUT_DIR
    / "cellresdb_quality_coverage_summary.csv"
)

drug_path = (
    OUT_DIR
    / "cellresdb_quality_coverage_per_drug.csv"
)

tier1_path = (
    OUT_DIR
    / "cellresdb_rds_tier1_manifest.csv"
)

tier2_path = (
    OUT_DIR
    / "cellresdb_rds_tier2_weak_manifest.csv"
)


summary.to_csv(
    summary_path,
    index=False,
)

drug_summary.to_csv(
    drug_path,
    index=False,
)

tier1.to_csv(
    tier1_path,
    index=False,
)

tier2.to_csv(
    tier2_path,
    index=False,
)


# ============================================================
# REPORT
# ============================================================

print("=" * 100)
print("CELLRESDB QUALITY-AWARE COVERAGE")
print("=" * 100)

print(
    summary.to_string(
        index=False
    )
)


print()
print("=" * 100)
print("PER-DRUG BEST EVIDENCE QUALITY")
print("=" * 100)

print(
    drug_summary.to_string(
        index=False
    )
)


print()
print("=" * 100)
print("TIER 1 DOWNLOAD MANIFEST")
print("=" * 100)

print(
    "Datasets:",
    len(tier1)
)

print(
    tier1.to_string(
        index=False
    )
)


print()
print("=" * 100)
print("TIER 2 WEAK MANIFEST")
print("=" * 100)

print(
    "Datasets:",
    len(tier2)
)

print(
    tier2.to_string(
        index=False
    )
)


print()
print("=" * 100)
print("SAVED")
print("=" * 100)

print(summary_path)
print(drug_path)
print(tier1_path)
print(tier2_path)
