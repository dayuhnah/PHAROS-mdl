from pathlib import Path
from urllib.parse import urljoin
import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup


BASE = "https://cellknowledge.com.cn/cellresponse/"

EVIDENCE_PATH = Path(
    "outputs/v6/cellresdb_expansion/"
    "cellresdb_evidence_context_dataset_drug.csv"
)

OUT_DIR = Path(
    "outputs/v6/cellresdb_expansion"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SESSION = requests.Session()

SESSION.headers.update({
    "User-Agent":
        "PHAROS-FYP-CellResDB-Response-Audit/1.0"
})


# ============================================================
# HELPERS
# ============================================================

def clean(value):

    if value is None:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value),
    ).strip()


def norm(value):

    return clean(
        value
    ).lower()


def detail_url(dataset_id):

    return urljoin(
        BASE,
        (
            "php_mysql/"
            "datasetDetail.php"
            f"?id={dataset_id}"
        ),
    )


def fetch(dataset_id):

    url = detail_url(
        dataset_id
    )

    response = SESSION.get(
        url,
        timeout=60,
    )

    response.raise_for_status()

    return (
        url,
        response.text,
    )


def find_sample_table(soup):
    """
    Find the CellResDB sample-information table.

    We require headers resembling:
        Sample ID
        Patient ID
        Sampling
        Response
    """

    for table in soup.find_all(
        "table"
    ):

        first_row = table.find(
            "tr"
        )

        if first_row is None:
            continue

        headers = [
            clean(
                x.get_text(
                    " ",
                    strip=True,
                )
            )
            for x in first_row.find_all(
                [
                    "th",
                    "td",
                ]
            )
        ]

        header_norm = [
            norm(
                x
            )
            for x in headers
        ]

        required = {
            "sample id",
            "patient id",
            "sampling",
            "response",
        }

        if required.issubset(
            set(
                header_norm
            )
        ):
            return (
                table,
                headers,
            )

    return (
        None,
        None,
    )


def parse_sample_table(
    table,
    headers,
):

    if table is None:
        return pd.DataFrame()

    # Ensure duplicate/blank headers are unique.
    safe_headers = []

    seen = {}

    for idx, header in enumerate(
        headers
    ):

        name = clean(
            header
        )

        if not name:
            name = (
                f"unnamed_{idx}"
            )

        base = name

        count = seen.get(
            base,
            0,
        )

        if count:
            name = (
                f"{base}_{count}"
            )

        seen[
            base
        ] = count + 1

        safe_headers.append(
            name
        )


    rows = []

    all_tr = table.find_all(
        "tr"
    )

    for tr in all_tr[1:]:

        cells = [
            clean(
                x.get_text(
                    " ",
                    strip=True,
                )
            )
            for x in tr.find_all(
                [
                    "td",
                    "th",
                ]
            )
        ]

        if not cells:
            continue

        # Pad or trim to match headers.
        if len(
            cells
        ) < len(
            safe_headers
        ):
            cells += (
                [""]
                * (
                    len(
                        safe_headers
                    )
                    - len(
                        cells
                    )
                )
            )

        if len(
            cells
        ) > len(
            safe_headers
        ):
            cells = cells[
                :len(
                    safe_headers
                )
            ]

        rows.append(
            dict(
                zip(
                    safe_headers,
                    cells,
                )
            )
        )

    return pd.DataFrame(
        rows
    )


def find_column(
    df,
    target,
):

    target = target.lower()

    for column in df.columns:

        if (
            clean(
                column
            ).lower()
            ==
            target
        ):
            return column

    return None


def valid_values(series):

    invalid = {
        "",
        "-",
        "na",
        "n/a",
        "nan",
        "none",
        "unknown",
        "not available",
    }

    result = []

    for value in series:

        value = clean(
            value
        )

        if (
            value.lower()
            not in invalid
        ):
            result.append(
                value
            )

    return result


def response_class(value):

    value = norm(
        value
    )

    # CellResDB commonly uses R / NR.
    if value == "r":
        return "R"

    if value == "nr":
        return "NR"

    if value in {
        "responder",
        "responders",
    }:
        return "R"

    if value in {
        "non-responder",
        "non responder",
        "nonresponder",
        "non-responders",
        "non responders",
        "nonresponders",
    }:
        return "NR"

    # Preserve other meaningful response schemes.
    return clean(
        value
    )


def extract_rds_url(
    soup,
    page_url,
):

    candidates = []

    for a in soup.find_all(
        "a",
        href=True,
    ):

        href = urljoin(
            page_url,
            a[
                "href"
            ],
        )

        text = norm(
            a.get_text(
                " ",
                strip=True,
            )
        )

        if (
            ".rds" in href.lower()
            or
            "download expression matrix"
            in text
        ):
            candidates.append(
                href
            )

    candidates = list(
        dict.fromkeys(
            candidates
        )
    )

    if not candidates:
        return ""

    return candidates[0]


# ============================================================
# LOAD EXISTING EVIDENCE MAP
# ============================================================

evidence = pd.read_csv(
    EVIDENCE_PATH
)

dataset_ids = sorted(
    evidence[
        "dataset_id"
    ]
    .dropna()
    .astype(str)
    .unique()
)


print("=" * 100)
print("CELLRESDB RESPONSE / RESISTANCE ELIGIBILITY AUDIT")
print("=" * 100)

print(
    "Matched dataset × drug rows:",
    len(
        evidence
    )
)

print(
    "Unique matched datasets:",
    len(
        dataset_ids
    )
)


# ============================================================
# AUDIT EACH UNIQUE DATASET
# ============================================================

dataset_records = []

sample_records = []


for i, dataset_id in enumerate(
    dataset_ids,
    start=1,
):

    print()
    print(
        f"[{i}/{len(dataset_ids)}] "
        f"{dataset_id}"
    )


    try:

        page_url, html = fetch(
            dataset_id
        )

    except Exception as exc:

        print(
            "  ERROR:",
            exc
        )

        dataset_records.append({
            "dataset_id":
                dataset_id,

            "fetch_ok":
                False,

            "error":
                str(
                    exc
                ),
        })

        continue


    soup = BeautifulSoup(
        html,
        "html.parser",
    )


    table, headers = (
        find_sample_table(
            soup
        )
    )


    samples = parse_sample_table(
        table,
        headers,
    )


    if samples.empty:

        print(
            "  WARNING: sample table not found"
        )

        dataset_records.append({
            "dataset_id":
                dataset_id,

            "fetch_ok":
                True,

            "sample_table_found":
                False,

            "sample_rows":
                0,

            "patient_count":
                0,

            "has_response":
                False,

            "response_class_count":
                0,

            "has_R":
                False,

            "has_NR":
                False,

            "has_R_and_NR":
                False,

            "has_pre":
                False,

            "has_post":
                False,

            "has_pre_and_post":
                False,

            "rds_url":
                extract_rds_url(
                    soup,
                    page_url,
                ),

            "error":
                None,
        })

        continue


    patient_col = find_column(
        samples,
        "Patient ID",
    )

    sample_col = find_column(
        samples,
        "Sample ID",
    )

    sampling_col = find_column(
        samples,
        "Sampling",
    )

    response_col = find_column(
        samples,
        "Response",
    )


    # --------------------------------------------------------
    # Patient count
    # --------------------------------------------------------

    if patient_col:

        patient_values = (
            valid_values(
                samples[
                    patient_col
                ]
            )
        )

        patient_count = len(
            set(
                patient_values
            )
        )

    else:

        patient_count = 0


    # --------------------------------------------------------
    # Sampling
    # --------------------------------------------------------

    if sampling_col:

        sampling_values = (
            valid_values(
                samples[
                    sampling_col
                ]
            )
        )

    else:

        sampling_values = []


    sampling_norm = {
        norm(
            x
        )
        for x in sampling_values
    }


    has_pre = any(
        (
            x == "pre"
            or
            x.startswith(
                "pre "
            )
            or
            "pre-treatment"
            in x
            or
            "pretreatment"
            in x
        )
        for x in sampling_norm
    )


    has_post = any(
        (
            x == "post"
            or
            x.startswith(
                "post "
            )
            or
            "post-treatment"
            in x
            or
            "posttreatment"
            in x
        )
        for x in sampling_norm
    )


    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    if response_col:

        raw_response_values = (
            valid_values(
                samples[
                    response_col
                ]
            )
        )

    else:

        raw_response_values = []


    mapped_responses = [
        response_class(
            x
        )
        for x in raw_response_values
    ]


    response_classes = sorted(
        set(
            mapped_responses
        )
    )


    has_R = (
        "R"
        in response_classes
    )

    has_NR = (
        "NR"
        in response_classes
    )

    has_R_and_NR = (
        has_R
        and
        has_NR
    )


    response_counts = (
        pd.Series(
            mapped_responses,
            dtype="object",
        )
        .value_counts()
        .to_dict()
    )


    print(
        "  Samples:",
        len(
            samples
        )
    )

    print(
        "  Patients:",
        patient_count
    )

    print(
        "  Response classes:",
        response_classes
    )

    print(
        "  Sampling:",
        sorted(
            sampling_norm
        )
    )

    print(
        "  R + NR:",
        has_R_and_NR
    )


    rds_url = extract_rds_url(
        soup,
        page_url,
    )


    dataset_records.append({
        "dataset_id":
            dataset_id,

        "fetch_ok":
            True,

        "sample_table_found":
            True,

        "sample_rows":
            len(
                samples
            ),

        "patient_count":
            patient_count,

        "has_response":
            bool(
                response_classes
            ),

        "response_classes":
            ";".join(
                response_classes
            ),

        "response_class_count":
            len(
                response_classes
            ),

        "response_counts":
            ";".join(
                f"{k}:{v}"
                for k, v
                in sorted(
                    response_counts.items()
                )
            ),

        "has_R":
            has_R,

        "has_NR":
            has_NR,

        "has_R_and_NR":
            has_R_and_NR,

        "sampling_values":
            ";".join(
                sorted(
                    sampling_norm
                )
            ),

        "has_pre":
            has_pre,

        "has_post":
            has_post,

        "has_pre_and_post":
            (
                has_pre
                and
                has_post
            ),

        "rds_url":
            rds_url,

        "detail_url":
            page_url,

        "error":
            None,
    })


    # Keep sample-level metadata too.
    for _, sample in samples.iterrows():

        sample_records.append({
            "dataset_id":
                dataset_id,

            "sample_id":
                (
                    sample.get(
                        sample_col,
                        "",
                    )
                    if sample_col
                    else ""
                ),

            "patient_id":
                (
                    sample.get(
                        patient_col,
                        "",
                    )
                    if patient_col
                    else ""
                ),

            "sampling":
                (
                    sample.get(
                        sampling_col,
                        "",
                    )
                    if sampling_col
                    else ""
                ),

            "response":
                (
                    sample.get(
                        response_col,
                        "",
                    )
                    if response_col
                    else ""
                ),
        })


    time.sleep(
        0.1
    )


dataset_audit = pd.DataFrame(
    dataset_records
)

sample_audit = pd.DataFrame(
    sample_records
)


# ============================================================
# JOIN BACK TO DATASET × DRUG EVIDENCE
# ============================================================

full = evidence.merge(
    dataset_audit,
    on="dataset_id",
    how="left",
)


# ============================================================
# EVIDENCE QUALITY GRADING
# ============================================================

def classify(row):

    context = str(
        row.get(
            "evidence_context",
            ""
        )
    )

    has_two_classes = bool(
        row.get(
            "has_R_and_NR",
            False,
        )
    )

    has_response = bool(
        row.get(
            "has_response",
            False,
        )
    )

    has_prepost = bool(
        row.get(
            "has_pre_and_post",
            False,
        )
    )


    if (
        context
        ==
        "direct_monotherapy"
        and
        has_two_classes
    ):
        return (
            "A_direct_monotherapy_response"
        )


    if (
        context
        ==
        "combination_context"
        and
        has_two_classes
    ):
        return (
            "B_contextual_response"
        )


    if has_response:
        return (
            "C_response_single_class_or_nonstandard"
        )


    if has_prepost:
        return (
            "D_prepost_only"
        )


    return (
        "E_insufficient_response_metadata"
    )


full[
    "eligibility_grade"
] = full.apply(
    classify,
    axis=1,
)


full[
    "signature_eligible"
] = full[
    "eligibility_grade"
].isin(
    [
        "A_direct_monotherapy_response",
        "B_contextual_response",
    ]
)


# ============================================================
# OUTPUTS
# ============================================================

dataset_path = (
    OUT_DIR
    / "cellresdb_response_dataset_audit.csv"
)

sample_path = (
    OUT_DIR
    / "cellresdb_response_sample_metadata.csv"
)

full_path = (
    OUT_DIR
    / "cellresdb_response_eligibility_dataset_drug.csv"
)


dataset_audit.to_csv(
    dataset_path,
    index=False,
)

sample_audit.to_csv(
    sample_path,
    index=False,
)

full.to_csv(
    full_path,
    index=False,
)


# ============================================================
# DOWNLOAD MANIFEST
# ============================================================

eligible_datasets = (
    full[
        full[
            "signature_eligible"
        ]
    ][
        [
            "dataset_id",
            "eligibility_grade",
            "rds_url",
        ]
    ]
    .drop_duplicates(
        subset=[
            "dataset_id"
        ]
    )
    .sort_values(
        [
            "eligibility_grade",
            "dataset_id",
        ]
    )
)


manifest_path = (
    OUT_DIR
    / "cellresdb_rds_candidate_manifest.csv"
)

eligible_datasets.to_csv(
    manifest_path,
    index=False,
)


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 100)
print("RESPONSE ELIGIBILITY SUMMARY")
print("=" * 100)


print(
    "Matched unique datasets:",
    full[
        "dataset_id"
    ].nunique()
)

print(
    "Datasets with sample table:",
    dataset_audit[
        "sample_table_found"
    ].fillna(
        False
    ).sum()
)

print(
    "Datasets containing R and NR:",
    dataset_audit[
        "has_R_and_NR"
    ].fillna(
        False
    ).sum()
)

print(
    "Datasets containing pre + post:",
    dataset_audit[
        "has_pre_and_post"
    ].fillna(
        False
    ).sum()
)


print()
print("=" * 100)
print("ELIGIBILITY GRADE COUNTS — DATASET × DRUG")
print("=" * 100)

print(
    full[
        "eligibility_grade"
    ]
    .value_counts()
    .to_string()
)


print()
print("=" * 100)
print("DIRECT MONOTHERAPY RESPONSE EVIDENCE")
print("=" * 100)

direct = full[
    full[
        "eligibility_grade"
    ]
    ==
    "A_direct_monotherapy_response"
]


if len(
    direct
):

    print(
        direct[
            [
                "dataset_id",
                "cellres_drug",
                "cancer_type",
                "patient_count",
                "sample_rows",
                "response_classes",
                "response_counts",
                "sampling_values",
            ]
        ]
        .to_string(
            index=False
        )
    )

else:

    print(
        "None"
    )


print()
print("=" * 100)
print("CONTEXTUAL RESPONSE EVIDENCE")
print("=" * 100)

contextual = full[
    full[
        "eligibility_grade"
    ]
    ==
    "B_contextual_response"
]


if len(
    contextual
):

    print(
        contextual[
            [
                "dataset_id",
                "cellres_drug",
                "cancer_type",
                "therapeutic_regimen",
                "patient_count",
                "response_classes",
                "response_counts",
            ]
        ]
        .sort_values(
            [
                "dataset_id",
                "cellres_drug",
            ]
        )
        .to_string(
            index=False
        )
    )

else:

    print(
        "None"
    )


print()
print("=" * 100)
print("CANDIDATE RDS DOWNLOADS")
print("=" * 100)

print(
    "Eligible unique datasets:",
    eligible_datasets[
        "dataset_id"
    ].nunique()
)

print(
    "With discovered RDS URL:",
    eligible_datasets[
        "rds_url"
    ]
    .fillna("")
    .str.strip()
    .ne("")
    .sum()
)

print()

print(
    eligible_datasets.to_string(
        index=False
    )
)


print()
print("=" * 100)
print("SAVED")
print("=" * 100)

print(dataset_path)
print(sample_path)
print(full_path)
print(manifest_path)
