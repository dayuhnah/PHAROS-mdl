from pathlib import Path
from urllib.parse import urljoin

import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup


BASE = "https://cellknowledge.com.cn/cellresponse/"

HOME_URL = BASE

AJAX_URL = urljoin(
    BASE,
    "php_mysql/getTissueDataSet.php",
)

OUT_DIR = Path(
    "data/external/cellresdb_full/catalog"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SESSION = requests.Session()

SESSION.headers.update({
    "User-Agent":
        "PHAROS-FYP-CellResDB-Full-Catalog/1.0"
})


# ============================================================
# HELPERS
# ============================================================

def fetch_text(
    url,
    params=None,
):
    response = SESSION.get(
        url,
        params=params,
        timeout=60,
    )

    response.raise_for_status()

    return response.text


def fetch_json(
    url,
    params=None,
):
    response = SESSION.get(
        url,
        params=params,
        timeout=60,
    )

    response.raise_for_status()

    return response.json()


def clean(
    value,
):
    if value is None:
        return None

    value = re.sub(
        r"\s+",
        " ",
        str(value),
    ).strip()

    return value or None


def extract_dataset_map(
    html,
):
    """
    Extract the embedded CellResDB JavaScript variable:

        var datasetIDDiseaseMap =
            'GSE...___BCC;GSE...___BCC';

    Returns the raw encoded map string.
    """

    matches = re.findall(
        r"""
        datasetIDDiseaseMap
        \s*=\s*
        ['"]
        (.*?)
        ['"]
        """,
        html,
        flags=(
            re.I
            | re.S
            | re.X
        ),
    )

    if not matches:
        return None

    # Normally there is only one.
    # Join in case the site ever emits several.
    values = []

    for value in matches:
        value = value.strip()

        if value:
            values.append(
                value
            )

    if not values:
        return None

    return ";".join(
        values
    )


def extract_drugs_from_regimen(
    regimen,
):
    """
    Extract drug names + DrugBank IDs from strings such as:

        Pembrolizumab(DB09037) + Vismodegib(DB08828)

    Returns list of:
        (drug_name, drugbank_id)
    """

    if not regimen:
        return []

    matches = re.findall(
        r"""
        ([A-Za-z0-9][A-Za-z0-9 ./'_-]*?)
        \s*
        \(
        (DB\d+)
        \)
        """,
        regimen,
        flags=re.X,
    )

    result = []

    for name, dbid in matches:

        name = clean(
            name
        )

        dbid = clean(
            dbid
        )

        if name:
            result.append(
                (
                    name,
                    dbid,
                )
            )

    return result


# ============================================================
# DISCOVER ALL BROWSE PAGES
# ============================================================

print(
    "=" * 90
)

print(
    "CELLRESDB FULL CATALOG BUILDER"
)

print(
    "=" * 90
)

print(
    "Fetching CellResDB homepage..."
)

home_html = fetch_text(
    HOME_URL
)

home_soup = BeautifulSoup(
    home_html,
    "html.parser",
)

browse_urls = set()

for a in home_soup.find_all(
    "a",
    href=True,
):

    href = a["href"]

    if (
        "getDataSetByOrgan.php"
        in href
    ):
        browse_urls.add(
            urljoin(
                BASE,
                href,
            )
        )


browse_urls = sorted(
    browse_urls
)

print(
    "Browse pages discovered:",
    len(browse_urls),
)


# ============================================================
# QUERY EACH BROWSE PAGE
# ============================================================

all_records = []

page_summary = []

for i, browse_url in enumerate(
    browse_urls,
    start=1,
):

    print()
    print(
        f"[{i}/{len(browse_urls)}] "
        f"{browse_url}"
    )

    try:
        html = fetch_text(
            browse_url
        )

    except Exception as exc:
        print(
            "  PAGE ERROR:",
            exc,
        )

        page_summary.append({
            "browse_url":
                browse_url,
            "status":
                "page_error",
            "dataset_map_entries":
                0,
            "api_rows":
                0,
            "error":
                str(exc),
        })

        continue


    encoded_map = extract_dataset_map(
        html
    )

    if not encoded_map:

        print(
            "  No datasetIDDiseaseMap found."
        )

        page_summary.append({
            "browse_url":
                browse_url,
            "status":
                "no_dataset_map",
            "dataset_map_entries":
                0,
            "api_rows":
                0,
            "error":
                None,
        })

        continue


    entries = [
        x.strip()
        for x in encoded_map.split(";")
        if x.strip()
    ]

    print(
        "  Embedded dataset entries:",
        len(entries),
    )


    try:
        payload = fetch_json(
            AJAX_URL,
            params={
                "dataSetInfo":
                    encoded_map,
            },
        )

    except Exception as exc:
        print(
            "  API ERROR:",
            exc,
        )

        page_summary.append({
            "browse_url":
                browse_url,
            "status":
                "api_error",
            "dataset_map_entries":
                len(entries),
            "api_rows":
                0,
            "error":
                str(exc),
        })

        continue


    rows = payload.get(
        "data",
        []
    )

    print(
        "  API rows:",
        len(rows),
    )


    for row in rows:

        record = dict(
            row
        )

        record[
            "_browse_url"
        ] = browse_url

        all_records.append(
            record
        )


    page_summary.append({
        "browse_url":
            browse_url,
        "status":
            "ok",
        "dataset_map_entries":
            len(entries),
        "api_rows":
            len(rows),
        "error":
            None,
    })


    time.sleep(
        0.1
    )


# ============================================================
# BUILD DATASET CATALOG
# ============================================================

if not all_records:
    raise RuntimeError(
        "No CellResDB dataset rows were returned."
    )


raw_df = pd.DataFrame(
    all_records
)

print()
print(
    "=" * 90
)

print(
    "RAW CATALOG"
)

print(
    "=" * 90
)

print(
    "Rows before deduplication:",
    len(raw_df),
)


if "Database_ID" not in raw_df.columns:
    raise RuntimeError(
        "CellResDB API response does not contain Database_ID."
    )


duplicate_rows = raw_df.duplicated(
    subset=[
        "Database_ID"
    ],
    keep=False,
).sum()

print(
    "Rows participating in duplicate IDs:",
    duplicate_rows,
)


# Keep one canonical copy of every dataset.
# Disease/tissue browse pages can expose the same dataset repeatedly.

catalog = (
    raw_df
    .sort_values(
        "Database_ID"
    )
    .drop_duplicates(
        subset=[
            "Database_ID"
        ],
        keep="first",
    )
    .reset_index(
        drop=True
    )
)


catalog[
    "detail_url"
] = catalog[
    "Database_ID"
].map(
    lambda x:
        urljoin(
            BASE,
            (
                "php_mysql/"
                "datasetDetail.php"
                f"?id={x}"
            ),
        )
)


# ============================================================
# BUILD DRUG TABLE
# ============================================================

drug_rows = []

for _, row in catalog.iterrows():

    dataset_id = row.get(
        "Database_ID"
    )

    regimen = clean(
        row.get(
            "Therapeutic_regimen"
        )
    )

    cancer = clean(
        row.get(
            "Cancer_subtype_recorded_in_the_data_source"
        )
    )

    drug_type = clean(
        row.get(
            "Drug_type"
        )
    )

    source_dataset = clean(
        row.get(
            "Source.dataset"
        )
    )

    pmid = clean(
        row.get(
            "Reference.Pubmed.ID."
        )
    )

    drugs = extract_drugs_from_regimen(
        regimen
    )

    if not drugs:

        drug_rows.append({
            "dataset_id":
                dataset_id,

            "drug_name":
                None,

            "drugbank_id":
                None,

            "therapeutic_regimen":
                regimen,

            "drug_type":
                drug_type,

            "cancer_type":
                cancer,

            "source_dataset":
                source_dataset,

            "pmid":
                pmid,
        })

        continue


    for drug_name, drugbank_id in drugs:

        drug_rows.append({
            "dataset_id":
                dataset_id,

            "drug_name":
                drug_name,

            "drugbank_id":
                drugbank_id,

            "therapeutic_regimen":
                regimen,

            "drug_type":
                drug_type,

            "cancer_type":
                cancer,

            "source_dataset":
                source_dataset,

            "pmid":
                pmid,
        })


drug_df = pd.DataFrame(
    drug_rows
)


# ============================================================
# SAVE OUTPUTS
# ============================================================

raw_path = (
    OUT_DIR
    / "cellresdb_full_catalog_raw.csv"
)

catalog_path = (
    OUT_DIR
    / "cellresdb_full_catalog.csv"
)

drug_path = (
    OUT_DIR
    / "cellresdb_full_drugs.csv"
)

page_path = (
    OUT_DIR
    / "cellresdb_browse_page_audit.csv"
)


raw_df.to_csv(
    raw_path,
    index=False,
)

catalog.to_csv(
    catalog_path,
    index=False,
)

drug_df.to_csv(
    drug_path,
    index=False,
)

pd.DataFrame(
    page_summary
).to_csv(
    page_path,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

print()
print(
    "=" * 90
)

print(
    "CATALOG SUMMARY"
)

print(
    "=" * 90
)

print(
    "Unique CellResDB datasets:",
    catalog[
        "Database_ID"
    ].nunique()
)


if (
    "Source.dataset"
    in catalog.columns
):

    print(
        "Unique source datasets:",
        catalog[
            "Source.dataset"
        ].nunique(
            dropna=True
        )
    )


if (
    "standardDisease"
    in catalog.columns
):

    print(
        "Unique standardized cancers:",
        catalog[
            "standardDisease"
        ].nunique(
            dropna=True
        )
    )


if (
    "Cell.counts"
    in catalog.columns
):

    cell_counts = pd.to_numeric(
        catalog[
            "Cell.counts"
        ],
        errors="coerce",
    )

    print(
        "Total cells:",
        int(
            cell_counts.sum()
        )
    )


if (
    "Sample.counts"
    in catalog.columns
):

    sample_counts = pd.to_numeric(
        catalog[
            "Sample.counts"
        ],
        errors="coerce",
    )

    print(
        "Total samples:",
        int(
            sample_counts.sum()
        )
    )


valid_drugs = (
    drug_df[
        "drug_name"
    ]
    .dropna()
    .astype(str)
    .str.strip()
)

print(
    "Unique parsed treatment drugs:",
    valid_drugs.nunique()
)


print()
print(
    "=" * 90
)

print(
    "TREATMENT DRUGS"
)

print(
    "=" * 90
)

unique_drugs = sorted(
    valid_drugs.unique(),
    key=str.lower,
)

for drug in unique_drugs:
    print(
        drug
    )


print()
print(
    "=" * 90
)

print(
    "THERAPY TYPES"
)

print(
    "=" * 90
)


if (
    "Drug_type"
    in catalog.columns
):

    print(
        catalog[
            "Drug_type"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )


print()
print(
    "=" * 90
)

print(
    "OUTPUTS"
)

print(
    "=" * 90
)

print(
    raw_path
)

print(
    catalog_path
)

print(
    drug_path
)

print(
    page_path
)
