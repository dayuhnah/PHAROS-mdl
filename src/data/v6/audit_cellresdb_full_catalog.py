from pathlib import Path
from urllib.parse import (
    urljoin,
    urlparse,
    parse_qs,
)

import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup


BASE = "https://cellknowledge.com.cn/cellresponse/"

HOME_URL = urljoin(
    BASE,
    "",
)

DOWNLOAD_URL = urljoin(
    BASE,
    "download.html",
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
        "PHAROS-FYP-CellResDB-Catalog-Audit/1.0"
})


# ============================================================
# HELPERS
# ============================================================

def fetch(url):
    response = SESSION.get(
        url,
        timeout=60,
    )

    response.raise_for_status()

    return response.text


def soup_for(url):
    return BeautifulSoup(
        fetch(url),
        "html.parser",
    )


def clean(text):
    if text is None:
        return None

    text = re.sub(
        r"\s+",
        " ",
        str(text),
    ).strip()

    return text or None


def field_from_text(
    text,
    label,
    next_labels,
):
    """
    Best-effort parser for visible dataset detail text.
    """

    escaped_next = "|".join(
        re.escape(x)
        for x in next_labels
    )

    pattern = (
        rf"{re.escape(label)}\s*\??\s*"
        rf"(.*?)"
        rf"(?=\s*(?:{escaped_next})\s*\??\s*|$)"
    )

    match = re.search(
        pattern,
        text,
        flags=re.I,
    )

    if not match:
        return None

    return clean(
        match.group(1)
    )


# ============================================================
# DISCOVER TISSUE/BROWSE PAGES
# ============================================================

print("=" * 80)
print("CELLRESDB FULL CATALOG AUDIT")
print("=" * 80)

print("Fetching homepage...")

home = soup_for(
    HOME_URL
)

tissue_urls = set()

for a in home.find_all(
    "a",
    href=True,
):

    href = a["href"]

    if (
        "getDataSetByOrgan.php"
        in href
    ):
        tissue_urls.add(
            urljoin(
                BASE,
                href,
            )
        )


print(
    "Tissue browse URLs discovered:",
    len(tissue_urls),
)


# ============================================================
# DISCOVER DATASET DETAIL PAGES
# ============================================================

detail_urls = set()

pages_to_scan = set(
    tissue_urls
)

pages_to_scan.add(
    DOWNLOAD_URL
)

for page_num, page_url in enumerate(
    sorted(pages_to_scan),
    start=1,
):

    print(
        f"[{page_num}/{len(pages_to_scan)}] "
        f"Scanning: {page_url}"
    )

    try:
        soup = soup_for(
            page_url
        )

    except Exception as exc:
        print(
            "  WARNING:",
            exc,
        )
        continue

    for a in soup.find_all(
        "a",
        href=True,
    ):

        href = a["href"]

        if (
            "datasetDetail.php"
            in href
            and "id=" in href
        ):
            detail_urls.add(
                urljoin(
                    BASE,
                    href,
                )
            )

    time.sleep(
        0.1
    )


print()
print(
    "Dataset detail pages discovered:",
    len(detail_urls),
)


# ============================================================
# PARSE EACH DATASET PAGE
# ============================================================

records = []

for i, url in enumerate(
    sorted(detail_urls),
    start=1,
):

    print(
        f"[{i}/{len(detail_urls)}] "
        f"{url}"
    )

    try:
        soup = soup_for(
            url
        )

    except Exception as exc:
        print(
            "  ERROR:",
            exc,
        )

        records.append({
            "detail_url":
                url,
            "parse_error":
                str(exc),
        })

        continue

    visible_text = clean(
        soup.get_text(
            " ",
            strip=True,
        )
    ) or ""

    query = parse_qs(
        urlparse(url).query
    )

    dataset_id = (
        query.get(
            "id",
            [None],
        )[0]
    )

    # --------------------------------------------------------
    # Discover downloadable RDS / download-looking links
    # --------------------------------------------------------

    download_links = []

    for a in soup.find_all(
        "a",
        href=True,
    ):

        href = urljoin(
            url,
            a["href"],
        )

        anchor_text = clean(
            a.get_text(
                " ",
                strip=True,
            )
        ) or ""

        if (
            ".rds" in href.lower()
            or
            "download" in href.lower()
            or
            "download" in anchor_text.lower()
        ):
            download_links.append(
                href
            )

    download_links = sorted(
        set(download_links)
    )

    # --------------------------------------------------------
    # Best-effort structured metadata extraction
    # --------------------------------------------------------

    labels = [
        "Dataset ID",
        "Download",
        "Sample size",
        "Cell number",
        "Cancer type",
        "Therapy type",
        "Therapeutic regimen",
        "Drug Information",
        "Administration Dosage",
        "Data type",
        "Source Dataset",
        "Platform",
        "Reference",
        "Sample Information",
        "Analysis Results",
    ]

    sample_size = field_from_text(
        visible_text,
        "Sample size",
        labels,
    )

    cell_number = field_from_text(
        visible_text,
        "Cell number",
        labels,
    )

    cancer_type = field_from_text(
        visible_text,
        "Cancer type",
        labels,
    )

    therapy_type = field_from_text(
        visible_text,
        "Therapy type",
        labels,
    )

    regimen = field_from_text(
        visible_text,
        "Therapeutic regimen",
        labels,
    )

    source_dataset = field_from_text(
        visible_text,
        "Source Dataset",
        labels,
    )

    platform = field_from_text(
        visible_text,
        "Platform",
        labels,
    )

    reference = field_from_text(
        visible_text,
        "Reference",
        labels,
    )

    # --------------------------------------------------------
    # Drug names:
    # capture Drug Information table rows where possible
    # --------------------------------------------------------

    drug_names = []

    for table in soup.find_all(
        "table"
    ):

        headers = [
            clean(th.get_text(
                " ",
                strip=True,
            ))
            for th in table.find_all(
                "th"
            )
        ]

        header_text = " | ".join(
            x
            for x in headers
            if x
        )

        if (
            "Drug Name"
            not in header_text
        ):
            continue

        for tr in table.find_all(
            "tr"
        )[1:]:

            cells = [
                clean(td.get_text(
                    " ",
                    strip=True,
                ))
                for td in tr.find_all(
                    ["td", "th"]
                )
            ]

            if cells and cells[0]:
                drug_names.append(
                    cells[0]
                )

    drug_names = sorted(
        set(drug_names)
    )

    records.append({
        "dataset_id":
            dataset_id,

        "sample_size":
            sample_size,

        "cell_number":
            cell_number,

        "cancer_type":
            cancer_type,

        "therapy_type":
            therapy_type,

        "therapeutic_regimen":
            regimen,

        "drug_names":
            ";".join(drug_names),

        "source_dataset":
            source_dataset,

        "platform":
            platform,

        "reference":
            reference,

        "download_url":
            (
                download_links[0]
                if download_links
                else None
            ),

        "download_link_count":
            len(download_links),

        "detail_url":
            url,

        "parse_error":
            None,
    })

    time.sleep(
        0.1
    )


# ============================================================
# SAVE
# ============================================================

df = pd.DataFrame(
    records
)

df = df.sort_values(
    "dataset_id",
    na_position="last",
).reset_index(
    drop=True
)

out_path = (
    OUT_DIR
    / "cellresdb_full_catalog.csv"
)

df.to_csv(
    out_path,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 80)
print("CATALOG SUMMARY")
print("=" * 80)

print(
    "Rows:",
    len(df)
)

print(
    "Unique dataset IDs:",
    df["dataset_id"]
    .nunique(
        dropna=True
    )
)

print(
    "Datasets with drug names:",
    df["drug_names"]
    .fillna("")
    .ne("")
    .sum()
)

print(
    "Datasets with download URL:",
    df["download_url"]
    .notna()
    .sum()
)

print(
    "Unique cancer types:",
    df["cancer_type"]
    .nunique(
        dropna=True
    )
)

all_drugs = set()

for value in df[
    "drug_names"
].dropna():

    for drug in str(
        value
    ).split(";"):

        drug = drug.strip()

        if drug:
            all_drugs.add(
                drug
            )


print(
    "Unique parsed treatment drugs:",
    len(all_drugs)
)

print()
print(
    "Treatment drugs:"
)

for drug in sorted(
    all_drugs
):
    print(
        " ",
        drug
    )


print()
print("=" * 80)
print("FIRST 20 DATASETS")
print("=" * 80)

display_cols = [
    "dataset_id",
    "cancer_type",
    "therapy_type",
    "therapeutic_regimen",
    "drug_names",
]

print(
    df[
        display_cols
    ]
    .head(20)
    .to_string(
        index=False
    )
)


print()
print("=" * 80)
print("SAVED")
print("=" * 80)

print(
    out_path
)
