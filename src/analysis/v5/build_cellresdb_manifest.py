from pathlib import Path
from io import StringIO
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[3]

DATASETS_XLSX = ROOT / "data/raw/cellresdb/Supplementary_Data_1.xlsx"
MATCHED_DRUGS = (
    ROOT
    / "data/processed/v5/cellresdb/cellresdb_matched_drugs_refined.csv"
)

OUT = (
    ROOT
    / "data/processed/v5/cellresdb/cellresdb_dataset_manifest.csv"
)


BASE_DETAIL = (
    "https://cellknowledge.com.cn/cellresponse/"
    "php_mysql/datasetDetail.php?id={}"
)

BASE_RDS = (
    "https://cellknowledge.com.cn/cellresponse/"
    "data/from_qiao/compress_data/{}.rds"
)


def norm(x):
    if pd.isna(x):
        return ""
    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(x).strip().lower(),
    )


# ------------------------------------------------------------
# Load CellResDB dataset IDs
# ------------------------------------------------------------

datasets = pd.read_excel(
    DATASETS_XLSX,
    header=1,
)

datasets = datasets.rename(
    columns={
        "Database ID": "database_id",
        "Tumor type": "tumor_type_supp",
        "Tissue type": "tissue_type",
        "Cells": "cells_supp",
        "Patients": "patients_supp",
        "Samples": "samples_supp",
        "Data Accession": "accession",
        "PMID": "pmid",
        "Article": "article",
        "Journal": "journal",
    }
)

datasets = datasets[
    datasets["database_id"].notna()
].copy()


# ------------------------------------------------------------
# Load PHAROS-matched CellResDB drugs
# ------------------------------------------------------------

matched = pd.read_csv(MATCHED_DRUGS)

matched_names = sorted(
    set(matched["normalized_drug"].dropna())
)

print("PHAROS-matched CellResDB drug identities:")
print(len(matched_names))
print(matched_names)


# ------------------------------------------------------------
# Helper: extract labelled field from HTML table
# ------------------------------------------------------------

def extract_field(soup, label):
    label_norm = norm(label)

    for tr in soup.find_all("tr"):
        cells = tr.find_all(["td", "th"])

        if len(cells) < 2:
            continue

        left = cells[0].get_text(
            " ",
            strip=True,
        )

        if label_norm in norm(left):
            return cells[1].get_text(
                " ",
                strip=True,
            )

    return None


# ------------------------------------------------------------
# Scrape each dataset
# ------------------------------------------------------------

session = requests.Session()

rows = []

for i, row in datasets.reset_index(drop=True).iterrows():

    dataset_id = str(row["database_id"]).strip()

    print(
        f"[{i + 1}/{len(datasets)}] "
        f"{dataset_id}"
    )

    detail_url = BASE_DETAIL.format(dataset_id)
    rds_url = BASE_RDS.format(dataset_id)

    record = {
        "database_id": dataset_id,
        "detail_url": detail_url,
        "rds_url": rds_url,

        "tumor_type_supp":
            row.get("tumor_type_supp"),

        "tissue_type":
            row.get("tissue_type"),

        "cells_supp":
            row.get("cells_supp"),

        "patients_supp":
            row.get("patients_supp"),

        "samples_supp":
            row.get("samples_supp"),

        "accession":
            row.get("accession"),

        "pmid":
            row.get("pmid"),
    }

    try:
        response = session.get(
            detail_url,
            timeout=30,
        )

        response.raise_for_status()

        html = response.text
        soup = BeautifulSoup(
            html,
            "html.parser",
        )

        record["cancer_type"] = extract_field(
            soup,
            "Cancer type",
        )

        record["therapy_type"] = extract_field(
            soup,
            "Therapy type",
        )

        record["therapeutic_regimen"] = extract_field(
            soup,
            "Therapeutic regimen",
        )

        record["sample_size_page"] = extract_field(
            soup,
            "Sample size",
        )

        record["cell_number_page"] = extract_field(
            soup,
            "Cell number",
        )


        # ----------------------------------------------------
        # Sample-level clinical metadata
        # ----------------------------------------------------

        r_count = 0
        nr_count = 0
        untreated_count = 0

        pre_count = 0
        post_count = 0

        try:
            tables = pd.read_html(
                StringIO(html)
            )

            sample_table = None

            for table in tables:
                cols = [
                    str(c).strip()
                    for c in table.columns
                ]

                if (
                    "Sample ID" in cols
                    and "Response" in cols
                ):
                    sample_table = table
                    break

            if sample_table is not None:

                response_col = (
                    sample_table["Response"]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                )

                r_count = int(
                    (response_col == "R").sum()
                )

                nr_count = int(
                    (response_col == "NR").sum()
                )

                untreated_count = int(
                    (
                        response_col
                        == "UNTREATED"
                    ).sum()
                )

                if "Sampling" in sample_table.columns:

                    sampling = (
                        sample_table["Sampling"]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                    )

                    pre_count = int(
                        (sampling == "pre").sum()
                    )

                    post_count = int(
                        (sampling == "post").sum()
                    )

        except Exception as e:
            record["sample_parse_error"] = str(e)

        record["n_response"] = r_count
        record["n_nonresponse"] = nr_count
        record["n_untreated"] = untreated_count

        record["n_pre"] = pre_count
        record["n_post"] = post_count


        # ----------------------------------------------------
        # PHAROS drug matching against regimen
        # ----------------------------------------------------

        regimen_norm = norm(
            record.get(
                "therapeutic_regimen",
                "",
            )
        )

        found = [
            d
            for d in matched_names
            if d and d in regimen_norm
        ]

        record["matched_pharos_drugs"] = ";".join(
            found
        )

        record["n_matched_pharos_drugs"] = len(
            found
        )

        record["has_pharos_drug"] = (
            len(found) > 0
        )

        # Useful for direct R-vs-NR signature
        record["has_r_and_nr"] = (
            r_count > 0
            and nr_count > 0
        )

        record["candidate_for_v5"] = (
            record["has_pharos_drug"]
            and record["has_r_and_nr"]
        )

        record["page_ok"] = True

    except Exception as e:

        record["page_ok"] = False
        record["error"] = str(e)

    rows.append(record)

    time.sleep(0.25)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

manifest = pd.DataFrame(rows)

manifest.to_csv(
    OUT,
    index=False,
)


print("\n" + "=" * 80)
print("CELLRESDB MANIFEST SUMMARY")
print("=" * 80)

print(
    "Datasets:",
    len(manifest),
)

print(
    "Pages successfully parsed:",
    int(manifest["page_ok"].sum()),
)

print(
    "Datasets containing ≥1 matched PHAROS drug:",
    int(manifest["has_pharos_drug"].sum()),
)

print(
    "Datasets containing both R and NR:",
    int(manifest["has_r_and_nr"].sum()),
)

print(
    "V5 candidate datasets:",
    int(manifest["candidate_for_v5"].sum()),
)


print("\nCandidate datasets:")

cols = [
    "database_id",
    "cancer_type",
    "therapy_type",
    "therapeutic_regimen",
    "matched_pharos_drugs",
    "n_response",
    "n_nonresponse",
    "n_pre",
    "n_post",
]

print(
    manifest.loc[
        manifest["candidate_for_v5"].fillna(False).astype(bool),
        cols,
    ].to_string(index=False)
)

print("\nSaved:")
print(OUT)
