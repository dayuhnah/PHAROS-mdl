from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

import requests
import pandas as pd


TIER1_PATH = Path(
    "outputs/v6/cellresdb_expansion/"
    "cellresdb_rds_tier1_manifest.csv"
)

TIER2_PATH = Path(
    "outputs/v6/cellresdb_expansion/"
    "cellresdb_rds_tier2_weak_manifest.csv"
)

OUT_PATH = Path(
    "outputs/v6/cellresdb_expansion/"
    "cellresdb_rds_size_audit.csv"
)


session = requests.Session()

session.headers.update({
    "User-Agent":
        "PHAROS-FYP-CellResDB-Downloader/1.0"
})


def safe_url(url):
    """
    Preserve scheme/host while URL-encoding path spaces etc.
    """

    parts = urlsplit(
        str(url).strip()
    )

    encoded_path = quote(
        parts.path,
        safe="/",
    )

    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            encoded_path,
            parts.query,
            parts.fragment,
        )
    )


tier1 = pd.read_csv(
    TIER1_PATH
)

tier1[
    "tier"
] = "tier1"


tier2 = pd.read_csv(
    TIER2_PATH
)

tier2[
    "tier"
] = "tier2_weak"


manifest = pd.concat(
    [
        tier1,
        tier2,
    ],
    ignore_index=True,
)


records = []


print("=" * 100)
print("CELLRESDB RDS SIZE AUDIT")
print("=" * 100)


for i, row in manifest.iterrows():

    dataset_id = str(
        row[
            "dataset_id"
        ]
    )

    url = safe_url(
        row[
            "rds_url"
        ]
    )

    tier = row[
        "tier"
    ]


    print(
        f"[{i + 1}/{len(manifest)}] "
        f"{dataset_id}"
    )


    try:

        response = session.head(
            url,
            allow_redirects=True,
            timeout=60,
        )


        # Some servers don't implement HEAD properly.
        if (
            response.status_code
            >= 400
        ):

            response = session.get(
                url,
                stream=True,
                timeout=60,
            )


        size_bytes = int(
            response.headers.get(
                "Content-Length",
                0,
            )
            or 0
        )


        size_mb = (
            size_bytes
            / 1024
            / 1024
        )


        print(
            f"  HTTP: {response.status_code}"
        )

        print(
            f"  Size: {size_mb:.2f} MB"
        )


        records.append({
            "dataset_id":
                dataset_id,

            "tier":
                tier,

            "patient_quality":
                row.get(
                    "patient_quality"
                ),

            "status_code":
                response.status_code,

            "size_bytes":
                size_bytes,

            "size_mb":
                size_mb,

            "rds_url":
                url,
        })


        response.close()


    except Exception as exc:

        print(
            "  ERROR:",
            exc
        )

        records.append({
            "dataset_id":
                dataset_id,

            "tier":
                tier,

            "patient_quality":
                row.get(
                    "patient_quality"
                ),

            "status_code":
                None,

            "size_bytes":
                0,

            "size_mb":
                0,

            "rds_url":
                url,

            "error":
                str(
                    exc
                ),
        })


audit = pd.DataFrame(
    records
)


audit.to_csv(
    OUT_PATH,
    index=False,
)


print()
print("=" * 100)
print("SIZE SUMMARY")
print("=" * 100)


for tier, group in audit.groupby(
    "tier"
):

    print(
        f"{tier:<12} "
        f"{len(group):>2} files | "
        f"{group['size_mb'].sum():,.2f} MB"
    )


print()

print(
    "TOTAL:",
    f"{audit['size_mb'].sum():,.2f} MB"
)


print()
print("=" * 100)
print("FILES")
print("=" * 100)

print(
    audit[
        [
            "dataset_id",
            "tier",
            "patient_quality",
            "status_code",
            "size_mb",
        ]
    ]
    .sort_values(
        "size_mb",
        ascending=False,
    )
    .to_string(
        index=False
    )
)


print()
print("Saved:")
print(OUT_PATH)
