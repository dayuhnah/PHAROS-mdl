from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit
import hashlib
import time

import pandas as pd
import requests


# ============================================================
# CONFIG
# ============================================================

MANIFEST = Path(
    "outputs/v6/cellresdb_expansion/"
    "cellresdb_rds_tier1_manifest.csv"
)

OUT_ROOT = Path(
    "data/external/cellresdb_full/raw_rds/tier1"
)

AUDIT_OUT = Path(
    "outputs/v6/cellresdb_expansion/"
    "cellresdb_tier1_download_audit.csv"
)

CHUNK_SIZE = 4 * 1024 * 1024

CONNECT_TIMEOUT = 30
READ_TIMEOUT = 180

MAX_ATTEMPTS = 30
RETRY_SLEEP = 15


# ============================================================
# HELPERS
# ============================================================

def safe_url(url):
    parts = urlsplit(
        str(url).strip()
    )

    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            quote(
                parts.path,
                safe="/"
            ),
            parts.query,
            parts.fragment,
        )
    )


def safe_filename(dataset_id):
    return (
        str(dataset_id)
        .strip()
        .replace("/", "_")
        + ".rds"
    )


def md5_file(path):
    digest = hashlib.md5()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(
                8 * 1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def save_audit(records):
    if records:
        pd.DataFrame(
            records
        ).to_csv(
            AUDIT_OUT,
            index=False
        )


def human_mb(size):
    return (
        size
        / 1024
        / 1024
    )


# ============================================================
# MANIFEST
# ============================================================

if not MANIFEST.exists():
    raise FileNotFoundError(
        f"Manifest not found: {MANIFEST}"
    )


manifest = pd.read_csv(
    MANIFEST
)


required_columns = {
    "dataset_id",
    "patient_quality",
    "rds_url",
}


missing_columns = (
    required_columns
    -
    set(
        manifest.columns
    )
)


if missing_columns:
    raise ValueError(
        "Manifest missing columns: "
        + ", ".join(
            sorted(
                missing_columns
            )
        )
    )


OUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

AUDIT_OUT.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HTTP SESSION
# ============================================================

session = requests.Session()

session.headers.update(
    {
        "User-Agent":
            "PHAROS-FYP-CellResDB-Downloader/2.0",

        # Important for byte-size / Range consistency.
        "Accept-Encoding":
            "identity",
    }
)


# ============================================================
# DISCOVER REMOTE SIZE
# ============================================================

def get_remote_size(url):

    try:
        response = session.head(
            url,
            allow_redirects=True,
            timeout=(
                CONNECT_TIMEOUT,
                READ_TIMEOUT
            )
        )

        size = int(
            response.headers.get(
                "Content-Length",
                0
            )
            or 0
        )

        response.close()

        return size

    except requests.RequestException as exc:

        print(
            f"  HEAD warning: {exc}"
        )

        return 0


# ============================================================
# RETRY / RESUME DOWNLOADER
# ============================================================

def download_with_retries(
    url,
    out_path,
    expected_size
):

    no_progress_attempts = 0
    previous_size = (
        out_path.stat().st_size
        if out_path.exists()
        else 0
    )


    for attempt in range(
        1,
        MAX_ATTEMPTS + 1
    ):

        existing_size = (
            out_path.stat().st_size
            if out_path.exists()
            else 0
        )


        # ----------------------------------------------------
        # COMPLETE
        # ----------------------------------------------------

        if (
            expected_size > 0
            and
            existing_size == expected_size
        ):

            return existing_size


        # ----------------------------------------------------
        # BAD OVERSIZED FILE
        # ----------------------------------------------------

        if (
            expected_size > 0
            and
            existing_size > expected_size
        ):

            print(
                "  Local file is larger than remote."
            )

            print(
                "  Restarting from byte 0."
            )

            out_path.unlink()

            existing_size = 0


        print()
        print(
            f"  Attempt {attempt}/{MAX_ATTEMPTS}"
        )

        print(
            "  Current size: "
            f"{human_mb(existing_size):.2f} MB"
        )


        headers = {}

        mode = "wb"


        if existing_size > 0:

            headers[
                "Range"
            ] = (
                f"bytes={existing_size}-"
            )

            mode = "ab"

            print(
                f"  Requesting resume from byte "
                f"{existing_size:,}"
            )

        else:

            print(
                "  Starting from byte 0"
            )


        response = None


        try:

            response = session.get(
                url,
                headers=headers,
                stream=True,
                allow_redirects=True,
                timeout=(
                    CONNECT_TIMEOUT,
                    READ_TIMEOUT
                )
            )


            # ------------------------------------------------
            # RANGE BEHAVIOUR
            # ------------------------------------------------

            if (
                existing_size > 0
                and
                response.status_code == 206
            ):

                print(
                    "  Server accepted Range resume ✅"
                )


            elif (
                existing_size > 0
                and
                response.status_code == 200
            ):

                print(
                    "  Server ignored Range request."
                )

                print(
                    "  Restarting this attempt from byte 0."
                )

                mode = "wb"

                existing_size = 0


            elif response.status_code == 416:

                current_size = (
                    out_path.stat().st_size
                    if out_path.exists()
                    else 0
                )

                if (
                    expected_size > 0
                    and
                    current_size == expected_size
                ):

                    print(
                        "  Server returned 416 but "
                        "local file is complete ✅"
                    )

                    response.close()

                    return current_size


                raise RuntimeError(
                    "HTTP 416 Range Not Satisfiable"
                )


            response.raise_for_status()


            # ------------------------------------------------
            # STREAM
            # ------------------------------------------------

            downloaded_this_attempt = 0


            with out_path.open(
                mode
            ) as handle:

                for chunk in response.iter_content(
                    chunk_size=CHUNK_SIZE
                ):

                    if not chunk:
                        continue

                    handle.write(
                        chunk
                    )

                    downloaded_this_attempt += (
                        len(chunk)
                    )


            response.close()

            response = None


            final_size = (
                out_path.stat().st_size
            )


            print(
                "  Attempt finished."
            )

            print(
                "  Downloaded this attempt: "
                f"{human_mb(downloaded_this_attempt):.2f} MB"
            )

            print(
                "  Current total: "
                f"{human_mb(final_size):.2f} MB"
            )


            # ------------------------------------------------
            # SIZE VALIDATION
            # ------------------------------------------------

            if expected_size > 0:

                if final_size == expected_size:

                    print(
                        "  File size verified ✅"
                    )

                    return final_size


                if final_size < expected_size:

                    print(
                        "  Connection ended before full file."
                    )

                    print(
                        "  Will resume..."
                    )


            else:

                # If Content-Length is unavailable, a clean
                # completed response is considered successful.
                return final_size


        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.RequestException,
            RuntimeError,
        ) as exc:

            if response is not None:

                try:
                    response.close()

                except Exception:
                    pass


            current_size = (
                out_path.stat().st_size
                if out_path.exists()
                else 0
            )


            gained = (
                current_size
                -
                existing_size
            )


            print()
            print(
                f"  Network interruption: {exc}"
            )

            print(
                "  Partial file preserved."
            )

            print(
                "  Current size: "
                f"{human_mb(current_size):.2f} MB"
            )

            print(
                "  Progress this attempt: "
                f"{human_mb(max(gained, 0)):.2f} MB"
            )


        # ----------------------------------------------------
        # NO-PROGRESS DETECTION
        # ----------------------------------------------------

        current_size = (
            out_path.stat().st_size
            if out_path.exists()
            else 0
        )


        if current_size <= previous_size:

            no_progress_attempts += 1

        else:

            no_progress_attempts = 0


        previous_size = current_size


        if no_progress_attempts >= 5:

            print()
            print(
                "  WARNING: 5 consecutive attempts "
                "made no progress."
            )

            print(
                "  CellResDB may currently be "
                "unresponsive."
            )


        if attempt < MAX_ATTEMPTS:

            print(
                f"  Retrying in {RETRY_SLEEP} seconds..."
            )

            time.sleep(
                RETRY_SLEEP
            )


    raise RuntimeError(
        f"Download failed after "
        f"{MAX_ATTEMPTS} attempts: {out_path.name}"
    )


# ============================================================
# MAIN DOWNLOAD LOOP
# ============================================================

records = []


print("=" * 100)
print("PHAROS V6 — CELLRESDB TIER-1 DOWNLOAD")
print("=" * 100)

print(
    f"Datasets in manifest: {len(manifest)}"
)


for i, row in manifest.iterrows():

    dataset_id = str(
        row["dataset_id"]
    ).strip()

    quality = str(
        row["patient_quality"]
    ).strip()

    url = safe_url(
        row["rds_url"]
    )


    quality_dir = (
        OUT_ROOT
        / quality
    )

    quality_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    out_path = (
        quality_dir
        / safe_filename(
            dataset_id
        )
    )


    print()
    print("=" * 100)

    print(
        f"[{i + 1}/{len(manifest)}] "
        f"{dataset_id}"
    )

    print(
        f"Quality : {quality}"
    )

    print(
        f"Output  : {out_path}"
    )


    expected_size = get_remote_size(
        url
    )


    existing_size = (
        out_path.stat().st_size
        if out_path.exists()
        else 0
    )


    if expected_size > 0:

        print(
            "Expected: "
            f"{human_mb(expected_size):.2f} MB"
        )

    else:

        print(
            "Expected: unknown"
        )


    print(
        "Existing: "
        f"{human_mb(existing_size):.2f} MB"
    )


    # --------------------------------------------------------
    # ALREADY COMPLETE
    # --------------------------------------------------------

    if (
        expected_size > 0
        and
        existing_size == expected_size
    ):

        print(
            "Already complete ✅"
        )

        checksum = md5_file(
            out_path
        )

        records.append(
            {
                "dataset_id":
                    dataset_id,

                "patient_quality":
                    quality,

                "path":
                    str(out_path),

                "expected_bytes":
                    expected_size,

                "downloaded_bytes":
                    existing_size,

                "status":
                    "already_complete",

                "md5":
                    checksum,
            }
        )

        save_audit(
            records
        )

        continue


    # --------------------------------------------------------
    # DOWNLOAD / RESUME
    # --------------------------------------------------------

    try:

        final_size = download_with_retries(
            url,
            out_path,
            expected_size
        )

    except Exception as exc:

        print()
        print(
            f"FAILED ❌: {dataset_id}"
        )

        print(
            str(exc)
        )


        partial_size = (
            out_path.stat().st_size
            if out_path.exists()
            else 0
        )


        records.append(
            {
                "dataset_id":
                    dataset_id,

                "patient_quality":
                    quality,

                "path":
                    str(out_path),

                "expected_bytes":
                    expected_size,

                "downloaded_bytes":
                    partial_size,

                "status":
                    "failed_partial",

                "md5":
                    "",
            }
        )


        save_audit(
            records
        )


        # Continue to next dataset rather than killing the
        # whole 10-dataset run.

        continue


    # --------------------------------------------------------
    # FINAL CHECKSUM
    # --------------------------------------------------------

    checksum = md5_file(
        out_path
    )


    print()
    print(
        f"Final size: "
        f"{human_mb(final_size):.2f} MB"
    )

    print(
        f"MD5: {checksum}"
    )

    print(
        "Complete ✅"
    )


    records.append(
        {
            "dataset_id":
                dataset_id,

            "patient_quality":
                quality,

            "path":
                str(out_path),

            "expected_bytes":
                expected_size,

            "downloaded_bytes":
                final_size,

            "status":
                "downloaded",

            "md5":
                checksum,
        }
    )


    save_audit(
        records
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

audit = pd.DataFrame(
    records
)


audit.to_csv(
    AUDIT_OUT,
    index=False
)


print()
print("=" * 100)
print("DOWNLOAD SUMMARY")
print("=" * 100)


if len(audit) > 0:

    print(
        audit[
            [
                "dataset_id",
                "patient_quality",
                "status",
                "downloaded_bytes",
                "md5",
            ]
        ].to_string(
            index=False
        )
    )


    completed = audit[
        audit["status"].isin(
            [
                "downloaded",
                "already_complete",
            ]
        )
    ]


    print()

    print(
        "Completed:",
        len(completed),
        "/",
        len(audit)
    )


    print(
        "Completed size:",
        f"{completed['downloaded_bytes'].sum() / 1024**3:.3f} GB"
    )


print()
print(
    "Audit saved:",
    AUDIT_OUT
)

print()
print(
    "DONE ✅"
)
