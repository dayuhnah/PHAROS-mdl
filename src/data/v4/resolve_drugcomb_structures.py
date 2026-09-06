from pathlib import Path
import json
import re
import time
import urllib.parse
import urllib.request

import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


# ============================================================
# PATHS
# ============================================================

PRIORITY_PATH = Path(
    "outputs/v4_audit/v4_unresolved_drug_priority.csv"
)

OUTPUT_DIR = Path("data/processed/v4")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUTPUT_DIR / "pubchem_drug_resolution_v4.csv"


# ============================================================
# SETTINGS
# ============================================================

TOP_N = 200
REQUEST_DELAY = 0.35
MAX_RETRIES = 3


# ============================================================
# MANUAL HIGH-CONFIDENCE ALIASES
# ============================================================

MANUAL_ALIASES = {

    "ADM HYDROCHLORIDE":
        "Doxorubicin hydrochloride",

    "QUINACRINE HYDROCHLORIDE":
        "Quinacrine hydrochloride hydrate",

    "CYTARABINE HYDROCHLORIDE":
        "Cytarabine hydrochloride",

    "ELOXATIN (TN) (SANOFI SYNTHELAB)":
        "Oxaliplatin",

    "ABT-888":
        "Veliparib",

    "34793-34-5":
        "34793-34-5",
}


# ============================================================
# HTTP
# ============================================================

def http_json(url):

    for attempt in range(MAX_RETRIES):

        try:

            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent":
                    "PHAROS-FYP/1.0 academic-research"
                },
            )

            with urllib.request.urlopen(
                req,
                timeout=30,
            ) as response:

                return json.loads(
                    response.read().decode("utf-8")
                )

        except Exception as e:

            if attempt == MAX_RETRIES - 1:
                return None

            time.sleep(
                1.5 * (attempt + 1)
            )

    return None


# ============================================================
# PUBCHEM
# ============================================================

def pubchem_cids(query):

    encoded = urllib.parse.quote(
        str(query),
        safe="",
    )

    url = (
        "https://pubchem.ncbi.nlm.nih.gov/"
        "rest/pug/compound/name/"
        f"{encoded}/cids/JSON"
    )

    data = http_json(url)

    time.sleep(REQUEST_DELAY)

    if not data:
        return []

    return (
        data
        .get("IdentifierList", {})
        .get("CID", [])
    )


def pubchem_properties(cid):

    url = (
        "https://pubchem.ncbi.nlm.nih.gov/"
        "rest/pug/compound/cid/"
        f"{cid}/property/"
        "SMILES,InChIKey/JSON"
    )

    data = http_json(url)

    time.sleep(REQUEST_DELAY)

    if not data:
        return None

    props = (
        data
        .get("PropertyTable", {})
        .get("Properties", [])
    )

    if not props:
        return None

    return props[0]


# ============================================================
# SMILES STANDARDIZATION
# ============================================================

def standardize_smiles(smiles):

    if not smiles:
        return None

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    try:

        mol = rdMolStandardize.Cleanup(mol)

        # Convert salt forms to main molecular parent
        mol = rdMolStandardize.FragmentParent(mol)

        Chem.SanitizeMol(mol)

        return Chem.MolToSmiles(
            mol,
            canonical=True,
            isomericSmiles=True,
        )

    except Exception:
        return None


# ============================================================
# QUERY GENERATION
# ============================================================

def query_variants(name):

    name = str(name).strip()

    variants = []

    # --------------------------------------------------------
    # Manual alias first
    # --------------------------------------------------------

    if name in MANUAL_ALIASES:

        variants.append(
            MANUAL_ALIASES[name]
        )

    # --------------------------------------------------------
    # Original name
    # --------------------------------------------------------

    variants.append(name)

    # Normalize duplicate whitespace
    cleaned = re.sub(
        r"\s+",
        " ",
        name,
    ).strip()

    variants.append(cleaned)

    # Remove trailing parenthetical trade-name information
    no_tail = re.sub(
        r"\s+\([^)]*\)\s*$",
        "",
        cleaned,
    ).strip()

    if no_tail:
        variants.append(no_tail)

    # Remove duplicates while preserving order
    return list(dict.fromkeys(variants))


# ============================================================
# LOAD PRIORITY LIST
# ============================================================

priority = pd.read_csv(
    PRIORITY_PATH
)

priority = priority.head(
    TOP_N
).copy()


# ============================================================
# RESOLVE
# ============================================================

results = []

for number, (_, row) in enumerate(
    priority.iterrows(),
    start=1,
):

    drug_key = str(
        row["drug_key"]
    )

    raw_name = str(
        row["representative_name"]
    )

    print(
        f"[{number}/{len(priority)}] "
        f"{raw_name}"
    )

    result = None

    attempted = []

    for query in query_variants(raw_name):

        attempted.append(query)

        cids = pubchem_cids(query)

        # Conservative:
        # only accept unique PubChem matches
        if len(cids) != 1:
            continue

        cid = int(cids[0])

        props = pubchem_properties(cid)

        if not props:
            continue

        pubchem_smiles = props.get(
            "SMILES"
        )

        canonical_smiles = standardize_smiles(
            pubchem_smiles
        )

        if canonical_smiles is None:
            continue

        # Final RDKit validation
        if Chem.MolFromSmiles(
            canonical_smiles
        ) is None:
            continue

        manual_alias_used = (
            raw_name in MANUAL_ALIASES
            and
            query == MANUAL_ALIASES[raw_name]
        )

        result = {

            "drug_key":
                drug_key,

            "representative_name":
                raw_name,

            "status":
                "resolved",

            "resolution_source":
                "PubChem",

            "resolution_method":
                (
                    "manual_alias_pubchem_unique"
                    if manual_alias_used
                    else
                    "pubchem_unique_name_match"
                ),

            "query_used":
                query,

            "pubchem_cid":
                cid,

            "pubchem_inchikey":
                props.get("InChIKey"),

            "pubchem_smiles":
                pubchem_smiles,

            "canonical_parent_smiles":
                canonical_smiles,

            "confidence":
                "high",

            "immediately_recoverable":
                int(
                    row["immediately_recoverable"]
                ),

            "occurrences":
                int(
                    row["occurrences"]
                ),
        }

        break

    # --------------------------------------------------------
    # Failed resolution
    # --------------------------------------------------------

    if result is None:

        result = {

            "drug_key":
                drug_key,

            "representative_name":
                raw_name,

            "status":
                "unresolved",

            "resolution_source":
                "",

            "resolution_method":
                "",

            "query_used":
                " | ".join(attempted),

            "pubchem_cid":
                "",

            "pubchem_inchikey":
                "",

            "pubchem_smiles":
                "",

            "canonical_parent_smiles":
                "",

            "confidence":
                "",

            "immediately_recoverable":
                int(
                    row["immediately_recoverable"]
                ),

            "occurrences":
                int(
                    row["occurrences"]
                ),
        }

    results.append(result)

    # Save progress continuously
    pd.DataFrame(results).to_csv(
        OUT_PATH,
        index=False,
    )


# ============================================================
# SUMMARY
# ============================================================

out = pd.DataFrame(results)

resolved = out[
    out["status"] == "resolved"
].copy()

unresolved = out[
    out["status"] != "resolved"
].copy()

recoverable = int(
    resolved[
        "immediately_recoverable"
    ].sum()
)


print()
print("=" * 70)
print("PHAROS V4 PUBCHEM RESOLUTION COMPLETE")
print("=" * 70)

print(
    "Attempted:",
    len(out)
)

print(
    "Resolved:",
    len(resolved)
)

print(
    "Still unresolved:",
    len(unresolved)
)

print(
    "Potential immediately recoverable rows:",
    f"{recoverable:,}"
)

print()
print("Saved:")
print(OUT_PATH)


if len(unresolved) > 0:

    print()
    print("Top unresolved compounds:")

    print(
        unresolved[
            [
                "representative_name",
                "immediately_recoverable",
            ]
        ]
        .sort_values(
            "immediately_recoverable",
            ascending=False,
        )
        .head(30)
        .to_string(index=False)
    )
