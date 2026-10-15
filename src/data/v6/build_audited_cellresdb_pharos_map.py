from pathlib import Path
import pandas as pd


IN_PATH = Path(
    "outputs/v6/cellresdb_expansion/"
    "cellresdb_pharos_drug_overlap.csv"
)

OUT_PATH = Path(
    "outputs/v6/cellresdb_expansion/"
    "cellresdb_pharos_drug_overlap_audited.csv"
)


# ============================================================
# EXPLICIT IDENTITY-LEVEL ALIAS APPROVALS
# ============================================================
#
# Do NOT automatically trust same-SMILES aliases.
#
# Resistance evidence is transferred only when the alternate
# PHAROS key represents the same drug identity.
#

APPROVED_ALIAS_KEYS = {

    # PR-171 is Carfilzomib
    "Carfilzomib": [
        "CARFILZOMIB",
        "CARFILZOMIBPR171",
    ],

    # GSK1120212 is Trametinib
    "Trametinib": [
        "TRAMETINIB",
        "TRAMETINIBGSK1120212",
    ],

    # Cytidine is NOT Cytarabine.
    # Do not propagate Cytarabine resistance evidence.
    "Cytarabine": [
        "CYTARABINE",
    ],
}


df = pd.read_csv(
    IN_PATH
)


def approved_keys(row):

    drug = str(
        row["cellres_drug"]
    )

    status = str(
        row["match_status"]
    )

    if status == "unmatched":
        return ""

    if drug in APPROVED_ALIAS_KEYS:
        return ";".join(
            APPROVED_ALIAS_KEYS[
                drug
            ]
        )

    # Exact matches receive only their exact PHAROS keys.
    exact = row.get(
        "exact_pharos_keys"
    )

    if pd.isna(exact):
        return ""

    return str(
        exact
    )


df[
    "approved_pharos_keys"
] = df.apply(
    approved_keys,
    axis=1,
)


df[
    "approved"
] = (
    df[
        "approved_pharos_keys"
    ]
    .fillna("")
    .str.strip()
    .ne("")
)


df.to_csv(
    OUT_PATH,
    index=False,
)


print("=" * 100)
print("AUDITED CELLRESDB → PHAROS MAP")
print("=" * 100)

print(
    "CellResDB drugs:",
    len(df)
)

print(
    "Approved matched drugs:",
    int(
        df["approved"].sum()
    )
)

all_keys = set()

for value in df.loc[
    df["approved"],
    "approved_pharos_keys",
]:

    for key in str(
        value
    ).split(";"):

        key = key.strip()

        if key:
            all_keys.add(
                key
            )


print(
    "Approved PHAROS keys:",
    len(all_keys)
)

print()
print(
    df.loc[
        df["approved"],
        [
            "cellres_drug",
            "drugbank_id",
            "dataset_count",
            "approved_pharos_keys",
        ],
    ].to_string(
        index=False
    )
)

print()
print("=" * 100)
print("IMPORTANT REJECTION")
print("=" * 100)

print(
    "CYTIDINE is NOT approved as an alias "
    "for Cytarabine."
)

print()
print("Saved:")
print(OUT_PATH)
