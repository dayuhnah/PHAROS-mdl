"""
PHAROS V6 — Stage 2: evidence coverage on the ACTUAL evaluation splits.

Read-only audit. Trains nothing, modifies nothing.

For each saved V4 split (seeds 42 / 123 / 456) it counts, in train and in
validation, how many rows / held-out exact pairs / cell lines fall into each
evidence stratum:

    DRMref (9 active drugs)        : none / exactly one drug / both drugs
    CellResDB, by evidence tier    : none / exactly one drug / both drugs
        broad          = all approved PHAROS-CellResDB matches
        rnr_eligible   = R/NR-signature-eligible datasets
        strong_usable  = strong + usable patient quality
        strong         = strong patient quality only
        direct_mono    = drugs with direct-monotherapy evidence (if available)
    Joint DRMref x CellResDB(strong_usable / broad)

It also runs two sanity checks:

    1. Reproduce the known full-population coverage numbers
       (44,859 / 33,423 / 12,136 / 11,357 rows; 1,575 broad both-drug rows).
    2. Pair-orientation leak check: does any canonical (order-free) drug pair
       have rows in BOTH train and val? The V4 split is drawn over the raw
       (drug_a_key, drug_b_key) tuple, so this must be confirmed to be zero.

Pre-registered rule (Stage 2 decision point):
    a stratum is used for a FORMAL subset test only if every seed has
    >= MIN_VAL_PAIRS held-out pairs in it; otherwise it is DESCRIPTIVE only.

Run from the repo root:
    python -m src.analysis.v6.audit_split_evidence_coverage
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

SEEDS = [42, 123, 456]

MIN_VAL_PAIRS = 30   # pre-registered; do not tune after seeing results

SPLIT_DIR = Path("data/processed/v4/training")

DRMREF_NPZ = Path(
    "data/processed/combination/drmref/"
    "pharos_drmref_resistance_weights.npz"
)

EXP = Path("outputs/v6/cellresdb_expansion")
CRDB_MAP = EXP / "cellresdb_pharos_drug_overlap_audited.csv"
CRDB_QUALITY = EXP / "cellresdb_patient_response_quality.csv"
CRDB_ELIGIBILITY = EXP / "cellresdb_response_eligibility_dataset_drug.csv"
CRDB_CONTEXT = EXP / "cellresdb_evidence_context_per_drug.csv"

OUT_DIR = Path("outputs/v6/split_coverage")

EXPECTED_POPULATION = 280_584

# Known full-population "any drug covered" rows (from the V6 audits).
EXPECTED_ANY_ROWS = {
    "broad": 44_859,
    "rnr_eligible": 33_423,
    "strong_usable": 12_136,
    "strong": 11_357,
}
EXPECTED_BROAD_BOTH_ROWS = 1_575

EXPECTED_DRMREF = {
    "JQ1", "CERITINIB", "DASATINIB", "DOCETAXEL", "ERLOTINIB",
    "GEMCITABINE", "PACLITAXEL", "PLX4720", "VEMURAFENIB",
}


# ============================================================
# KEY NORMALISATION
# (each mirrors the script whose numbers we must reproduce)
# ============================================================

def norm_dataset(x):
    """Same as src/data/v4/mocktail_v4_dataset.py::norm (used for DRMref)."""
    return re.sub(r"[^A-Z0-9]", "", str(x).upper())


def norm_key(x):
    """Same as the src/data/v6 CellResDB audits."""
    if pd.isna(x):
        return ""
    return str(x).strip().upper()


def parse_bool(x):
    return str(x).strip().lower() in {"true", "1", "yes"}


def canonical_pair(a, b):
    return f"{a}||{b}" if a <= b else f"{b}||{a}"


# ============================================================
# EVIDENCE KEY SETS
# ============================================================

def load_drmref_active():
    """Active = mask has any non-zero entry; duplicates prefer active,
    exactly as PharosMocktailV4Dataset does."""
    drm = np.load(DRMREF_NPZ, allow_pickle=True)
    available = {}
    for name, mask in zip(drm["drug_names"], drm["mask"]):
        key = norm_dataset(name)
        active = bool(np.asarray(mask).sum() > 0)
        if available.get(key, False) and not active:
            continue
        available[key] = active
    return {k for k, v in available.items() if v}


def load_cellresdb_tiers():
    mapping = pd.read_csv(CRDB_MAP)
    mapping = mapping[mapping["approved"].map(parse_bool)].copy()

    drug_to_keys = {}
    for _, row in mapping.iterrows():
        drug_to_keys[str(row["cellres_drug"])] = {
            norm_key(x)
            for x in str(row["approved_pharos_keys"]).split(";")
            if str(x).strip()
        }

    def keys_for(drugs):
        out = set()
        for d in drugs:
            out |= drug_to_keys.get(str(d), set())
        return out

    quality = pd.read_csv(CRDB_QUALITY)
    elig = pd.read_csv(CRDB_ELIGIBILITY)
    elig = elig[elig["signature_eligible"].map(parse_bool)].copy()
    elig["dataset_id"] = elig["dataset_id"].astype(str)
    quality["dataset_id"] = quality["dataset_id"].astype(str)

    def drugs_with_quality(allowed):
        ds = set(quality.loc[quality["patient_quality"].isin(allowed),
                             "dataset_id"])
        return set(elig.loc[elig["dataset_id"].isin(ds),
                            "cellres_drug"].astype(str))

    tiers = {
        "broad": keys_for(drug_to_keys.keys()),
        "rnr_eligible": keys_for(set(elig["cellres_drug"].astype(str))),
        "strong_usable": keys_for(drugs_with_quality({"strong", "usable"})),
        "strong": keys_for(drugs_with_quality({"strong"})),
    }

    if CRDB_CONTEXT.exists():
        ctx = pd.read_csv(CRDB_CONTEXT)
        if {"cellres_drug", "has_direct_monotherapy"} <= set(ctx.columns):
            mono = ctx.loc[ctx["has_direct_monotherapy"].map(parse_bool),
                           "cellres_drug"].astype(str)
            tiers["direct_mono"] = keys_for(mono)
        else:
            print("NOTE: context file lacks expected columns; "
                  "skipping direct_mono tier.")
    else:
        print(f"NOTE: {CRDB_CONTEXT} not found; skipping direct_mono tier.")

    key_to_drug = {k: d for d, ks in drug_to_keys.items() for k in ks}
    return tiers, key_to_drug


# ============================================================
# SPLIT LOADING
# ============================================================

def load_split(seed):
    path = SPLIT_DIR / f"mocktail_v4_split_seed{seed}.parquet"
    df = pd.read_parquet(path)
    need = {"drug_a_key", "drug_b_key", "split"}
    missing = need - set(df.columns)
    if missing:
        raise RuntimeError(f"{path} missing columns {sorted(missing)}")

    keep = [c for c in ["drug_a_key", "drug_b_key", "split", "depmap_id"]
            if c in df.columns]
    df = df[keep].copy()

    # Two key spellings: one per evidence source's convention.
    df["a_ds"] = df["drug_a_key"].map(norm_dataset)
    df["b_ds"] = df["drug_b_key"].map(norm_dataset)
    df["a_k"] = df["drug_a_key"].map(norm_key)
    df["b_k"] = df["drug_b_key"].map(norm_key)

    df["pair_key"] = [canonical_pair(a, b)
                      for a, b in zip(df["a_k"], df["b_k"])]
    return df


def n_covered(df, keys, a_col, b_col):
    return (df[a_col].isin(keys).astype(int)
            + df[b_col].isin(keys).astype(int))


# ============================================================
# MAIN
# ============================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    drmref = load_drmref_active()
    tiers, key_to_drug = load_cellresdb_tiers()

    print("=" * 90)
    print("EVIDENCE KEY SETS")
    print("=" * 90)
    print(f"DRMref active drugs ({len(drmref)}): {sorted(drmref)}")
    if drmref != EXPECTED_DRMREF:
        print("  WARNING: differs from the expected 9 drugs.")
        print("  extra  :", sorted(drmref - EXPECTED_DRMREF))
        print("  missing:", sorted(EXPECTED_DRMREF - drmref))
    for t, ks in tiers.items():
        print(f"CellResDB {t:14s}: {len(ks):3d} PHAROS keys")

    rows = []        # stratum counts
    joint_rows = []  # DRMref x CellResDB
    drug_rows = []   # per evidence drug, val
    checks = []      # sanity checks

    for seed in SEEDS:
        df = load_split(seed)
        n = len(df)

        print()
        print("=" * 90)
        print(f"SEED {seed}: {n:,} rows | "
              f"train {(df['split'] == 'train').sum():,} | "
              f"val {(df['split'] == 'val').sum():,}")
        print("=" * 90)

        # ---------- check: population size ----------
        checks.append({"seed": seed, "check": "population_rows",
                       "observed": n, "expected": EXPECTED_POPULATION,
                       "ok": n == EXPECTED_POPULATION})

        # ---------- check: orientation leak ----------
        sides = df.groupby("pair_key")["split"].nunique()
        leaked = sides[sides > 1].index
        leak_rows = int(df["pair_key"].isin(leaked).sum())
        orient = (df.assign(o=df["a_k"] + "||" + df["b_k"])
                    .groupby("pair_key")["o"].nunique())
        both_orient = int((orient > 1).sum())
        checks.append({"seed": seed,
                       "check": "canonical_pairs_in_both_train_and_val",
                       "observed": len(leaked), "expected": 0,
                       "ok": len(leaked) == 0})
        checks.append({"seed": seed,
                       "check": "rows_in_leaked_pairs",
                       "observed": leak_rows, "expected": 0,
                       "ok": leak_rows == 0})
        checks.append({"seed": seed,
                       "check": "canonical_pairs_with_both_orientations",
                       "observed": both_orient, "expected": 0,
                       "ok": both_orient == 0})
        print(f"Orientation leak: {len(leaked)} canonical pairs span "
              f"train+val ({leak_rows:,} rows); "
              f"{both_orient} pairs stored in both orientations.")
        if len(leaked):
            df[df["pair_key"].isin(leaked)][
                ["pair_key", "drug_a_key", "drug_b_key", "split"]
            ].drop_duplicates().to_csv(
                OUT_DIR / f"orientation_leak_pairs_seed{seed}.csv",
                index=False)

        # ---------- per-source strata ----------
        sources = {"drmref": (drmref, "a_ds", "b_ds")}
        for t, ks in tiers.items():
            sources[f"cellresdb_{t}"] = (ks, "a_k", "b_k")

        cov = {}
        for src, (keys, ac, bc) in sources.items():
            c = n_covered(df, keys, ac, bc)
            cov[src] = c
            strata = {"none": c == 0, "exactly_one": c == 1,
                      "both": c == 2, "any": c >= 1}
            for split in ["train", "val"]:
                in_split = df["split"] == split
                for sname, m in strata.items():
                    sub = df[in_split & m]
                    rows.append({
                        "seed": seed, "source": src, "stratum": sname,
                        "split": split, "rows": len(sub),
                        "pct_of_split": 100 * len(sub) / max(in_split.sum(), 1),
                        "unique_pairs": sub["pair_key"].nunique(),
                        "cell_lines": (sub["depmap_id"].nunique()
                                       if "depmap_id" in sub else np.nan),
                    })

            # population reproduction (train+val = full population)
            tier = src.replace("cellresdb_", "")
            if tier in EXPECTED_ANY_ROWS:
                obs = int((c >= 1).sum())
                checks.append({"seed": seed,
                               "check": f"population_any_rows_{tier}",
                               "observed": obs,
                               "expected": EXPECTED_ANY_ROWS[tier],
                               "ok": obs == EXPECTED_ANY_ROWS[tier]})
            if tier == "broad":
                obs = int((c == 2).sum())
                checks.append({"seed": seed,
                               "check": "population_both_rows_broad",
                               "observed": obs,
                               "expected": EXPECTED_BROAD_BOTH_ROWS,
                               "ok": obs == EXPECTED_BROAD_BOTH_ROWS})

        # ---------- joint DRMref x CellResDB ----------
        d_any = cov["drmref"] >= 1
        for t in ["strong_usable", "broad"]:
            c_any = cov[f"cellresdb_{t}"] >= 1
            joint = {"neither": ~d_any & ~c_any,
                     "drmref_only": d_any & ~c_any,
                     "cellresdb_only": ~d_any & c_any,
                     "both_sources": d_any & c_any}
            val = df["split"] == "val"
            for jname, m in joint.items():
                sub = df[val & m]
                joint_rows.append({
                    "seed": seed, "cellresdb_tier": t, "stratum": jname,
                    "val_rows": len(sub),
                    "val_unique_pairs": sub["pair_key"].nunique(),
                })

        # ---------- per evidence drug (val) ----------
        val_df = df[df["split"] == "val"]
        for src_name, keys, ac, bc in (
            [("drmref", drmref, "a_ds", "b_ds")]
            + [("cellresdb_broad", tiers["broad"], "a_k", "b_k")]
        ):
            for k in sorted(keys):
                m = (val_df[ac] == k) | (val_df[bc] == k)
                if not m.any():
                    continue
                drug_rows.append({
                    "seed": seed, "source": src_name, "pharos_key": k,
                    "cellres_drug": (key_to_drug.get(k, "")
                                     if src_name != "drmref" else ""),
                    "val_rows": int(m.sum()),
                    "val_unique_pairs": val_df.loc[m, "pair_key"].nunique(),
                })

    # ============================================================
    # TABLES
    # ============================================================

    strata = pd.DataFrame(rows)
    joint = pd.DataFrame(joint_rows)
    per_drug = pd.DataFrame(drug_rows)
    checks = pd.DataFrame(checks)

    # Decision table: formal test only if EVERY seed clears the bar.
    val = strata[strata["split"] == "val"]
    decision = (val.groupby(["source", "stratum"])
                   .agg(val_rows_min=("rows", "min"),
                        val_rows_mean=("rows", "mean"),
                        val_pairs_min=("unique_pairs", "min"),
                        val_pairs_mean=("unique_pairs", "mean"))
                   .reset_index())
    decision["use"] = np.where(decision["val_pairs_min"] >= MIN_VAL_PAIRS,
                               "FORMAL", "DESCRIPTIVE")
    decision.loc[decision["stratum"] == "none", "use"] = "REFERENCE"

    jdec = (joint.groupby(["cellresdb_tier", "stratum"])
                 .agg(val_rows_min=("val_rows", "min"),
                      val_pairs_min=("val_unique_pairs", "min"),
                      val_pairs_mean=("val_unique_pairs", "mean"))
                 .reset_index())
    jdec["use"] = np.where(jdec["val_pairs_min"] >= MIN_VAL_PAIRS,
                           "FORMAL", "DESCRIPTIVE")
    jdec.loc[jdec["stratum"] == "neither", "use"] = "REFERENCE"

    strata.to_csv(OUT_DIR / "coverage_by_split.csv", index=False)
    joint.to_csv(OUT_DIR / "coverage_joint_by_seed.csv", index=False)
    per_drug.to_csv(OUT_DIR / "coverage_per_drug_val.csv", index=False)
    checks.to_csv(OUT_DIR / "sanity_checks.csv", index=False)
    decision.to_csv(OUT_DIR / "subset_test_decision.csv", index=False)
    jdec.to_csv(OUT_DIR / "subset_test_decision_joint.csv", index=False)

    # ============================================================
    # PRINT
    # ============================================================

    pd.set_option("display.width", 160)

    print()
    print("=" * 90)
    print("SANITY CHECKS")
    print("=" * 90)
    print(checks.to_string(index=False))
    if not checks["ok"].all():
        print("\n>>> Some checks did not match. Read these before "
              "using the coverage numbers. <<<")

    print()
    print("=" * 90)
    print(f"SUBSET-TEST DECISION (min val pairs across seeds "
          f">= {MIN_VAL_PAIRS} -> FORMAL)")
    print("=" * 90)
    print(decision.to_string(index=False,
                             float_format=lambda x: f"{x:,.1f}"))

    print()
    print("=" * 90)
    print("JOINT DRMref x CellResDB (validation)")
    print("=" * 90)
    print(jdec.to_string(index=False,
                         float_format=lambda x: f"{x:,.1f}"))

    print()
    print("Saved to", OUT_DIR)
    for f in sorted(OUT_DIR.glob("*.csv")):
        print("  ", f)


if __name__ == "__main__":
    main()
