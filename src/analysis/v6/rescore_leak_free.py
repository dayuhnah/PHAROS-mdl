"""
PHAROS V6: leak-free validation re-scoring (sensitivity analysis).

Read-only. Retrains nothing; re-uses saved val_predictions.parquet files.

Background
----------
The V4 split was drawn over the raw (drug_a_key, drug_b_key) tuple.
89 canonical pairs exist in both orientations, so 26-31 canonical pairs
per seed have rows in BOTH train and val (113-268 rows).

This script:
  1. defines, per seed, the LEAKED canonical pairs (rows on both sides);
  2. re-scores every available model on
        full      = the original validation set (should reproduce known RMSE)
        leak_free = validation rows whose canonical pair never appears in train;
  3. writes per-seed val-row annotations (leak flag + evidence strata) that
     later V6 analyses can join on, so every model is scored on the same rows.

The splits themselves are NOT changed.

Run from the repo root:
    python -m src.analysis.v6.rescore_leak_free
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.v6.audit_split_evidence_coverage import (
    SEEDS,
    load_split,
    load_drmref_active,
    load_cellresdb_tiers,
    norm_key,
    canonical_pair,
)


OUT_DIR = Path("outputs/v6/leak_free")

# Same paths as src/analysis/v6/compare_ablation_regimes.py and
# src/analysis/v6/export_validation_predictions.py
def prediction_path(model, seed):
    if model == "synergyx":
        return Path(f"outputs/v4/benchmarks/synergyx/seed_{seed}/"
                    "val_predictions.parquet")
    if model == "v4":
        return Path(f"outputs/v4/predictions/v4/seed_{seed}/"
                    "val_predictions.parquet")
    if model in {"v5", "v5_1"}:
        return Path(f"outputs/v5/predictions/{model}/seed_{seed}/"
                    "val_predictions.parquet")
    return Path(f"outputs/v4/ablations/{model}/seed_{seed}/"
                "val_predictions.parquet")


MODELS = [
    "synergyx",
    "v4",
    "v5",
    "v5_1",
    "no_resistance",
    "no_drmref",
    "no_core29",
    "no_mutation",
    "mutation_only",
    "expression_only",
]

# Known 3-seed mean RMSE on the FULL validation set (project record).
# Used only to confirm the saved predictions reproduce the reported numbers.
EXPECTED_FULL_RMSE_MEAN = {
    "synergyx": 4.397157,
    "v4": 4.438550,
    "v5_1": 4.438687,
    "v5": 4.4518,
}


# ============================================================
# METRICS
# ============================================================

def metrics(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    e = p - y
    ss_res = float(np.sum(e ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "n": len(y),
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "mae": float(np.mean(np.abs(e))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        "pearson": float(np.corrcoef(y, p)[0, 1]),
        "spearman": float(pd.Series(y).rank().corr(pd.Series(p).rank())),
    }


# ============================================================
# LEAK DEFINITION + ANNOTATIONS
# ============================================================

def leaked_pairs(split_df):
    sides = split_df.groupby("pair_key")["split"].nunique()
    return set(sides[sides > 1].index)


def build_annotations(split_df, leaked, drmref, tiers):
    val = split_df[split_df["split"] == "val"].reset_index(drop=True).copy()
    ann = pd.DataFrame({
        "val_index": np.arange(len(val)),
        "drug_a_key": val["drug_a_key"],
        "drug_b_key": val["drug_b_key"],
        "pair_key": val["pair_key"],
    })
    if "depmap_id" in val:
        ann["depmap_id"] = val["depmap_id"].values
    ann["leaked_pair"] = ann["pair_key"].isin(leaked)
    ann["drmref_n"] = (val["a_ds"].isin(drmref).astype(int)
                       + val["b_ds"].isin(drmref).astype(int))
    for t, ks in tiers.items():
        ann[f"cellresdb_{t}_n"] = (val["a_k"].isin(ks).astype(int)
                                   + val["b_k"].isin(ks).astype(int))
    return ann


# ============================================================
# MAIN
# ============================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    drmref = load_drmref_active()
    tiers, _ = load_cellresdb_tiers()

    rows = []
    missing = []

    for seed in SEEDS:
        split_df = load_split(seed)
        leaked = leaked_pairs(split_df)
        n_val = int((split_df["split"] == "val").sum())

        ann = build_annotations(split_df, leaked, drmref, tiers)
        ann.to_parquet(OUT_DIR / f"val_row_annotations_seed{seed}.parquet",
                       index=False)
        pd.Series(sorted(leaked), name="pair_key").to_csv(
            OUT_DIR / f"leaked_pairs_seed{seed}.csv", index=False)

        print(f"\nSEED {seed}: {len(leaked)} leaked pairs | "
              f"{int(ann['leaked_pair'].sum()):,} of {n_val:,} val rows "
              f"removed for leak_free")

        for model in MODELS:
            path = prediction_path(model, seed)
            if not path.exists():
                missing.append(str(path))
                continue

            pred = pd.read_parquet(path)
            for col in ["drug_a_key", "drug_b_key", "zip_score",
                        "prediction"]:
                if col not in pred.columns:
                    raise RuntimeError(f"{path} lacks column '{col}'")

            if len(pred) != n_val:
                raise RuntimeError(
                    f"{model} seed {seed}: {len(pred):,} predictions vs "
                    f"{n_val:,} val rows in split")

            # Order-independent: leak flag comes from the row's own drug keys.
            pk = [canonical_pair(norm_key(a), norm_key(b))
                  for a, b in zip(pred["drug_a_key"], pred["drug_b_key"])]
            is_leak = pd.Series(pk).isin(leaked).to_numpy()

            for subset, mask in [("full", np.ones(len(pred), bool)),
                                 ("leak_free", ~is_leak)]:
                m = metrics(pred.loc[mask, "zip_score"],
                            pred.loc[mask, "prediction"])
                rows.append({"seed": seed, "model": model,
                             "subset": subset, **m})

    per_seed = pd.DataFrame(rows)
    if per_seed.empty:
        raise RuntimeError("No prediction files found. Missing:\n"
                           + "\n".join(missing))

    summary = (per_seed
               .groupby(["model", "subset"])
               .agg(seeds=("seed", "nunique"),
                    n_mean=("n", "mean"),
                    rmse_mean=("rmse", "mean"),
                    rmse_sd=("rmse", "std"),
                    mae_mean=("mae", "mean"),
                    pearson_mean=("pearson", "mean"),
                    spearman_mean=("spearman", "mean"))
               .reset_index())

    wide = summary.pivot(index="model", columns="subset",
                         values=["rmse_mean", "rmse_sd", "pearson_mean"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide["delta_rmse_leakfree_minus_full"] = (
        wide["rmse_mean_leak_free"] - wide["rmse_mean_full"])
    wide = wide.reset_index().sort_values("rmse_mean_leak_free")

    # Ranking stability: does any model order change?
    rank_full = wide.sort_values("rmse_mean_full")["model"].tolist()
    rank_free = wide.sort_values("rmse_mean_leak_free")["model"].tolist()

    # Reproduction check against the project record.
    repro = []
    for model, expected in EXPECTED_FULL_RMSE_MEAN.items():
        r = summary[(summary["model"] == model) & (summary["subset"] == "full")]
        if r.empty:
            continue
        obs = float(r["rmse_mean"].iloc[0])
        repro.append({"model": model, "observed_full_rmse_mean": obs,
                      "expected": expected,
                      "ok": abs(obs - expected) < 1e-3})
    repro = pd.DataFrame(repro)

    per_seed.to_csv(OUT_DIR / "rescore_by_seed.csv", index=False)
    summary.to_csv(OUT_DIR / "rescore_summary_long.csv", index=False)
    wide.to_csv(OUT_DIR / "rescore_full_vs_leak_free.csv", index=False)
    repro.to_csv(OUT_DIR / "reproduction_check.csv", index=False)

    pd.set_option("display.width", 180)
    print("\n" + "=" * 90)
    print("REPRODUCTION CHECK (full validation vs project record)")
    print("=" * 90)
    print(repro.to_string(index=False) if len(repro) else "(none available)")

    print("\n" + "=" * 90)
    print("FULL vs LEAK-FREE (3-seed mean)")
    print("=" * 90)
    print(wide.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nRanking (full)     :", " < ".join(rank_full))
    print("Ranking (leak-free):", " < ".join(rank_free))
    print("Ranking unchanged  :", rank_full == rank_free)

    if missing:
        print("\nSkipped (file not found):")
        for m in missing:
            print("  ", m)

    print("\nSaved to", OUT_DIR)


if __name__ == "__main__":
    main()
