from pathlib import Path
import re
import pandas as pd


ROOT = Path("outputs/v4")
ABLATION_ROOT = ROOT / "ablations"
TABLE_DIR = ROOT / "tables"

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


runs = []


# ============================================================
# FULL V4
# ============================================================

for path in sorted(
    ROOT.glob(
        "mocktail_v4_seed*_best_metrics.csv"
    )
):

    df = pd.read_csv(path)

    if len(df) == 0:
        continue

    row = df.iloc[0].to_dict()

    row["experiment"] = "full_v4"
    row["source_file"] = str(path)

    runs.append(row)


# ============================================================
# ABLATIONS
# ============================================================

for path in sorted(
    ABLATION_ROOT.glob(
        "*/seed_*/best_metrics.csv"
    )
):

    df = pd.read_csv(path)

    if len(df) == 0:
        continue

    row = df.iloc[0].to_dict()

    if "experiment" not in row:
        row["experiment"] = path.parents[1].name

    row["source_file"] = str(path)

    runs.append(row)


if not runs:
    raise RuntimeError(
        "No experiment metric files found."
    )


# ============================================================
# RUN-LEVEL TABLE
# ============================================================

runs_df = pd.DataFrame(runs)

preferred = [
    "experiment",
    "seed",
    "best_epoch",
    "rmse",
    "mae",
    "r2",
    "pearson",
    "spearman",
    "expression_weight",
    "cnv_weight",
    "mutation_weight",
    "crispr_weight",
    "drmref_alpha",
    "source_file",
]

columns = [
    c for c in preferred
    if c in runs_df.columns
]

extra = [
    c for c in runs_df.columns
    if c not in columns
]

runs_df = runs_df[
    columns + extra
]

runs_df = runs_df.sort_values(
    ["experiment", "seed"]
).reset_index(drop=True)

runs_path = (
    TABLE_DIR /
    "ablation_all_runs.csv"
)

runs_df.to_csv(
    runs_path,
    index=False,
)


# ============================================================
# NUMERIC SUMMARY
# ============================================================

metrics = [
    "rmse",
    "mae",
    "r2",
    "pearson",
    "spearman",
    "expression_weight",
    "cnv_weight",
    "mutation_weight",
    "crispr_weight",
    "drmref_alpha",
]

metrics = [
    m for m in metrics
    if m in runs_df.columns
]

summary_rows = []


for experiment, group in runs_df.groupby(
    "experiment"
):

    result = {
        "experiment": experiment,
        "n_seeds": group["seed"].nunique(),
    }

    for metric in metrics:

        values = pd.to_numeric(
            group[metric],
            errors="coerce",
        ).dropna()

        if len(values) == 0:
            continue

        result[
            f"{metric}_mean"
        ] = values.mean()

        result[
            f"{metric}_std"
        ] = (
            values.std(ddof=1)
            if len(values) > 1
            else 0.0
        )

    summary_rows.append(
        result
    )


summary = pd.DataFrame(
    summary_rows
)

summary = summary.sort_values(
    "rmse_mean",
    na_position="last",
).reset_index(drop=True)

summary_path = (
    TABLE_DIR /
    "ablation_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False,
)


# ============================================================
# REPORT-FRIENDLY TABLE
# ============================================================

pretty = []

for _, row in summary.iterrows():

    out = {
        "Experiment":
            row["experiment"],

        "Seeds":
            int(row["n_seeds"]),
    }

    for metric in [
        "rmse",
        "mae",
        "r2",
        "pearson",
        "spearman",
    ]:

        mean_col = (
            f"{metric}_mean"
        )

        std_col = (
            f"{metric}_std"
        )

        if mean_col in row.index:

            out[
                metric.upper()
                if metric in ["rmse", "mae"]
                else metric.capitalize()
            ] = (
                f"{row[mean_col]:.4f} "
                f"± "
                f"{row[std_col]:.4f}"
            )

    pretty.append(out)


pretty_df = pd.DataFrame(
    pretty
)

pretty_path = (
    TABLE_DIR /
    "ablation_summary_pretty.csv"
)

pretty_df.to_csv(
    pretty_path,
    index=False,
)


# ============================================================
# PRINT
# ============================================================

print("=" * 70)
print("PHAROS V4 EXPERIMENT CONSOLIDATION")
print("=" * 70)

print()
print(
    pretty_df.to_string(
        index=False
    )
)

print()
print("Saved:")
print(runs_path)
print(summary_path)
print(pretty_path)

print()
print(
    "Original experiment files were NOT modified."
)
