from pathlib import Path

import numpy as np
import pandas as pd

SEEDS = [42,123,456,789,2026]

METRICS = [
    "mse",
    "rmse",
    "mae",
    "r2",
    "pearson",
    "spearman",
]

def load_results(model_prefix: str):
    rows = []

    for seed in SEEDS:
        path = Path(
            f"outputs/{model_prefix}_seed_{seed}_best_metrics.csv"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing result file: {path}"
            )
        
        df = pd.read_csv(path)

        row = df.iloc[0].to_dict()
        row["seed"] = seed

        rows.append(row)

    return pd.DataFrame(rows)

def main():
    pharos = load_results("pharos")
    pharos_rx = load_results("pharos_rx")

    print("\n------------------------")
    print("PHAROS results")
    print(pharos[["seed"] + METRICS])
    print("\n------------------------")

    print("\n------------------------")
    print("PHAROS-RX results")
    print(pharos_rx[["seed"] + METRICS])
    print("\n------------------------")

    summary_rows = []

    for model_name, df in [
        ("PHAROS", pharos),
        ("PHAROS-RX", pharos_rx),
    ]:
        row = {
            "model" : model_name
        }

        for metric in METRICS:
            row[f"{metric}_mean"] = df[metric].mean()
            row[f"{metric}_std"] = df[metric].std(ddof=1)

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    print("\n------------------------")
    print("5-SEED SUMMARY")
    print("\n------------------------")

    for _, row in summary.iterrows():
        print(f"\n{row['model']}")
        
        for metric in METRICS:
            mean = row[f"{metric}_mean"]
            std = row[f"{metric}_std"]

            print(
                f"{metric.upper():10s}: "
                f"{mean:.6f} ± {std:.6f}"
            )
    
    paired_rows = []

    for seed in SEEDS:
        base = pharos[pharos["seed"] == seed].iloc[0]

        rx = pharos_rx[pharos_rx["seed"] == seed].iloc[0]

        row = {
            "seed": seed
        }

        for metric in METRICS:
            row[f"{metric}_pharos"] = base[metric]
            row[f"{metric}_rx"] = rx[metric]

            row[f"{metric}_delta"] = (
                rx[metric] - base[metric]
            )

        paired_rows.append(row)

    paired = pd.DataFrame(paired_rows)

    print("\n------------------------")
    print("PAIRED DIFFERENCES")
    print("RX - PHAROS")
    print("\n------------------------")

    print(
        paired[
            [
                "seed",
                "rmse_delta",
                "mae_delta",
                "r2_delta",
                "pearson_delta",
                "spearman_delta",
            ]
        ]
    )

    print("\n------------------------")
    print("PHAROS-RX WINS")
    print("\n------------------------")

    for metric in ["mse", "rmse", "mae"]:

        wins = (
            paired[f"{metric}_delta"] < 0
        ).sum()

        print(
            f"{metric.upper():10s}: "
            f"{wins}/{len(SEEDS)} seeds"
        )

    for metric in [
        "r2",
        "pearson",
        "spearman",
    ]:
        
        wins = (
            paired[f"{metric}_delta"] > 0
        ).sum()

        print(
            f"{metric.upper():10s}: "
            f"{wins}/{len(SEEDS)} seeds"
        )

    summary_path = Path(
        "outputs/pharos_vs_rx_5seed_summary.csv"
    )

    paired_path = Path(
        "outputs/pharos_vs_rx_paired_results.csv"
    )

    summary.to_csv(
        summary_path,
        index = False,
    )

    paired.to_csv(
        paired_path,
        index=False,
    )

    print(
        f"Saved summary to: {summary_path}"
    )

if __name__ == "__main__":
    main()