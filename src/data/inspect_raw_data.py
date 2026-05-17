from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw")

FILES = {
    "response": RAW_DIR / "primary-screen-replicate-collapsed-logfold-change.csv",
    "expression": RAW_DIR / "OmicsExpressionProteinCodingGenesTPMLogp1.csv",
    "treatment_info": RAW_DIR / "prism-repurposing-20q2-primary-screen-replicate-treatment-info.csv",
}


def inspect_csv(name: str, path: Path, nrows: int = 3) -> None:
    print(f"\n==============================")
    print(f"{name}: {path}")
    print("==============================")

    if not path.exists():
        print("Missing file.")
        return

    df = pd.read_csv(path, nrows=nrows)
    print(f"Shape preview: {df.shape}")
    print("\nColumns:")
    print(list(df.columns[:30]))

    print("\nPreview:")
    print(df.iloc[:, :8])


def main() -> None:
    for name, path in FILES.items():
        inspect_csv(name, path)


if __name__ == "__main__":
    main()