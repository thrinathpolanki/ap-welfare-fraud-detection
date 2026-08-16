"""
dataset_analysis.py
--------------------
Prints a full structural and statistical summary of the generated
AP welfare-scheme dataset. Useful for verifying data quality before
training and for demonstrating the dataset during an academic review.

Run:
    python dataset_analysis.py
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "ap_welfare_dataset.csv")


def load_dataset(path: str = DATASET_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run 'python generate_dataset.py' first."
        )
    return pd.read_csv(path)


def analyze_dataset(df: pd.DataFrame) -> None:
    sep = "=" * 70

    print(sep)
    print("AP WELFARE SCHEME FRAUD DETECTION - DATASET ANALYSIS")
    print(sep)

    print(f"\nNumber of rows      : {df.shape[0]:,}")
    print(f"Number of columns   : {df.shape[1]}")

    print("\nColumn names:")
    for col in df.columns:
        print(f"  - {col}")

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values per column:")
    missing = df.isnull().sum()
    print(missing[missing >= 0])

    print(f"\nTotal missing values : {df.isnull().sum().sum()}")

    print(f"\nFully duplicated records (all columns identical): {df.duplicated().sum()}")
    print(f"Records with duplicate Aadhaar_No (excluding first): "
          f"{df.duplicated(subset=['Aadhaar_No']).sum()}")
    print(f"Records with duplicate Mobile_No (excluding first): "
          f"{df.duplicated(subset=['Mobile_No']).sum()}")

    print("\nFraud distribution:")
    print(df["Fraud"].value_counts())
    print(f"Fraud percentage: {df['Fraud'].mean() * 100:.2f}%")

    print("\nBasic statistical summary (numerical columns):")
    print(df.describe().T)

    print("\nUnique value counts (categorical columns):")
    categorical_cols = ["District", "Village", "Scheme_Name", "Gender",
                         "Bank_Account", "Location_Risk"]
    for col in categorical_cols:
        print(f"  {col}: {df[col].nunique()} unique values")

    print("\nScheme-wise record counts:")
    print(df["Scheme_Name"].value_counts())

    print("\nDistrict-wise record counts (top 10):")
    print(df["District"].value_counts().head(10))

    print("\nDataset memory usage:")
    print(df.memory_usage(deep=True).sum() / (1024 ** 2), "MB")

    print(sep)
    print("Dataset summary complete.")
    print(sep)


def main():
    df = load_dataset()
    analyze_dataset(df)


if __name__ == "__main__":
    main()
