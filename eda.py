"""
eda.py
------
Exploratory Data Analysis for the AP welfare fraud-detection dataset.
Generates and saves a set of visualizations into the visualizations/ folder
so they can be reused in the Flask dashboard / analytics page and in the
project README / presentation.

Run:
    python eda.py
"""

import os
import matplotlib
matplotlib.use("Agg")  # headless backend, safe for servers / no display
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from feature_engineering import engineer_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "ap_welfare_dataset.csv")
VIS_DIR = os.path.join(BASE_DIR, "visualizations")

sns.set_theme(style="whitegrid")
PALETTE = ["#0B3D91", "#D64550"]  # Genuine (navy), Fraud (red)


def _savefig(fig, name):
    os.makedirs(VIS_DIR, exist_ok=True)
    path = os.path.join(VIS_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"Saved -> {path}")


def plot_fraud_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 5))
    counts = df["Fraud"].value_counts().sort_index()
    labels = ["Genuine (0)", "Fraud (1)"]
    ax.bar(labels, counts.values, color=PALETTE)
    ax.set_title("Fraud vs Genuine Claim Distribution")
    ax.set_ylabel("Number of Records")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 500, f"{v:,}", ha="center", fontweight="bold")
    _savefig(fig, "fraud_distribution.png")


def plot_fraud_by_scheme(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    rate = df.groupby("Scheme_Name")["Fraud"].mean().sort_values(ascending=False) * 100
    sns.barplot(x=rate.values, y=rate.index, ax=ax, palette="Reds_r")
    ax.set_title("Fraud Rate by Welfare Scheme")
    ax.set_xlabel("Fraud Rate (%)")
    _savefig(fig, "fraud_by_scheme.png")


def plot_fraud_by_district(df):
    fig, ax = plt.subplots(figsize=(9, 8))
    rate = df.groupby("District")["Fraud"].mean().sort_values(ascending=False) * 100
    sns.barplot(x=rate.values, y=rate.index, ax=ax, palette="Reds_r")
    ax.set_title("Fraud Rate by District")
    ax.set_xlabel("Fraud Rate (%)")
    _savefig(fig, "fraud_by_district.png")


def plot_fraud_by_gender(df):
    fig, ax = plt.subplots(figsize=(6, 5))
    rate = df.groupby("Gender")["Fraud"].mean() * 100
    sns.barplot(x=rate.index, y=rate.values, ax=ax, palette="Blues_d")
    ax.set_title("Fraud Rate by Gender")
    ax.set_ylabel("Fraud Rate (%)")
    _savefig(fig, "fraud_by_gender.png")


def plot_fraud_by_age(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x="Age", hue="Fraud", multiple="stack",
                 palette=PALETTE, bins=30, ax=ax)
    ax.set_title("Age Distribution by Fraud Status")
    _savefig(fig, "fraud_by_age.png")


def plot_fraud_by_income(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="Fraud", y="Income", palette=PALETTE, ax=ax)
    ax.set_xticklabels(["Genuine", "Fraud"])
    ax.set_title("Income Distribution by Fraud Status")
    _savefig(fig, "fraud_by_income.png")


def plot_fraud_by_claim_amount(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="Fraud", y="Claim_Amount", palette=PALETTE, ax=ax)
    ax.set_xticklabels(["Genuine", "Fraud"])
    ax.set_title("Claim Amount Distribution by Fraud Status")
    _savefig(fig, "fraud_by_claim_amount.png")


def plot_fraud_by_previous_fraud(df):
    fig, ax = plt.subplots(figsize=(6, 5))
    rate = df.groupby("Previous_Fraud")["Fraud"].mean() * 100
    ax.bar(["No Prior Fraud", "Prior Fraud"], rate.values, color=PALETTE)
    ax.set_title("Fraud Rate by Previous Fraud History")
    ax.set_ylabel("Fraud Rate (%)")
    _savefig(fig, "fraud_by_previous_fraud.png")


def plot_fraud_by_duplicate_aadhaar(df):
    fig, ax = plt.subplots(figsize=(6, 5))
    rate = df.groupby("Duplicate_Aadhaar")["Fraud"].mean() * 100
    ax.bar(["Unique Aadhaar", "Duplicate Aadhaar"], rate.values, color=PALETTE)
    ax.set_title("Fraud Rate by Duplicate Aadhaar")
    ax.set_ylabel("Fraud Rate (%)")
    _savefig(fig, "fraud_by_duplicate_aadhaar.png")


def plot_fraud_by_duplicate_mobile(df):
    fig, ax = plt.subplots(figsize=(6, 5))
    rate = df.groupby("Duplicate_Mobile")["Fraud"].mean() * 100
    ax.bar(["Unique Mobile", "Duplicate Mobile"], rate.values, color=PALETTE)
    ax.set_title("Fraud Rate by Duplicate Mobile Number")
    ax.set_ylabel("Fraud Rate (%)")
    _savefig(fig, "fraud_by_duplicate_mobile.png")


def plot_fraud_by_location_risk(df):
    fig, ax = plt.subplots(figsize=(6, 5))
    rate = df.groupby("Location_Risk")["Fraud"].mean().reindex(["Low", "Medium", "High"]) * 100
    ax.bar(rate.index, rate.values, color=["#4CAF50", "#FF9800", "#D64550"])
    ax.set_title("Fraud Rate by Location Risk Category")
    ax.set_ylabel("Fraud Rate (%)")
    _savefig(fig, "fraud_by_location_risk.png")


def plot_correlation_heatmap(df):
    engineered = engineer_features(df)
    numeric_cols = [
        "Age", "Income", "Claim_Amount", "Claim_Frequency", "Transaction_Count",
        "Duplicate_Aadhaar", "Duplicate_Mobile", "Previous_Fraud",
        "Duplicate_Identity_Risk", "Location_Risk_Score", "Combined_Risk_Score",
        "Fraud"
    ]
    corr = engineered[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
                cbar_kws={"label": "Correlation"})
    ax.set_title("Correlation Heatmap of Numerical Features")
    _savefig(fig, "correlation_heatmap.png")


def run_all(df):
    plot_fraud_distribution(df)
    plot_fraud_by_scheme(df)
    plot_fraud_by_district(df)
    plot_fraud_by_gender(df)
    plot_fraud_by_age(df)
    plot_fraud_by_income(df)
    plot_fraud_by_claim_amount(df)
    plot_fraud_by_previous_fraud(df)
    plot_fraud_by_duplicate_aadhaar(df)
    plot_fraud_by_duplicate_mobile(df)
    plot_fraud_by_location_risk(df)
    plot_correlation_heatmap(df)


def main():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. Run 'python generate_dataset.py' first."
        )
    df = pd.read_csv(DATASET_PATH)
    print(f"Generating EDA visualizations from {len(df):,} records ...")
    run_all(df)
    print("EDA complete. All charts saved to the visualizations/ folder.")


if __name__ == "__main__":
    main()
