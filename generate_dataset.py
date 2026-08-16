"""
generate_dataset.py
--------------------
Generates a realistic synthetic dataset for the
"Fraud Detection in Andhra Pradesh Welfare Schemes Using Machine Learning" project.

The dataset simulates 100,000 beneficiary records across four AP welfare schemes:
    - Thalliki Vandanam
    - NTR Bharosa Pension
    - Annadata
    - Deepam 2.0

Fraud labels are NOT random. They are generated using meaningful, logically
correlated risk factors (duplicate identity, previous fraud history, abnormal
claim behaviour, high transaction counts and high location risk) so that a
machine learning model can genuinely learn to separate fraud from genuine claims.

Run:
    python generate_dataset.py
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

N_RECORDS = 100_000

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "ap_welfare_dataset.csv")

# ----------------------------------------------------------------------
# Reference data (realistic Andhra Pradesh districts / villages / schemes)
# ----------------------------------------------------------------------
DISTRICTS = [
    "Srikakulam", "Vizianagaram", "Visakhapatnam", "Anakapalli", "Kakinada",
    "East Godavari", "West Godavari", "Konaseema", "Eluru", "Krishna",
    "NTR", "Guntur", "Palnadu", "Bapatla", "Prakasam", "Nellore",
    "Kurnool", "Nandyal", "Anantapur", "Sri Sathya Sai", "YSR Kadapa",
    "Annamayya", "Chittoor", "Tirupati"
]

VILLAGES_BY_DISTRICT = {
    district: [f"{district} Village {i}" for i in range(1, 13)]
    for district in DISTRICTS
}
# Add a few realistic named villages/mandal-towns to make data feel authentic
REALISTIC_VILLAGE_NAMES = [
    "Amalapuram", "Yanam", "Bhimavaram", "Tanuku", "Palakollu", "Narsapuram",
    "Machilipatnam", "Gudivada", "Vuyyuru", "Mangalagiri", "Tenali",
    "Chirala", "Ongole", "Kandukur", "Gudur", "Kavali", "Adoni",
    "Nandikotkur", "Dharmavaram", "Hindupur", "Madanapalle", "Punganur",
    "Srikalahasti", "Puttur", "Rajam", "Palasa", "Bobbili", "Anakapalle"
]
for d in DISTRICTS:
    VILLAGES_BY_DISTRICT[d].extend(random.sample(REALISTIC_VILLAGE_NAMES, 3))

SCHEMES = ["Thalliki Vandanam", "NTR Bharosa Pension", "Annadata", "Deepam 2.0"]

# Scheme specific realistic parameters (min age, max age, income cap, typical claim range)
SCHEME_PARAMS = {
    "Thalliki Vandanam": {"age_range": (20, 45), "income_cap": 250000, "claim_range": (12000, 15000)},
    "NTR Bharosa Pension": {"age_range": (60, 90), "income_cap": 150000, "claim_range": (2500, 4000)},
    "Annadata": {"age_range": (25, 70), "income_cap": 300000, "claim_range": (10000, 13500)},
    "Deepam 2.0": {"age_range": (18, 75), "income_cap": 400000, "claim_range": (500, 1200)},
}

GENDERS = ["Male", "Female", "Other"]


def random_date(start_year=2023, end_year=2026):
    """Generate a random application date between start_year and end_year."""
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 6, 30)
    delta_days = (end - start).days
    return start + timedelta(days=random.randint(0, delta_days))


def generate_aadhaar():
    """Generate a random 12-digit Aadhaar-like number as a string."""
    return "".join([str(random.randint(0, 9)) for _ in range(12)])


def generate_mobile():
    """Generate a random 10-digit Indian-style mobile number."""
    first_digit = random.choice(["6", "7", "8", "9"])
    rest = "".join([str(random.randint(0, 9)) for _ in range(9)])
    return first_digit + rest


def build_dataset(n_records=N_RECORDS):
    print(f"Generating {n_records:,} synthetic welfare-scheme records ...")

    # ------------------------------------------------------------------
    # Step 1: Create a pool of "identity" values so a controlled fraction
    # can be intentionally duplicated (this is what drives duplicate-based
    # fraud signals rather than pure randomness).
    # ------------------------------------------------------------------
    n_unique_aadhaar = int(n_records * 0.93)   # ~7% of aadhaar numbers repeat
    n_unique_mobile = int(n_records * 0.90)    # ~10% of mobile numbers repeat

    aadhaar_pool = [generate_aadhaar() for _ in range(n_unique_aadhaar)]
    mobile_pool = [generate_mobile() for _ in range(n_unique_mobile)]

    aadhaar_choices = np.random.choice(aadhaar_pool, size=n_records)
    mobile_choices = np.random.choice(mobile_pool, size=n_records)

    records = []

    for i in range(n_records):
        scheme = random.choice(SCHEMES)
        params = SCHEME_PARAMS[scheme]

        district = random.choice(DISTRICTS)
        village = random.choice(VILLAGES_BY_DISTRICT[district])
        gender = random.choices(GENDERS, weights=[0.48, 0.50, 0.02])[0]

        age = int(np.clip(np.random.normal(
            (params["age_range"][0] + params["age_range"][1]) / 2,
            8
        ), params["age_range"][0], params["age_range"][1]))

        income = int(np.clip(np.random.normal(params["income_cap"] * 0.4, params["income_cap"] * 0.25),
                              10000, params["income_cap"] * 1.3))

        base_claim = np.random.uniform(params["claim_range"][0], params["claim_range"][1])

        application_date = random_date()

        claim_frequency = np.random.poisson(1.4) + 1          # typical 1-4 claims/year
        transaction_count = np.random.poisson(3) + 1            # typical small number of transactions
        bank_account = random.choices(["Yes", "No"], weights=[0.92, 0.08])[0]

        # Duplicate flags derived from whether this record's aadhaar/mobile
        # value occurs more than once in the generated pool.
        duplicate_aadhaar_flag = 0  # filled after full generation (needs counts)
        duplicate_mobile_flag = 0

        previous_fraud = random.choices([0, 1], weights=[0.92, 0.08])[0]
        location_risk = random.choices(
            ["Low", "Medium", "High"], weights=[0.65, 0.25, 0.10]
        )[0]

        records.append({
            "Beneficiary_ID": f"AP-BEN-{100000 + i}",
            "Aadhaar_No": aadhaar_choices[i],
            "Mobile_No": mobile_choices[i],
            "District": district,
            "Village": village,
            "Scheme_Name": scheme,
            "Gender": gender,
            "Age": age,
            "Income": income,
            "Claim_Amount": round(base_claim, 2),
            "Application_Date": application_date.strftime("%Y-%m-%d"),
            "Claim_Frequency": claim_frequency,
            "Bank_Account": bank_account,
            "Duplicate_Aadhaar": duplicate_aadhaar_flag,
            "Duplicate_Mobile": duplicate_mobile_flag,
            "Previous_Fraud": previous_fraud,
            "Transaction_Count": transaction_count,
            "Location_Risk": location_risk,
        })

    df = pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Step 2: Compute genuine duplicate flags from the actual occurrence
    # counts of Aadhaar / Mobile numbers in the generated data.
    # ------------------------------------------------------------------
    aadhaar_counts = df["Aadhaar_No"].value_counts()
    mobile_counts = df["Mobile_No"].value_counts()

    df["Duplicate_Aadhaar"] = df["Aadhaar_No"].map(lambda x: 1 if aadhaar_counts[x] > 1 else 0)
    df["Duplicate_Mobile"] = df["Mobile_No"].map(lambda x: 1 if mobile_counts[x] > 1 else 0)

    # ------------------------------------------------------------------
    # Step 3: Inject abnormal behaviour for a subset of records so that
    # "suspicious combinations" genuinely exist in the data (rather than
    # relying only on the base random distributions above).
    # ------------------------------------------------------------------
    suspicious_idx = df.sample(frac=0.12, random_state=RANDOM_STATE).index
    df.loc[suspicious_idx, "Claim_Frequency"] = np.random.randint(6, 15, size=len(suspicious_idx))
    df.loc[suspicious_idx, "Transaction_Count"] = np.random.randint(10, 25, size=len(suspicious_idx))

    high_claim_idx = df.sample(frac=0.08, random_state=RANDOM_STATE + 1).index
    df.loc[high_claim_idx, "Claim_Amount"] = df.loc[high_claim_idx, "Claim_Amount"] * np.random.uniform(3, 6)

    # ------------------------------------------------------------------
    # Step 4: Compute a weighted "fraud risk score" from meaningful signals.
    # This score is the mechanism that ties the Fraud label to real feature
    # patterns instead of being fully random.
    # ------------------------------------------------------------------
    location_risk_score_map = {"Low": 0.0, "Medium": 0.4, "High": 0.8}

    claim_freq_norm = (df["Claim_Frequency"] - df["Claim_Frequency"].min()) / \
                       (df["Claim_Frequency"].max() - df["Claim_Frequency"].min())
    txn_norm = (df["Transaction_Count"] - df["Transaction_Count"].min()) / \
               (df["Transaction_Count"].max() - df["Transaction_Count"].min())
    claim_amt_norm = (df["Claim_Amount"] - df["Claim_Amount"].min()) / \
                      (df["Claim_Amount"].max() - df["Claim_Amount"].min())

    fraud_score = (
        df["Duplicate_Aadhaar"] * 0.28 +
        df["Duplicate_Mobile"] * 0.18 +
        df["Previous_Fraud"] * 0.22 +
        claim_freq_norm * 0.14 +
        txn_norm * 0.10 +
        claim_amt_norm * 0.08 +
        df["Location_Risk"].map(location_risk_score_map) * 0.20
    )

    # Add small random noise so the boundary is not perfectly deterministic
    # (mirrors real-world label noise while keeping the signal strong).
    fraud_score = fraud_score + np.random.normal(0, 0.05, size=len(df))

    # Convert score to a binary label using a threshold calibrated to give
    # a realistic overall fraud rate (~9-11%).
    threshold = np.quantile(fraud_score, 0.90)
    df["Fraud"] = (fraud_score >= threshold).astype(int)

    # Reorder columns to match the specification exactly (19 columns)
    column_order = [
        "Beneficiary_ID", "Aadhaar_No", "Mobile_No", "District", "Village",
        "Scheme_Name", "Gender", "Age", "Income", "Claim_Amount",
        "Application_Date", "Claim_Frequency", "Bank_Account",
        "Duplicate_Aadhaar", "Duplicate_Mobile", "Previous_Fraud",
        "Transaction_Count", "Location_Risk", "Fraud"
    ]
    df = df[column_order]

    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = build_dataset(N_RECORDS)

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Dataset generated successfully -> {OUTPUT_FILE}")
    print(f"Records: {len(df):,} | Columns: {df.shape[1]}")
    print("Fraud distribution:")
    print(df["Fraud"].value_counts())
    print(f"Fraud rate: {df['Fraud'].mean() * 100:.2f}%")


if __name__ == "__main__":
    main()
