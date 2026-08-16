"""
feature_engineering.py
-----------------------
Generates engineered features on top of the raw AP welfare dataset.

Original (raw) columns collected at the point of application are kept
untouched. This module ADDS new, clearly-named engineered columns that
capture risk patterns the Random Forest model can use. Both training
(train_model.py) and prediction (predict.py) call `engineer_features()`
so that the exact same transformation logic is used everywhere and no
data leakage / inconsistency is introduced.
"""

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# ORIGINAL (raw, collected-at-source) columns
# ----------------------------------------------------------------------
RAW_COLUMNS = [
    "District", "Village", "Scheme_Name", "Gender", "Age", "Income",
    "Claim_Amount", "Application_Date", "Claim_Frequency", "Bank_Account",
    "Duplicate_Aadhaar", "Duplicate_Mobile", "Previous_Fraud",
    "Transaction_Count", "Location_Risk"
]

# ----------------------------------------------------------------------
# ENGINEERED (derived) columns created by this module
# ----------------------------------------------------------------------
ENGINEERED_COLUMNS = [
    "Age_Group", "Income_Category", "Claim_Amount_Category",
    "Claim_Frequency_Risk", "Transaction_Risk", "Duplicate_Identity_Risk",
    "Previous_Fraud_Risk", "Location_Risk_Score", "Combined_Risk_Score",
    "Application_Month", "Application_Quarter", "Application_DayOfWeek",
    "Days_Since_Application"
]


def _age_group(age):
    if age < 18:
        return "Minor"
    elif age <= 30:
        return "Young Adult"
    elif age <= 45:
        return "Adult"
    elif age <= 60:
        return "Middle Age"
    else:
        return "Senior"


def _income_category(income):
    if income <= 50000:
        return "Very Low"
    elif income <= 150000:
        return "Low"
    elif income <= 300000:
        return "Medium"
    else:
        return "High"


def _claim_amount_category(amount):
    if amount <= 2000:
        return "Small"
    elif amount <= 10000:
        return "Moderate"
    elif amount <= 20000:
        return "Large"
    else:
        return "Very Large"


def _claim_frequency_risk(freq):
    if freq <= 2:
        return "Low"
    elif freq <= 5:
        return "Medium"
    else:
        return "High"


def _transaction_risk(count):
    if count <= 4:
        return "Low"
    elif count <= 9:
        return "Medium"
    else:
        return "High"


def engineer_features(df: pd.DataFrame, reference_date: str = "2026-07-01") -> pd.DataFrame:
    """
    Add engineered features to a copy of the input dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the RAW_COLUMNS listed above.
    reference_date : str
        Fixed reference date used to compute "days since application" so
        that the feature is reproducible for both training and inference
        (using "today" at inference time would silently change the
        feature distribution the model was trained on).

    Returns
    -------
    pd.DataFrame with original columns + engineered columns appended.
    """
    data = df.copy()

    # ---- date-based features -----------------------------------------
    data["Application_Date"] = pd.to_datetime(data["Application_Date"])
    ref = pd.to_datetime(reference_date)

    data["Application_Month"] = data["Application_Date"].dt.month
    data["Application_Quarter"] = data["Application_Date"].dt.quarter
    data["Application_DayOfWeek"] = data["Application_Date"].dt.dayofweek
    data["Days_Since_Application"] = (ref - data["Application_Date"]).dt.days
    # Guard against negative values if a future date slips through
    data["Days_Since_Application"] = data["Days_Since_Application"].clip(lower=0)

    # ---- categorical risk buckets --------------------------------------
    data["Age_Group"] = data["Age"].apply(_age_group)
    data["Income_Category"] = data["Income"].apply(_income_category)
    data["Claim_Amount_Category"] = data["Claim_Amount"].apply(_claim_amount_category)
    data["Claim_Frequency_Risk"] = data["Claim_Frequency"].apply(_claim_frequency_risk)
    data["Transaction_Risk"] = data["Transaction_Count"].apply(_transaction_risk)

    # ---- combined identity / history risk indicators -------------------
    data["Duplicate_Identity_Risk"] = data["Duplicate_Aadhaar"].astype(int) + \
        data["Duplicate_Mobile"].astype(int)  # 0, 1, or 2

    data["Previous_Fraud_Risk"] = data["Previous_Fraud"].astype(int)

    location_score_map = {"Low": 0, "Medium": 1, "High": 2}
    data["Location_Risk_Score"] = data["Location_Risk"].map(location_score_map).fillna(0).astype(int)

    # ---- combined fraud-risk indicator (weighted sum of key signals) ---
    freq_risk_map = {"Low": 0, "Medium": 1, "High": 2}
    txn_risk_map = {"Low": 0, "Medium": 1, "High": 2}

    data["Combined_Risk_Score"] = (
        data["Duplicate_Identity_Risk"] * 2 +
        data["Previous_Fraud_Risk"] * 2 +
        data["Claim_Frequency_Risk"].map(freq_risk_map) +
        data["Transaction_Risk"].map(txn_risk_map) +
        data["Location_Risk_Score"]
    )

    return data


if __name__ == "__main__":
    # Quick self-test when run directly
    import os
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "dataset", "ap_welfare_dataset.csv")
    sample = pd.read_csv(dataset_path, nrows=5)
    engineered = engineer_features(sample)
    print(engineered[ENGINEERED_COLUMNS])
