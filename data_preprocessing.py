"""
data_preprocessing.py
----------------------
Handles all data-cleaning and preprocessing logic for the AP welfare
fraud-detection project:

  - loads the raw CSV dataset
  - checks missing values
  - removes exact duplicate rows
  - applies feature engineering (feature_engineering.py)
  - builds a single scikit-learn ColumnTransformer that one-hot-encodes
    categorical columns and standard-scales numerical columns
  - splits the data into train/test sets BEFORE fitting the transformer
    (fit only on training data) to avoid data leakage
  - saves the fitted preprocessing pipeline so predict.py / app.py can
    reuse the exact same transformation at inference time
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from feature_engineering import engineer_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "ap_welfare_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.pkl")

RANDOM_STATE = 42

TARGET_COLUMN = "Fraud"

# Columns that must NEVER be used as model features (identifiers / raw text)
ID_COLUMNS = ["Beneficiary_ID", "Aadhaar_No", "Mobile_No", "Village", "Application_Date"]

# Categorical columns fed into the model (raw + engineered)
CATEGORICAL_FEATURES = [
    "District", "Scheme_Name", "Gender", "Bank_Account", "Location_Risk",
    "Age_Group", "Income_Category", "Claim_Amount_Category",
    "Claim_Frequency_Risk", "Transaction_Risk"
]

# Numerical columns fed into the model (raw + engineered)
NUMERICAL_FEATURES = [
    "Age", "Income", "Claim_Amount", "Claim_Frequency", "Transaction_Count",
    "Duplicate_Aadhaar", "Duplicate_Mobile", "Previous_Fraud",
    "Duplicate_Identity_Risk", "Previous_Fraud_Risk", "Location_Risk_Score",
    "Combined_Risk_Score", "Application_Month", "Application_Quarter",
    "Application_DayOfWeek", "Days_Since_Application"
]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES


def load_data(path: str = DATASET_PATH) -> pd.DataFrame:
    """Load the raw CSV dataset from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run 'python generate_dataset.py' first."
        )
    return pd.read_csv(path)


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """Return a Series of missing-value counts per column."""
    return df.isnull().sum()


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
      - drop fully duplicated rows
      - drop rows with missing values in critical columns (should be none
        in the synthetic dataset, but this keeps the pipeline robust for
        any real-world CSV that is dropped in later).
    """
    cleaned = df.drop_duplicates().copy()
    cleaned = cleaned.dropna(subset=[c for c in df.columns if c != TARGET_COLUMN or True])
    return cleaned.reset_index(drop=True)


def build_preprocessor() -> ColumnTransformer:
    """Create (but do not fit) the ColumnTransformer used to encode/scale features."""
    categorical_pipeline = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    numerical_pipeline = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ("num", numerical_pipeline, NUMERICAL_FEATURES),
    ])
    return preprocessor


def get_feature_target_split(df: pd.DataFrame):
    """
    Apply feature engineering, then separate features (X) and target (y).
    Only ALL_FEATURES columns are returned in X to guarantee predict.py
    builds an identical feature set at inference time.
    """
    engineered = engineer_features(df)
    X = engineered[ALL_FEATURES].copy()
    y = engineered[TARGET_COLUMN].copy()
    return X, y


def train_test_split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    """Stratified train/test split so the fraud ratio is preserved in both sets."""
    return train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )


def fit_transform_preprocessor(preprocessor: ColumnTransformer, X_train: pd.DataFrame):
    """Fit the preprocessor ONLY on training data, then transform it."""
    X_train_transformed = preprocessor.fit_transform(X_train)
    return preprocessor, X_train_transformed


def save_preprocessor(preprocessor: ColumnTransformer, path: str = PREPROCESSOR_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(preprocessor, path)
    print(f"Preprocessor saved -> {path}")


def load_preprocessor(path: str = PREPROCESSOR_PATH) -> ColumnTransformer:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Preprocessor not found at {path}. Run 'python train_model.py' first."
        )
    return joblib.load(path)


def full_preprocessing_pipeline(path: str = DATASET_PATH, test_size: float = 0.2):
    """
    Convenience function used by train_model.py:
      load -> clean -> feature engineer -> split -> fit preprocessor -> transform
    Returns transformed train/test arrays, raw (untransformed) splits, target
    splits, and the fitted preprocessor.
    """
    df = load_data(path)
    print(f"Loaded dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")

    missing = check_missing_values(df)
    print(f"Missing values found: {missing.sum()}")

    df = clean_data(df)
    print(f"After removing duplicates: {df.shape[0]:,} rows")

    X, y = get_feature_target_split(df)

    X_train, X_test, y_train, y_test = train_test_split_data(X, y, test_size=test_size)
    print(f"Train size: {X_train.shape[0]:,} | Test size: {X_test.shape[0]:,}")

    preprocessor = build_preprocessor()
    preprocessor, X_train_transformed = fit_transform_preprocessor(preprocessor, X_train)
    X_test_transformed = preprocessor.transform(X_test)

    return {
        "X_train_raw": X_train, "X_test_raw": X_test,
        "X_train": X_train_transformed, "X_test": X_test_transformed,
        "y_train": y_train, "y_test": y_test,
        "preprocessor": preprocessor,
    }


if __name__ == "__main__":
    result = full_preprocessing_pipeline()
    save_preprocessor(result["preprocessor"])
    print("Preprocessing pipeline executed successfully.")
    print(f"Transformed feature matrix shape (train): {result['X_train'].shape}")
