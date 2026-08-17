"""
predict.py
----------
Prediction service for the AP welfare fraud-detection project.

Given raw beneficiary/claim details (the same fields collected on the
application form), this module:
  1. builds a single-row DataFrame,
  2. applies the exact same feature-engineering used during training,
  3. applies the exact same fitted preprocessor (encoding + scaling),
  4. runs the trained Random Forest model,
  5. returns the fraud prediction, fraud probability and a risk level.

This module is used both by the Flask web app (app.py) and can be run
directly from the command line for a quick manual test.
"""

import os
import joblib
import pandas as pd

from feature_engineering import engineer_features
from data_preprocessing import ALL_FEATURES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "random_forest_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.pkl")

REQUIRED_FIELDS = [
    "District", "Scheme_Name", "Gender", "Age", "Income", "Claim_Amount",
    "Application_Date", "Claim_Frequency", "Bank_Account",
    "Duplicate_Aadhaar", "Duplicate_Mobile", "Previous_Fraud",
    "Transaction_Count", "Location_Risk"
]


class FraudPredictionService:
    """Loads the trained artifacts once and serves repeated predictions."""

    def __init__(self, model_path: str = MODEL_PATH, preprocessor_path: str = PREPROCESSOR_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run 'python train_model.py' first."
            )
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(
                f"Preprocessor not found at {preprocessor_path}. Run 'python train_model.py' first."
            )
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)

    @staticmethod
    def _risk_level(probability: float) -> str:
        if probability < 0.35:
            return "Low Risk"
        elif probability < 0.70:
            return "Medium Risk"
        else:
            return "High Risk"

    @staticmethod
    def _validate_input(data: dict) -> None:
        missing = [f for f in REQUIRED_FIELDS if f not in data or data[f] in (None, "")]
        if missing:
            raise ValueError(f"Missing required field(s): {', '.join(missing)}")

    def _build_dataframe(self, data: dict) -> pd.DataFrame:
        row = {
            "District": data["District"],
            "Scheme_Name": data["Scheme_Name"],
            "Gender": data["Gender"],
            "Age": int(data["Age"]),
            "Income": float(data["Income"]),
            "Claim_Amount": float(data["Claim_Amount"]),
            "Application_Date": data["Application_Date"],
            "Claim_Frequency": int(data["Claim_Frequency"]),
            "Bank_Account": data["Bank_Account"],
            "Duplicate_Aadhaar": int(data["Duplicate_Aadhaar"]),
            "Duplicate_Mobile": int(data["Duplicate_Mobile"]),
            "Previous_Fraud": int(data["Previous_Fraud"]),
            "Transaction_Count": int(data["Transaction_Count"]),
            "Location_Risk": data["Location_Risk"],
        }
        return pd.DataFrame([row])

    def predict(self, data: dict) -> dict:
        """
        Run a fraud prediction for a single beneficiary claim.

        Parameters
        ----------
        data : dict
            Must contain all fields listed in REQUIRED_FIELDS.

        Returns
        -------
        dict with keys: prediction, prediction_label, fraud_probability,
        genuine_probability, risk_level, key_risk_indicators
        """
        self._validate_input(data)
        raw_df = self._build_dataframe(data)

        engineered_df = engineer_features(raw_df)
        X = engineered_df[ALL_FEATURES]

        X_transformed = self.preprocessor.transform(X)

        prediction = int(self.model.predict(X_transformed)[0])
        probabilities = self.model.predict_proba(X_transformed)[0]
        fraud_probability = float(probabilities[1])
        genuine_probability = float(probabilities[0])

        risk_level = self._risk_level(fraud_probability)

        key_indicators = []
        if int(data["Duplicate_Aadhaar"]) == 1:
            key_indicators.append("Duplicate Aadhaar number detected")
        if int(data["Duplicate_Mobile"]) == 1:
            key_indicators.append("Duplicate mobile number detected")
        if int(data["Previous_Fraud"]) == 1:
            key_indicators.append("Previous fraud history on record")
        if int(data["Claim_Frequency"]) >= 6:
            key_indicators.append("Abnormally high claim frequency")
        if int(data["Transaction_Count"]) >= 10:
            key_indicators.append("High transaction count")
        if data["Location_Risk"] == "High":
            key_indicators.append("Application from a high-risk location")
        if not key_indicators:
            key_indicators.append("No major individual risk indicators triggered")

        return {
            "prediction": prediction,
            "prediction_label": "Fraud" if prediction == 1 else "Genuine",
            "fraud_probability": round(fraud_probability * 100, 2),
            "genuine_probability": round(genuine_probability * 100, 2),
            "risk_level": risk_level,
            "key_risk_indicators": key_indicators,
        }


if __name__ == "__main__":
    # Quick manual test
    service = FraudPredictionService()

    sample_input = {
        "District": "Guntur",
        "Scheme_Name": "NTR Bharosa Pension",
        "Gender": "Female",
        "Age": 68,
        "Income": 45000,
        "Claim_Amount": 3200,
        "Application_Date": "2026-05-14",
        "Claim_Frequency": 9,
        "Bank_Account": "Yes",
        "Duplicate_Aadhaar": 1,
        "Duplicate_Mobile": 1,
        "Previous_Fraud": 1,
        "Transaction_Count": 14,
        "Location_Risk": "High",
    }

    result = service.predict(sample_input)
    print("Sample prediction (high-risk profile):")
    for k, v in result.items():
        print(f"  {k}: {v}")

    sample_input_2 = {
        "District": "Chittoor",
        "Scheme_Name": "Deepam 2.0",
        "Gender": "Male",
        "Age": 34,
        "Income": 180000,
        "Claim_Amount": 700,
        "Application_Date": "2026-03-02",
        "Claim_Frequency": 1,
        "Bank_Account": "Yes",
        "Duplicate_Aadhaar": 0,
        "Duplicate_Mobile": 0,
        "Previous_Fraud": 0,
        "Transaction_Count": 2,
        "Location_Risk": "Low",
    }

    result_2 = service.predict(sample_input_2)
    print("\nSample prediction (low-risk profile):")
    for k, v in result_2.items():
        print(f"  {k}: {v}")
