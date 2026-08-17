"""
train_model.py
---------------
Trains a Random Forest Classifier on the preprocessed AP welfare-scheme
dataset to detect fraudulent claims, then saves:

  - the trained model            -> models/random_forest_model.pkl
  - the fitted preprocessor      -> models/preprocessor.pkl
  - evaluation metrics (JSON)    -> models/metrics.json
  - the feature name list        -> models/feature_names.json

Run:
    python train_model.py
"""

import os
import json
import time
import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from data_preprocessing import full_preprocessing_pipeline, save_preprocessor, ALL_FEATURES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "random_forest_model.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.json")

RANDOM_STATE = 42


def train_random_forest(X_train, y_train):
    """Train a Random Forest Classifier with reproducible, well-suited parameters."""
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=16,
        min_samples_split=6,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",   # helps with the ~10% fraud class imbalance
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    print("Training Random Forest Classifier ...")
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"Training complete in {elapsed:.2f} seconds.")
    return model


def evaluate_model(model, X_test, y_test):
    """Compute the core evaluation metrics required by the project spec."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    cm = confusion_matrix(y_test, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["true_negatives"] = int(cm[0][0])
    metrics["false_positives"] = int(cm[0][1])
    metrics["false_negatives"] = int(cm[1][0])
    metrics["true_positives"] = int(cm[1][1])

    metrics["classification_report"] = classification_report(
        y_test, y_pred, target_names=["Genuine", "Fraud"], zero_division=0
    )

    return metrics, y_pred, y_proba


def get_output_feature_names(preprocessor):
    """Return the expanded feature names after one-hot encoding + scaling."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return ALL_FEATURES


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=" * 70)
    print("STEP 1: Preprocessing dataset")
    print("=" * 70)
    result = full_preprocessing_pipeline()

    X_train, X_test = result["X_train"], result["X_test"]
    y_train, y_test = result["y_train"], result["y_test"]
    preprocessor = result["preprocessor"]

    print("\n" + "=" * 70)
    print("STEP 2: Training Random Forest model")
    print("=" * 70)
    model = train_random_forest(X_train, y_train)

    print("\n" + "=" * 70)
    print("STEP 3: Evaluating model on the held-out test set")
    print("=" * 70)
    metrics, y_pred, y_proba = evaluate_model(model, X_test, y_test)

    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1 Score  : {metrics['f1_score']:.4f}")
    print(f"ROC-AUC   : {metrics['roc_auc']:.4f}")
    print("\nConfusion Matrix:")
    print(np.array(metrics["confusion_matrix"]))
    print(f"\nFalse Positives (genuine flagged as fraud): {metrics['false_positives']}")
    print(f"False Negatives (fraud missed by the model): {metrics['false_negatives']}")
    print("\nClassification Report:")
    print(metrics["classification_report"])

    print("\n" + "=" * 70)
    print("STEP 4: Saving model, preprocessor and metrics")
    print("=" * 70)

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved -> {MODEL_PATH}")

    save_preprocessor(preprocessor)

    feature_names = get_output_feature_names(preprocessor)
    with open(FEATURE_NAMES_PATH, "w") as f:
        json.dump(feature_names, f, indent=2)
    print(f"Feature names saved -> {FEATURE_NAMES_PATH}")

    metrics_to_save = {k: v for k, v in metrics.items() if k != "classification_report"}
    metrics_to_save["classification_report"] = metrics["classification_report"]
    metrics_to_save["n_train_samples"] = int(X_train.shape[0])
    metrics_to_save["n_test_samples"] = int(X_test.shape[0])
    metrics_to_save["n_features"] = int(X_train.shape[1])

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f"Metrics saved -> {METRICS_PATH}")

    print("\nTraining pipeline finished successfully.")


if __name__ == "__main__":
    main()
