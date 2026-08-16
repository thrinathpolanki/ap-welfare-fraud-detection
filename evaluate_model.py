"""
evaluate_model.py
------------------
Loads the trained Random Forest model + preprocessor, reproduces the exact
same train/test split used during training (same random_state), and
generates evaluation visualizations:

  - confusion matrix heatmap   -> visualizations/confusion_matrix.png
  - ROC curve                  -> visualizations/roc_curve.png
  - feature importance chart   -> visualizations/feature_importance.png

It also re-prints the core metrics so results can be verified independently
of train_model.py.

Run:
    python evaluate_model.py
"""

import os
import json
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

from data_preprocessing import full_preprocessing_pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
VIS_DIR = os.path.join(BASE_DIR, "visualizations")
MODEL_PATH = os.path.join(MODELS_DIR, "random_forest_model.pkl")


def load_trained_model(path: str = MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Trained model not found at {path}. Run 'python train_model.py' first."
        )
    return joblib.load(path)


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Genuine", "Fraud"], yticklabels=["Genuine", "Fraud"])
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix - Random Forest Classifier")
    os.makedirs(VIS_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(VIS_DIR, "confusion_matrix.png"), dpi=140)
    plt.close(fig)
    print(f"Saved -> {os.path.join(VIS_DIR, 'confusion_matrix.png')}")


def plot_roc_curve(y_test, y_proba):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#0B3D91", linewidth=2, label=f"ROC Curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - Random Forest Classifier")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(VIS_DIR, "roc_curve.png"), dpi=140)
    plt.close(fig)
    print(f"Saved -> {os.path.join(VIS_DIR, 'roc_curve.png')}")


def plot_feature_importance(model, feature_names, top_n=20):
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in order]
    top_importances = importances[order]

    fig, ax = plt.subplots(figsize=(9, 8))
    sns.barplot(x=top_importances, y=top_features, ax=ax, palette="viridis")
    ax.set_title(f"Top {top_n} Feature Importances - Random Forest")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(os.path.join(VIS_DIR, "feature_importance.png"), dpi=140)
    plt.close(fig)
    print(f"Saved -> {os.path.join(VIS_DIR, 'feature_importance.png')}")


def main():
    print("Reproducing the training pipeline's train/test split for evaluation ...")
    result = full_preprocessing_pipeline()
    X_test, y_test = result["X_test"], result["y_test"]
    preprocessor = result["preprocessor"]

    model = load_trained_model()

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n--- Evaluation Metrics ---")
    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision : {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall    : {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1 Score  : {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"ROC-AUC   : {roc_auc_score(y_test, y_proba):.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    print(f"False Positives: {cm[0][1]}  |  False Negatives: {cm[1][0]}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Genuine", "Fraud"], zero_division=0))

    print("\nGenerating evaluation visualizations ...")
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_proba)

    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]

    plot_feature_importance(model, feature_names)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
