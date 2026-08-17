# Fraud Detection in Andhra Pradesh Welfare Schemes Using Machine Learning

An end-to-end machine learning project that detects potentially fraudulent
welfare-scheme claims across four Andhra Pradesh government schemes —
**Thalliki Vandanam**, **NTR Bharosa Pension**, **Annadata**, and
**Deepam 2.0** — using a Random Forest Classifier, wrapped in a Flask web
dashboard.

---

## Abstract

Public welfare schemes disburse large sums of money to millions of citizens.
Fraud in the form of duplicate identities, inflated claim frequency, and
coordinated abuse quietly drains funds meant for genuine beneficiaries.
This project builds a complete, reproducible ML pipeline — synthetic dataset
generation, preprocessing, exploratory data analysis, feature engineering,
model training, evaluation, and a live prediction dashboard — to flag
high-risk welfare claims before payout.

## Problem Statement

Manual verification of welfare claims does not scale to the volume of
beneficiaries enrolled across AP's welfare schemes. Fraud indicators such as
duplicate Aadhaar/mobile numbers, prior fraud history, and abnormal claim
behaviour are scattered across records and hard to spot by eye. This project
automates that detection using supervised machine learning.

## Objectives

- Generate a realistic, logically-correlated synthetic dataset of welfare
  claims (not randomly labeled).
- Build a clean, leakage-free preprocessing and feature-engineering pipeline.
- Train and evaluate a Random Forest Classifier to detect fraud.
- Serve real-time fraud predictions with a probability score and risk level.
- Present the results in a professional, demonstrable Flask dashboard.

## Dataset Description

- **Records:** 100,000
- **Columns:** 19 (`Beneficiary_ID`, `Aadhaar_No`, `Mobile_No`, `District`,
  `Village`, `Scheme_Name`, `Gender`, `Age`, `Income`, `Claim_Amount`,
  `Application_Date`, `Claim_Frequency`, `Bank_Account`, `Duplicate_Aadhaar`,
  `Duplicate_Mobile`, `Previous_Fraud`, `Transaction_Count`,
  `Location_Risk`, `Fraud`)
- **Schemes covered:** Thalliki Vandanam, NTR Bharosa Pension, Annadata,
  Deepam 2.0
- **Geography:** 24 real Andhra Pradesh districts with generated village
  names.
- **Fraud generation logic:** the `Fraud` label is produced from a weighted
  risk score combining duplicate Aadhaar/mobile numbers, previous fraud
  history, claim-frequency and transaction-count anomalies, claim-amount
  spikes, and location risk — with a small amount of noise. This produces a
  learnable ~10% fraud rate rather than a random label.

## Features

- Synthetic dataset generator producing all 19 required columns.
- Full preprocessing pipeline (cleaning, encoding, scaling, leak-free split).
- Dataset structural/statistical analysis report.
- 12 exploratory data analysis visualizations.
- 13 engineered risk features layered on top of the raw fields.
- Random Forest Classifier with reproducible training.
- Full evaluation suite: accuracy, precision, recall, F1, ROC-AUC, confusion
  matrix, classification report, ROC curve, and feature-importance chart.
- Reusable prediction service with a 3-tier risk level (Low / Medium / High).
- Flask dashboard: overview stats, live prediction form, analytics page, and
  an about/documentation page.

## Technologies

| Layer            | Technology                                |
|-------------------|--------------------------------------------|
| Data / ML         | Python, Pandas, NumPy, scikit-learn         |
| Visualization      | Matplotlib, Seaborn, Chart.js               |
| Web application    | Flask, Jinja2                               |
| Frontend           | HTML5, CSS3, vanilla JavaScript             |
| Model persistence  | joblib                                      |

No external database (no Firebase, MongoDB, MySQL, Supabase). All data lives
in a CSV file and serialized model artifacts on disk.

## Project Architecture

```
AP_Welfare_Fraud_Detection/
│
├── dataset/
│   └── ap_welfare_dataset.csv        # generated 100,000-row dataset
│
├── models/
│   ├── random_forest_model.pkl       # trained classifier
│   ├── preprocessor.pkl              # fitted ColumnTransformer
│   ├── metrics.json                  # evaluation metrics (from real run)
│   └── feature_names.json            # expanded model feature names
│
├── visualizations/                   # EDA + evaluation charts (PNG)
│
├── templates/
│   ├── base.html
│   ├── index.html                    # dashboard
│   ├── prediction.html               # claim check form + result
│   ├── analytics.html                # deeper fraud breakdowns
│   └── about.html
│
├── static/
│   ├── css/style.css
│   └── js/script.js
│
├── generate_dataset.py
├── data_preprocessing.py
├── dataset_analysis.py
├── eda.py
├── feature_engineering.py
├── train_model.py
├── evaluate_model.py
├── predict.py
├── app.py
├── requirements.txt
└── README.md
```

## Machine Learning Workflow

1. **`generate_dataset.py`** — creates the 100,000-record synthetic dataset.
2. **`dataset_analysis.py`** — prints a structural/statistical summary.
3. **`eda.py`** — generates 12 exploratory visualizations into
   `visualizations/`.
4. **`feature_engineering.py`** — adds 13 engineered risk features
   (age/income/claim bands, combined risk score, date-derived features).
5. **`data_preprocessing.py`** — cleans data, builds a
   `ColumnTransformer` (one-hot encoding + scaling), and performs a
   stratified train/test split, fitting the transformer on the training
   split only (no leakage).
6. **`train_model.py`** — trains a `RandomForestClassifier`
   (300 trees, balanced class weights, `random_state=42`), evaluates it on
   the held-out test set, and saves the model, preprocessor, and metrics.
7. **`evaluate_model.py`** — reproduces the same split to generate a
   confusion matrix heatmap, ROC curve, and feature-importance chart.
8. **`predict.py`** — a `FraudPredictionService` class that applies the
   exact same feature engineering + preprocessing to a single new claim and
   returns a prediction, probability, and risk level.
9. **`app.py`** — the Flask dashboard that ties everything together.

## Installation

```bash
# 1. Clone / unzip the project, then move into the project folder
cd AP_Welfare_Fraud_Detection

# 2. (Recommended) create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

## Dataset Generation

```bash
python generate_dataset.py
```
Produces `dataset/ap_welfare_dataset.csv` (100,000 rows × 19 columns).

## Dataset Analysis & EDA

```bash
python dataset_analysis.py   # prints a full dataset summary to the console
python eda.py                 # saves 12 charts to visualizations/
```

## Training Instructions

```bash
python train_model.py
```
This runs the full preprocessing pipeline, trains the Random Forest model,
prints the evaluation metrics, and saves the model, preprocessor, feature
names, and metrics into `models/`.

## Evaluation Instructions

```bash
python evaluate_model.py
```
Reproduces the same train/test split, reprints all metrics, and saves the
confusion matrix, ROC curve, and feature-importance charts to
`visualizations/`.

## Running a Manual Prediction (no web server)

```bash
python predict.py
```
Runs two example predictions (a high-risk profile and a low-risk profile)
directly from the command line.

## Flask Execution Instructions

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

> If the dataset or trained model don't exist yet, `app.py` automatically
> runs `generate_dataset.py` and `train_model.py` on first launch, so the
> whole project can be demonstrated from a single command.

### Pages

- **`/`** — Dashboard: total beneficiaries, genuine/fraud counts, fraud
  percentage, high-risk case count, and live model-performance seals.
- **`/predict`** — Fraud Prediction: fill in a claim's details and get a
  Fraud/Genuine result, fraud probability, risk level, and the specific
  risk indicators that were triggered.
- **`/analytics`** — Fraud rate broken down by scheme, district, gender, age
  band, income band, and location risk, plus a confusion-matrix breakdown.
- **`/about`** — Problem statement, dataset, ML approach, technologies,
  objectives, and architecture.

## Project Screenshots

*(Add screenshots here after running the app locally)*

- `screenshots/dashboard.png`
- `screenshots/prediction-form.png`
- `screenshots/prediction-result.png`
- `screenshots/analytics.png`

## Expected Output (from an actual training run)

These figures come directly from `models/metrics.json`, generated by
running `train_model.py` on the shipped dataset — they are not fabricated.

| Metric      | Value   |
|-------------|---------|
| Accuracy    | 94.81%  |
| Precision   | 67.90%  |
| Recall      | 91.15%  |
| F1 Score    | 77.82%  |
| ROC-AUC     | 98.33%  |

Confusion matrix (test set, 20,000 records):

|                     | Predicted Genuine | Predicted Fraud |
|---------------------|-------------------|------------------|
| **Actual Genuine**  | 17,138 (TN)       | 862 (FP)         |
| **Actual Fraud**    | 177 (FN)          | 1,823 (TP)       |

The model favours **recall over precision** by design (`class_weight="balanced"`),
which is appropriate for fraud triage: it is far better to flag a genuine
claim for a second manual look (false positive) than to let an actual
fraudulent claim pass through unnoticed (false negative).

## Future Scope

- Integrate real (anonymized) disbursement data feeds via a secure pipeline.
- Add model explainability (e.g. SHAP values) directly on the prediction page.
- Support scheduled or on-demand model retraining from the dashboard.
- Extend to additional welfare schemes and other Indian states.
- Add authentication and role-based access for field verification officers.

## Conclusion

This project demonstrates that a well-engineered, interpretable classical
machine learning model — a Random Forest Classifier — can achieve strong
fraud-detection performance (ROC-AUC ≈ 0.98) while remaining transparent
and fast enough for real-time use in a government audit context, all built
with an accessible, dependency-light Python and Flask stack suitable for
academic demonstration.
