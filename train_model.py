"""
ScamShield Pay - Model Training & Evaluation Pipeline
======================================================
Trains a Random Forest Classifier on synthetic transaction data designed to model
behavioral fraud indicators (transaction value, hour of day, cross-border routing,
device recognition, and authentication attempt anomalies).

NOTICE:
This dataset is synthetically generated for demonstration and educational purposes.
It simulates non-linear risk interactions without claiming to replicate real-world
proprietary banking distributions. Real-world payment fraud systems require continuous
retraining on verified merchant and chargeback datasets.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# 1. SETUP & REPRODUCIBILITY
# --------------------------------------------------

os.makedirs("models", exist_ok=True)
np.random.seed(42)

TOTAL_RECORDS = 10000

# --------------------------------------------------
# 2. SYNTHETIC FEATURE GENERATION
# --------------------------------------------------

# Feature 1: Transaction Amount (USD) - uniform distribution between $10 and $50,000
amount = np.random.uniform(10, 50000, TOTAL_RECORDS)

# Feature 2: Transaction Hour (0-23)
transaction_hour = np.random.randint(0, 24, TOTAL_RECORDS)

# Feature 3: International Transaction Flag (0 = Domestic, 1 = Cross-Border)
international = np.random.randint(0, 2, TOTAL_RECORDS)

# Feature 4: New Device Flag (0 = Recognized Device, 1 = New Device Fingerprint)
new_device = np.random.randint(0, 2, TOTAL_RECORDS)

# Feature 5: Prior Failed Attempts (0-5 attempts)
failed_attempts = np.random.randint(0, 6, TOTAL_RECORDS)

# --------------------------------------------------
# 3. PROBABILISTIC FRAUD LABEL GENERATION
# --------------------------------------------------
# Latent risk probability modeled from behavioral heuristics with stochastic variance
fraud_probability = (
    0.02
    + (amount > 20000) * 0.20
    + international * 0.15
    + new_device * 0.20
    + (failed_attempts >= 3) * 0.25
    + ((transaction_hour <= 4) | (transaction_hour >= 23)) * 0.10
)

# Clip to realistic probability interval [0, 0.95]
fraud_probability = np.clip(fraud_probability, 0.0, 0.95)

# Sample ground-truth binary label via binomial draw
is_fraud = np.random.binomial(1, fraud_probability)

# Assemble DataFrame
data = pd.DataFrame({
    "amount": amount,
    "transaction_hour": transaction_hour,
    "international": international,
    "new_device": new_device,
    "failed_attempts": failed_attempts,
    "is_fraud": is_fraud
})

# --------------------------------------------------
# 4. STRATIFIED TRAIN / TEST SPLIT (NO DATA LEAKAGE)
# --------------------------------------------------

X = data.drop("is_fraud", axis=1)
y = data["is_fraud"]

# Stratified split ensures exact class distribution is preserved in both splits
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# 5. MODEL ARCHITECTURE & TRAINING
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# --------------------------------------------------
# 6. RIGOROUS MODEL EVALUATION
# --------------------------------------------------

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, predictions)
prec = precision_score(y_test, predictions, zero_division=0)
rec = recall_score(y_test, predictions, zero_division=0)
f1 = f1_score(y_test, predictions, zero_division=0)
auc = roc_auc_score(y_test, probabilities)
cm = confusion_matrix(y_test, predictions)

tn, fp, fn, tp = cm.ravel()

print("=" * 60)
print(" SCAMSHIELD PAY - RANDOM FOREST MODEL EVALUATION REPORT")
print("=" * 60)
print(f"Total Dataset Records   : {TOTAL_RECORDS:,}")
print(f"Training Split Records  : {len(X_train):,} (80%)")
print(f"Test Split Records      : {len(X_test):,} (20%)")
print(f"Fraud Class Prevalence  : {y.mean() * 100:.2f}%\n")

print("--- OVERALL PERFORMANCE METRICS ---")
print(f"Accuracy               : {acc * 100:.2f}%")
print(f"Precision (Class 1)    : {prec * 100:.2f}%")
print(f"Recall (Class 1)       : {rec * 100:.2f}%")
print(f"F1-Score (Class 1)     : {f1 * 100:.2f}%")
print(f"ROC-AUC Score          : {auc * 100:.2f}%\n")

print("--- CONFUSION MATRIX ---")
print(f"True Negatives  (Legitimate Correct) : {tn:5d}")
print(f"False Positives (Legitimate Flagged) : {fp:5d}")
print(f"False Negatives (Fraud Missed)       : {fn:5d}")
print(f"True Positives  (Fraud Caught)       : {tp:5d}\n")

print("--- DETAILED CLASSIFICATION REPORT ---")
print(classification_report(y_test, predictions, target_names=["Legitimate (0)", "Fraud (1)"]))

print("--- FEATURE IMPORTANCE BREAKDOWN ---")
feature_importances = sorted(
    zip(X.columns, model.feature_importances_),
    key=lambda x: x[1],
    reverse=True
)
for feature_name, importance in feature_importances:
    print(f"  • {feature_name:<20}: {importance * 100:6.2f}%")

# --------------------------------------------------
# 7. MODEL SERIALIZATION (COMPRESSED)
# --------------------------------------------------

model_path = "models/scam_model.pkl"
joblib.dump(model, model_path, compress=3)

file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
print("\n" + "=" * 60)
print(f"Model saved successfully at: {model_path} ({file_size_mb:.2f} MB)")
print("=" * 60)
