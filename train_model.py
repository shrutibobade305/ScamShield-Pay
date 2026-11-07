import os
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# Create models directory
os.makedirs("models", exist_ok=True)

# Generate sample transaction data
np.random.seed(42)

TOTAL_RECORDS = 10000

amount = np.random.uniform(10, 50000, TOTAL_RECORDS)
transaction_hour = np.random.randint(0, 24, TOTAL_RECORDS)
international = np.random.randint(0, 2, TOTAL_RECORDS)
new_device = np.random.randint(0, 2, TOTAL_RECORDS)
failed_attempts = np.random.randint(0, 6, TOTAL_RECORDS)

# Generate synthetic fraud labels for demonstration
fraud_probability = (
    0.02
    + (amount > 20000) * 0.20
    + international * 0.15
    + new_device * 0.20
    + (failed_attempts >= 3) * 0.25
    + ((transaction_hour <= 4) | (transaction_hour >= 23)) * 0.10
)

fraud_probability = np.clip(fraud_probability, 0, 0.95)

is_fraud = np.random.binomial(1, fraud_probability)

# Create DataFrame
data = pd.DataFrame({
    "amount": amount,
    "transaction_hour": transaction_hour,
    "international": international,
    "new_device": new_device,
    "failed_attempts": failed_attempts,
    "is_fraud": is_fraud
})

# Features and target
X = data.drop("is_fraud", axis=1)
y = data["is_fraud"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model Training Completed!")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Save model
model_path = "models/scam_model.pkl"

joblib.dump(model, model_path)

print(f"\nModel saved successfully at: {model_path}")