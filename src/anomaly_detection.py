from pathlib import Path
import joblib

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from preprocessing import load_and_split_data


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# Load dataset
X_train, X_test, y_train, y_test, preprocessor = (
    load_and_split_data()
)


print("\n========== ANOMALY DETECTION ==========")


# Normal training records
X_train_normal = X_train[y_train == 0]

print("\nNormal training records:", len(X_train_normal))


# Isolation Forest pipeline
anomaly_model = Pipeline([
    ("preprocessor", clone(preprocessor)),

    (
        "isolation_forest",
        IsolationForest(
            n_estimators=200,
            contamination=0.03,
            random_state=42,
            n_jobs=-1
        )
    )
])


# Train anomaly detector
print("\nTraining Isolation Forest...")

anomaly_model.fit(X_train_normal)


# Predict anomalies
raw_predictions = anomaly_model.predict(X_test)

# Convert -1 to anomaly class 1
anomaly_predictions = (
    raw_predictions == -1
).astype(int)


# Evaluation
precision = precision_score(
    y_test,
    anomaly_predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    anomaly_predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    anomaly_predictions,
    zero_division=0
)


print("\nAnomaly Detection Results")

print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, anomaly_predictions))


# Save model
joblib.dump(
    anomaly_model,
    MODELS_DIR / "isolation_forest.pkl"
)


print("\nModel saved: models/isolation_forest.pkl")

print("\n========== ANOMALY DETECTION COMPLETED ==========")