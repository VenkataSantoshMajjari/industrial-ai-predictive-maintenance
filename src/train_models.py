from pathlib import Path
import joblib
import pandas as pd

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier

from preprocessing import load_and_split_data


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Load dataset
X_train, X_test, y_train, y_test, preprocessor = (
    load_and_split_data()
)


# Class imbalance calculation
negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

if positive_count == 0:
    raise ValueError("Training data contains no positive failure samples.")

scale_pos_weight = negative_count / positive_count


# Define machine learning models
models = {

    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        max_depth=8,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        max_depth=12,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
}


results = []
trained_models = {}


print("\n========== MODEL TRAINING STARTED ==========")


for name, classifier in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred, zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    pr_auc = average_precision_score(
        y_test, y_prob
    )

    print("\nAccuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("PR-AUC:", round(pr_auc, 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        zero_division=0
    ))

    file_name = name.lower().replace(" ", "_") + ".pkl"

    joblib.dump(
        pipeline,
        MODELS_DIR / file_name
    )

    trained_models[name] = pipeline

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "PR-AUC": pr_auc
    })


# Model comparison
comparison = pd.DataFrame(results)

comparison = comparison.sort_values(
    by="PR-AUC",
    ascending=False
)

comparison.to_csv(
    RESULTS_DIR / "model_comparison.csv",
    index=False
)


# Select best model
best_model_name = comparison.iloc[0]["Model"]
best_model = trained_models[best_model_name]

joblib.dump(
    best_model,
    MODELS_DIR / "best_model.pkl"
)


print("\n========== MODEL COMPARISON ==========")

print(comparison.to_string(index=False))

print("\nBest Model:", best_model_name)

print("Best model saved to models/best_model.pkl")

print("\nAll trained models saved successfully!")

print("\n========== MODEL TRAINING COMPLETED ==========")