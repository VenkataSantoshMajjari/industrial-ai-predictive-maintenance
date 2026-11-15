from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Load best model
model = joblib.load(
    MODELS_DIR / "best_model.pkl"
)


# Extract pipeline components
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]


# Feature names
feature_names = preprocessor.get_feature_names_out()

# Feature importance
importance_scores = classifier.feature_importances_


importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_scores
})


importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)


print("\n========== FEATURE IMPORTANCE ==========")

print(importance_df.to_string(index=False))


# Save feature importance CSV
importance_df.to_csv(
    RESULTS_DIR / "feature_importance.csv",
    index=False
)


# Create feature importance visualization
plt.figure(figsize=(10, 6))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xlabel("Importance Score")
plt.ylabel("Features")

plt.title(
    "XGBoost - Machine Failure Feature Importance"
)

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "feature_importance.png",
    dpi=300
)

plt.close()


print("\nFeature importance CSV saved.")
print("Feature importance chart saved.")

print("\n========== EXPLAINABILITY COMPLETED ==========")