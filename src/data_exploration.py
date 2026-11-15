from pathlib import Path
import pandas as pd


# Dataset path
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "dataset" / "ai4i2020.csv"


# Check dataset
if not DATA_PATH.exists():

    raise FileNotFoundError(
        "Dataset not found! "
        "Place ai4i2020.csv inside the dataset folder."
    )


# Load dataset
df = pd.read_csv(DATA_PATH)


print("\n========== DATASET OVERVIEW ==========")

print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Description:")
print(df.describe())

print("\nMachine Failure Distribution:")
print(df["Machine failure"].value_counts())

print("\nMachine Failure Percentage:")
print(
    df["Machine failure"].value_counts(normalize=True) * 100
)

print("\n========== EXPLORATION COMPLETED ==========")