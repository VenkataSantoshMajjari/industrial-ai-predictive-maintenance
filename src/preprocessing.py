from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Project-relative dataset path
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "dataset" / "ai4i2020.csv"


NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

CATEGORICAL_FEATURES = ["Type"]

TARGET = "Machine failure"


def load_and_split_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
        + [TARGET]
    )

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing dataset columns: {missing_columns}"
        )

    if df[required_columns].isnull().any().any():
        raise ValueError(
            "Dataset contains missing values in required columns."
        )

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                NUMERIC_FEATURES
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES
            )
        ]
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    )


if __name__ == "__main__":

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    ) = load_and_split_data()

    print("\n========== PREPROCESSING RESULTS ==========")

    print("Training Features Shape:", X_train.shape)
    print("Testing Features Shape:", X_test.shape)

    print("\nTraining Target Distribution:")
    print(y_train.value_counts())

    print("\nTesting Target Distribution:")
    print(y_test.value_counts())

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print("\nProcessed Training Shape:", X_train_processed.shape)
    print("Processed Testing Shape:", X_test_processed.shape)

    print("\n========== PREPROCESSING COMPLETED ==========")