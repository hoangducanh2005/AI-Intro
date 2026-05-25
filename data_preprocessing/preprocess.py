from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET_COLUMN = "Outcome"
DEFAULT_DATASET_PATH = Path(__file__).resolve().parents[1] / "dataset" / "diabetes.csv"


def load_diabetes_data(dataset_path=DEFAULT_DATASET_PATH):
    df = pd.read_csv(dataset_path)
    missing_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def scale_train_test(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def prepare_diabetes_data(dataset_path=DEFAULT_DATASET_PATH, test_size=0.2, random_state=42):
    X, y = load_diabetes_data(dataset_path)
    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    X_train_scaled, X_test_scaled, scaler = scale_train_test(X_train, X_test)

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train.to_numpy(),
        "y_test": y_test.to_numpy(),
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
    }
