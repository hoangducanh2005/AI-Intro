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
PROJECT_ROOT = Path(__file__).resolve().parents[3]
KNN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "dataset" / "diabetes.csv"


def load_diabetes_data(dataset_path=DEFAULT_DATASET_PATH, feature_columns=None):
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS

    df = pd.read_csv(dataset_path)
    missing_columns = set(feature_columns + [TARGET_COLUMN]) - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")

    X = df[feature_columns]
    y = df[TARGET_COLUMN]
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    if test_size <= 0 or test_size >= 1:
        raise ValueError("test_size must be greater than 0 and less than 1")

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def scale_data_splits(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def prepare_diabetes_data(
    dataset_path=DEFAULT_DATASET_PATH,
    feature_columns=None,
    test_size=0.2,
    random_state=42,
):
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS

    X, y = load_diabetes_data(dataset_path, feature_columns=feature_columns)
    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    X_train_scaled, X_test_scaled, scaler = scale_data_splits(
        X_train,
        X_test,
    )

    return {
        "X_train_raw": X_train,
        "X_test_raw": X_test,
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train.to_numpy(),
        "y_test": y_test.to_numpy(),
        "scaler": scaler,
        "feature_columns": feature_columns,
    }
