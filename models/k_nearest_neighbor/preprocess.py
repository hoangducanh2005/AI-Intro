import pandas as pd
from pathlib import Path
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

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
KNN_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = PROJECT_ROOT / "dataset" / "diabetes.csv"

def load_diabetes_data(dataset_path=DEFAULT_DATASET_PATH, feature_columns=None):
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS

    df = pd.read_csv(dataset_path)
    missing_columns = set(feature_columns + [TARGET_COLUMN]) - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")

    # Remove outliers using the IQR method (factor 1.5)
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    outlier_threshold_factor = 1.5
    outliers = ((df < (Q1 - outlier_threshold_factor * IQR)) | (df > (Q3 + outlier_threshold_factor * IQR)))
    df = df[~outliers.any(axis=1)]

    X = df[feature_columns]
    y = df[TARGET_COLUMN]
    return X, y

def split_data(X, y, test_size=0.3, random_state=666):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

def scale_data_splits(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def prepare_diabetes_data(
    dataset_path=DEFAULT_DATASET_PATH,
    feature_columns=None,
    test_size=0.3,
    random_state=666,
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
