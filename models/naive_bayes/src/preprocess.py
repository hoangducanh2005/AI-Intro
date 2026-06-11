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
NAIVE_BAYES_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = Path(__file__).resolve().parents[3] / "dataset" / "diabetes.csv"


def prepare_diabetes_data(
    dataset_path=DEFAULT_DATASET_PATH,
    feature_columns=None,
    test_size=0.2,
    random_state=42,
):
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS

    df = pd.read_csv(dataset_path)
    
    # Xử lý các giá trị 0 không hợp lệ (Missing values) thành Median (Trung vị)
    invalid_zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in invalid_zero_cols:
        if col in df.columns:
            median_val = df.loc[df[col] != 0, col].median()
            df[col] = df[col].replace(0, median_val)

    X = df[feature_columns]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

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
