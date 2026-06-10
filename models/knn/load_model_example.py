import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
KNN_ROOT = Path(__file__).resolve().parent
MODEL_PATH = KNN_ROOT / "artifacts" / "knn_model.pkl"
DEFAULT_DATA_PATH = PROJECT_ROOT / "dataset" / "diabetes.csv"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_saved_model(model_path=MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Run `python models\\knn\\main.py` first."
        )

    with model_path.open("rb") as file:
        return pickle.load(file)


def load_example_from_csv(data_path, row_index):
    row_index -= 2
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    if row_index < 0 or row_index >= len(df):
        raise IndexError(f"row_index must be between 0 and {len(df) - 1}, got {row_index}")

    return df.iloc[row_index].to_dict()


def predict_example(payload, example):
    model = payload["model"]
    scaler = payload["scaler"]
    feature_columns = payload["feature_columns"]
    class_labels = list(payload["class_labels"])

    input_df = pd.DataFrame([example])
    input_df = input_df[feature_columns]
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    probability_by_class = {
        str(class_label): float(probability)
        for class_label, probability in zip(class_labels, probabilities)
    }

    return prediction, probability_by_class, feature_columns


def main():
    args = parse_args()
    payload = load_saved_model()
    example = load_example_from_csv(args.data_path, args.row)
    prediction, probability_by_class, feature_columns = predict_example(payload, example)
    actual = example.get("Outcome")

    print("Loaded KNN model example")
    print("------------------------")
    print(f"Model path: {MODEL_PATH}")
    print(f"Data path: {Path(args.data_path).resolve()}")
    print(f"Row index: {args.row}")
    print(f"Used features: {', '.join(feature_columns)}")
    print(f"Example input: {example}")
    if actual is not None:
        print(f"Actual label: {actual} ({'Positive' if actual == 1 else 'Negative'})")
    print(f"Prediction: {prediction} ({'Positive' if prediction == 1 else 'Negative'})")
    print(f"Class probabilities: {probability_by_class}")


def parse_args():
    parser = argparse.ArgumentParser(description="Load a saved KNN model and predict one CSV row.")
    parser.add_argument(
        "--data-path",
        default=DEFAULT_DATA_PATH,
        help="Path to the CSV file. Default: dataset/diabetes.csv",
    )
    parser.add_argument(
        "--row",
        type=int,
        default=0,
        help="Zero-based row index to predict. Default: 0",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
