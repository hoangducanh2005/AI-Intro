import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from models.naive_bayes.src.train import train_naive_bayes_model


def main():
    result = train_naive_bayes_model()

    print("Naive Bayes diabetes prediction results")
    print("---------------------------------------")
    print(f"Features used: {', '.join(result['feature_columns'])}")

    print("\nTest metrics")
    for metric, value in result["test_metrics"].items():
        print(f"{metric}: {value:.4f}")

    print("\nConfusion matrix")
    for row in result["classification_details"]["confusion_matrix"]:
        print(row)

    print("\nClassification Report:")
    print(result["classification_details"]["classification_report_text"])

    print(f"\nSaved model: {result['model_path']}")
    print(f"Confusion matrix plot: {result['confusion_matrix_plot_path']}")
    print(f"Decision boundary plot: {result['decision_boundary_plot_path']}")
    print(f"Training log: {result['log_path']}")


if __name__ == "__main__":
    main()
