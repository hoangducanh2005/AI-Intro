import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from models.k_nearest_neighbor.train import train_knn_model

def main():
    result = train_knn_model(tune_k=True, k_range=range(1, 30), verbose=True)

    print("KNN diabetes prediction results (k_nearest_neighbor)")
    print("-----------------------------------------------------")
    print(f"Selected-feature best k: {result['k']}")
    print(f"Selected features: {', '.join(result['feature_columns'])}")

    print("\nBaseline test metrics (all features)")
    for metric, value in result["baseline_test_metrics"].items():
        print(f"{metric}: {value:.4f}")

    print("\nSelected-feature test metrics")
    for metric, value in result["test_metrics"].items():
        print(f"{metric}: {value:.4f}")

    print("\nMetric deltas vs baseline")
    for metric, value in result["metric_deltas"].items():
        print(f"{metric}: {value:+.4f}")

    print("\nConfusion matrix")
    for row in result["classification_details"]["confusion_matrix"]:
        print(row)

    print("\nClassification Report:")
    print(result["classification_details"]["classification_report_text"])

    print(f"\nSaved model: {result['model_path']}")
    print(f"CV plot: {result['cv_plot_path']}")
    print(f"Correlation plot: {result['correlation_plot_path']}")
    print(f"Confusion matrix plot: {result['confusion_matrix_plot_path']}")
    print(f"Decision boundary plot: {result['decision_boundary_plot_path']}")
    print(f"Selected features: {result['selected_features_path']}")
    print(f"Training log: {result['log_path']}")

if __name__ == "__main__":
    main()
