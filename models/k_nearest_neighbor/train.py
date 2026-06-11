import json
import pickle
from datetime import datetime
from pathlib import Path
import pandas as pd

from models.k_nearest_neighbor.knn import KNeighborsClassifier as KNNClassifier
from models.k_nearest_neighbor.preprocess import (
    DEFAULT_DATASET_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    KNN_ROOT,
    prepare_diabetes_data,
)
from models.k_nearest_neighbor.tune import (
    tune_k_with_holdout,
    select_features_with_pso,
    save_feature_selection_result,
    save_cv_plot,
    save_correlation_heatmap,
    get_feature_target_correlations,
)
from models.k_nearest_neighbor.evaluate import (
    evaluate_predictions,
    evaluate_classification_details,
    save_confusion_matrix_plot,
    save_decision_boundary_plot,
)

DEFAULT_MODEL_DIR = KNN_ROOT / "artifacts"
DEFAULT_LOG_DIR = KNN_ROOT / "logs"

def load_cached_results(model_dir, log_dir, params):
    model_dir = Path(model_dir)
    log_dir = Path(log_dir)
    model_path = model_dir / "knn_model.pkl"
    selected_features_path = model_dir / "selected_features.json"
    cv_plot_path = model_dir / "knn_cv_accuracy.png"
    correlation_plot_path = model_dir / "feature_correlation_heatmap.png"
    confusion_matrix_plot_path = model_dir / "confusion_matrix.png"
    decision_boundary_plot_path = model_dir / "decision_boundary.png"
    log_path = log_dir / "training_results.jsonl"
    
    required_files = [
        model_path, selected_features_path, cv_plot_path,
        correlation_plot_path, confusion_matrix_plot_path,
        decision_boundary_plot_path, log_path
    ]
    if not all(f.exists() for f in required_files):
        return None
        
    try:
        last_record = None
        with log_path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    last_record = json.loads(line)
        if not last_record:
            return None
            
        log_params = last_record.get("params", {})
        for key in ["tune_k", "test_size", "random_state"]:
            if log_params.get(key) != params.get(key):
                return None
        if params.get("k") is not None and log_params.get("k") != params.get("k"):
            return None
            
        with model_path.open("rb") as file:
            payload = pickle.load(file)
            
        with selected_features_path.open("r", encoding="utf-8") as file:
            feature_selection_result = json.load(file)
            
        return {
            "model": payload["model"],
            "scaler": payload["scaler"],
            "metrics": last_record["test_metrics"],
            "test_metrics": last_record["test_metrics"],
            "baseline_test_metrics": last_record["baseline_test_metrics"],
            "metric_deltas": last_record["metric_deltas"],
            "classification_details": last_record["classification_details"],
            "feature_columns": payload["feature_columns"],
            "all_feature_columns": FEATURE_COLUMNS,
            "class_labels": payload["class_labels"],
            "test_predictions": [],
            "test_labels": [],
            "k": last_record["params"]["k"],
            "baseline_k": last_record["baseline_tuning_result"]["best_k"] if last_record.get("baseline_tuning_result") else last_record["params"]["k"],
            "tuning_result": last_record.get("baseline_tuning_result"),
            "feature_selection_result": feature_selection_result,
            "feature_target_correlations": feature_selection_result.get("feature_target_correlations", {}),
            "selected_features_path": selected_features_path,
            "correlation_plot_path": correlation_plot_path,
            "confusion_matrix_plot_path": confusion_matrix_plot_path,
            "decision_boundary_plot_path": decision_boundary_plot_path,
            "cv_plot_path": cv_plot_path,
            "model_path": model_path,
            "log_path": log_path,
        }
    except Exception:
        return None

def train_knn_model(
    k=None,
    tune_k=True,
    k_range=range(1, 30),
    cv=5,  # Parameter kept in signature for compatibility
    verbose=False,
    select_features=True,
    max_feature_candidates=8,
    max_feature_subset_size=4,
    test_size=0.3,
    random_state=666,
    dataset_path=DEFAULT_DATASET_PATH,
    model_dir=DEFAULT_MODEL_DIR,
    log_dir=DEFAULT_LOG_DIR,
):
    # Try loading cached results first
    current_params = {
        "k": k,
        "tune_k": tune_k,
        "test_size": test_size,
        "random_state": random_state,
    }
    cached = load_cached_results(model_dir, log_dir, current_params)
    if cached is not None:
        if verbose:
            print("Loaded results from pre-trained cache.")
        return cached

    # Prepare all features data
    all_feature_data = prepare_diabetes_data(
        dataset_path=dataset_path,
        feature_columns=FEATURE_COLUMNS,
        test_size=test_size,
        random_state=random_state,
    )

    baseline_tuning_result = None
    feature_selection_result = None
    selected_features_path = None
    correlation_plot_path = None
    feature_target_correlations = None
    cv_plot_path = None
    selected_k = k

    # Compute correlation and save correlation plot
    dataset_df = pd.read_csv(dataset_path)
    correlation_plot_path = save_correlation_heatmap(
        dataset_df[FEATURE_COLUMNS + [TARGET_COLUMN]],
        Path(model_dir) / "feature_correlation_heatmap.png",
    )
    feature_target_correlations = get_feature_target_correlations(
        dataset_df[FEATURE_COLUMNS + [TARGET_COLUMN]],
        TARGET_COLUMN,
    )

    # 1. Hyperparameter Tuning using Holdout Validation on scaled X_train
    if tune_k:
        baseline_tuning_result = tune_k_with_holdout(
            all_feature_data["X_train"],
            all_feature_data["y_train"],
            k_range=k_range,
            test_size=0.2,
            random_state=42,  # Split random state for tuning
        )
        cv_plot_path = save_cv_plot(
            baseline_tuning_result["k_values"],
            baseline_tuning_result["cv_scores"],
            Path(model_dir) / "knn_cv_accuracy.png",
        )
        if verbose:
            print(f"Best k from tuning: {baseline_tuning_result['best_k']}")
        selected_k = baseline_tuning_result["best_k"]

    if selected_k is None:
        selected_k = 5

    # Evaluate Baseline Model (All Features)
    baseline_model, baseline_test_metrics, baseline_pred = train_and_evaluate(
        all_feature_data,
        selected_k,
    )

    # 2. PSO Feature Selection (load from cache if available to skip slow search)
    selected_features = FEATURE_COLUMNS
    if select_features:
        selected_features_path = Path(model_dir) / "selected_features.json"
        if selected_features_path.exists():
            try:
                with selected_features_path.open("r", encoding="utf-8") as file:
                    feature_selection_result = json.load(file)
                selected_features = feature_selection_result["selected_features"]
                if verbose:
                    print(f"Loaded pre-computed PSO selected features: {selected_features}")
            except Exception:
                feature_selection_result = None

        if feature_selection_result is None:
            feature_selection_result = select_features_with_pso(
                all_feature_data["X_train"],
                all_feature_data["y_train"],
                feature_columns=FEATURE_COLUMNS,
                best_k=selected_k,
                num_particles=15,
                max_iter=30,
                random_state=42,
            )
            selected_features = feature_selection_result["selected_features"]
            selected_features_path = save_feature_selection_result(
                {
                    **feature_selection_result,
                    "feature_target_correlations": feature_target_correlations,
                },
                Path(model_dir) / "selected_features.json",
            )

    # Prepare selected features dataset
    selected_data = prepare_diabetes_data(
        dataset_path=dataset_path,
        feature_columns=selected_features,
        test_size=test_size,
        random_state=random_state,
    )
    
    # Train and Evaluate Final Model (Selected Features)
    model, test_metrics, y_pred = train_and_evaluate(selected_data, selected_k)
    classification_details = evaluate_classification_details(
        selected_data["y_test"],
        y_pred,
        target_names=["Class 0", "Class 1"],
    )
    confusion_matrix_plot_path = save_confusion_matrix_plot(
        selected_data["y_test"],
        y_pred,
        Path(model_dir) / "confusion_matrix.png",
        display_labels=["Class 0", "Class 1"],
        title=f"Confusion Matrix (k={selected_k})",
    )
    
    # Decision Boundary using first 2 selected features
    boundary_data = prepare_diabetes_data(
        dataset_path=dataset_path,
        feature_columns=selected_features[:2],
        test_size=test_size,
        random_state=random_state,
    )
    boundary_model = KNNClassifier(k=selected_k)
    boundary_model.fit(boundary_data["X_train"], boundary_data["y_train"])
    decision_boundary_plot_path = save_decision_boundary_plot(
        boundary_model,
        boundary_data["X_train"],
        boundary_data["y_train"],
        Path(model_dir) / "decision_boundary.png",
        feature_names=boundary_data["feature_columns"],
        k=selected_k,
    )

    # Save model and logger details
    model_path = save_trained_model(model, selected_data["scaler"], selected_data["feature_columns"], model_dir)
    log_path = log_training_result(
        log_dir=log_dir,
        params={
            "k": selected_k,
            "tune_k": tune_k,
            "k_range": list(k_range),
            "select_features": select_features,
            "test_size": test_size,
            "random_state": random_state,
            "dataset_path": str(Path(dataset_path).resolve()),
        },
        test_metrics=test_metrics,
        baseline_test_metrics=baseline_test_metrics,
        metric_deltas=calculate_metric_deltas(test_metrics, baseline_test_metrics),
        classification_details=classification_details,
        model_path=model_path,
        baseline_tuning_result=baseline_tuning_result,
        feature_selection_result=feature_selection_result,
        cv_plot_path=cv_plot_path,
        correlation_plot_path=correlation_plot_path,
        confusion_matrix_plot_path=confusion_matrix_plot_path,
        decision_boundary_plot_path=decision_boundary_plot_path,
        selected_features_path=selected_features_path,
    )

    return {
        "model": model,
        "scaler": selected_data["scaler"],
        "metrics": test_metrics,
        "test_metrics": test_metrics,
        "baseline_test_metrics": baseline_test_metrics,
        "metric_deltas": calculate_metric_deltas(test_metrics, baseline_test_metrics),
        "classification_details": classification_details,
        "feature_columns": selected_data["feature_columns"],
        "all_feature_columns": FEATURE_COLUMNS,
        "class_labels": model.classes_,
        "test_predictions": y_pred,
        "test_labels": selected_data["y_test"],
        "k": selected_k,
        "baseline_k": baseline_tuning_result["best_k"] if baseline_tuning_result else selected_k,
        "tuning_result": baseline_tuning_result,
        "feature_selection_result": feature_selection_result,
        "feature_target_correlations": feature_target_correlations,
        "selected_features_path": selected_features_path,
        "correlation_plot_path": correlation_plot_path,
        "confusion_matrix_plot_path": confusion_matrix_plot_path,
        "decision_boundary_plot_path": decision_boundary_plot_path,
        "cv_plot_path": cv_plot_path,
        "model_path": model_path,
        "log_path": log_path,
    }

def train_and_evaluate(data, k):
    model = KNNClassifier(k=k)
    model.fit(data["X_train"], data["y_train"])
    y_pred = model.predict(data["X_test"])
    test_metrics = evaluate_predictions(data["y_test"], y_pred)
    return model, test_metrics, y_pred

def calculate_metric_deltas(test_metrics, baseline_test_metrics):
    return {
        metric: test_metrics[metric] - baseline_test_metrics[metric]
        for metric in test_metrics
    }

def save_trained_model(model, scaler, feature_columns, model_dir):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "knn_model.pkl"

    payload = {
        "model": model,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "class_labels": model.classes_,
    }
    with model_path.open("wb") as file:
        pickle.dump(payload, file)

    return model_path

def log_training_result(
    log_dir,
    params,
    test_metrics,
    baseline_test_metrics,
    metric_deltas,
    classification_details,
    model_path,
    baseline_tuning_result=None,
    feature_selection_result=None,
    cv_plot_path=None,
    correlation_plot_path=None,
    confusion_matrix_plot_path=None,
    decision_boundary_plot_path=None,
    selected_features_path=None,
):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "training_results.jsonl"

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "params": params,
        "test_metrics": test_metrics,
        "baseline_test_metrics": baseline_test_metrics,
        "metric_deltas": metric_deltas,
        "classification_details": classification_details,
        "baseline_tuning_result": baseline_tuning_result,
        "feature_selection_result": feature_selection_result,
        "cv_plot_path": str(cv_plot_path.resolve()) if cv_plot_path else None,
        "correlation_plot_path": str(correlation_plot_path.resolve()) if correlation_plot_path else None,
        "confusion_matrix_plot_path": str(confusion_matrix_plot_path.resolve())
        if confusion_matrix_plot_path
        else None,
        "decision_boundary_plot_path": str(decision_boundary_plot_path.resolve())
        if decision_boundary_plot_path
        else None,
        "selected_features_path": str(selected_features_path.resolve()) if selected_features_path else None,
        "model_path": str(model_path.resolve()),
    }
    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")

    return log_path
