import json
import pickle
from datetime import datetime
from pathlib import Path

from models.naive_bayes.src.evaluate import (
    evaluate_classification_details,
    evaluate_predictions,
    save_decision_boundary_plot,
    save_confusion_matrix_plot,
)
from models.naive_bayes.src.naive_bayes import NaiveBayesClassifier
from models.naive_bayes.src.preprocess import (
    DEFAULT_DATASET_PATH,
    FEATURE_COLUMNS,
    NAIVE_BAYES_ROOT,
    prepare_diabetes_data,
)

DEFAULT_MODEL_DIR = NAIVE_BAYES_ROOT / "artifacts"
DEFAULT_LOG_DIR = NAIVE_BAYES_ROOT / "logs"


def train_naive_bayes_model(
    feature_columns=None,
    test_size=0.2,
    random_state=42,
    dataset_path=DEFAULT_DATASET_PATH,
    model_dir=DEFAULT_MODEL_DIR,
    log_dir=DEFAULT_LOG_DIR,
):
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS

    # Gọi preprocessor dùng chung (chia train/test đồng bộ với KNN)
    data = prepare_diabetes_data(
        dataset_path=dataset_path,
        feature_columns=feature_columns,
        test_size=test_size,
        random_state=random_state,
    )

    # Khởi tạo và huấn luyện mô hình Naive Bayes từ đầu
    model = NaiveBayesClassifier()
    model.fit(data["X_train"], data["y_train"])

    # Dự đoán trên tập kiểm thử
    y_pred = model.predict(data["X_test"])

    # Tính toán các chỉ số đánh giá hiệu năng
    test_metrics = evaluate_predictions(data["y_test"], y_pred)
    classification_details = evaluate_classification_details(
        data["y_test"],
        y_pred,
        target_names=["Class 0", "Class 1"],
    )

    # Đảm bảo thư mục lưu trữ tồn tại
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Lưu Confusion Matrix
    confusion_matrix_plot_path = save_confusion_matrix_plot(
        data["y_test"],
        y_pred,
        model_dir / "confusion_matrix.png",
        display_labels=["Class 0", "Class 1"],
        title="Naive Bayes Confusion Matrix",
    )

    # Vẽ Decision Boundary cho 2 đặc trưng quan trọng đầu tiên
    boundary_data = prepare_diabetes_data(
        dataset_path=dataset_path,
        feature_columns=feature_columns[:2],
        test_size=test_size,
        random_state=random_state,
    )
    boundary_model = NaiveBayesClassifier()
    boundary_model.fit(boundary_data["X_train"], boundary_data["y_train"])
    
    decision_boundary_plot_path = save_decision_boundary_plot(
        boundary_model,
        boundary_data["X_train"],
        boundary_data["y_train"],
        model_dir / "decision_boundary.png",
        feature_names=boundary_data["feature_columns"],
        title="Naive Bayes Decision Boundary",
    )

    # Lưu mô hình và scaler dưới dạng pickle
    model_path = save_trained_model(model, data["scaler"], data["feature_columns"], model_dir)

    # Lưu log huấn luyện dạng JSONL
    log_path = log_training_result(
        log_dir=log_dir,
        params={
            "test_size": test_size,
            "random_state": random_state,
            "dataset_path": str(Path(dataset_path).resolve()),
            "feature_columns": feature_columns,
        },
        test_metrics=test_metrics,
        classification_details=classification_details,
        model_path=model_path,
        confusion_matrix_plot_path=confusion_matrix_plot_path,
        decision_boundary_plot_path=decision_boundary_plot_path,
    )

    return {
        "model": model,
        "scaler": data["scaler"],
        "metrics": test_metrics,
        "test_metrics": test_metrics,
        "classification_details": classification_details,
        "feature_columns": data["feature_columns"],
        "class_labels": model.classes_,
        "test_predictions": y_pred,
        "test_labels": data["y_test"],
        "confusion_matrix_plot_path": confusion_matrix_plot_path,
        "decision_boundary_plot_path": decision_boundary_plot_path,
        "model_path": model_path,
        "log_path": log_path,
    }


def save_trained_model(model, scaler, feature_columns, model_dir):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "nb_model.pkl"

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
    classification_details,
    model_path,
    confusion_matrix_plot_path=None,
    decision_boundary_plot_path=None,
):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "training_results.jsonl"

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "params": params,
        "test_metrics": test_metrics,
        "classification_details": classification_details,
        "confusion_matrix_plot_path": str(confusion_matrix_plot_path.resolve()) if confusion_matrix_plot_path else None,
        "decision_boundary_plot_path": str(decision_boundary_plot_path.resolve()) if decision_boundary_plot_path else None,
        "model_path": str(model_path.resolve()),
    }
    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")

    return log_path
