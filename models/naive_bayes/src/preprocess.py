from pathlib import Path

# Nhập các cấu hình và hàm chuẩn bị dữ liệu từ KNN để đảm bảo đồng bộ hoàn toàn
from models.knn.src.preprocess import (
    DEFAULT_DATASET_PATH,
    FEATURE_COLUMNS,
    PROJECT_ROOT,
    TARGET_COLUMN,
    prepare_diabetes_data,
)

NAIVE_BAYES_ROOT = Path(__file__).resolve().parents[1]
