import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from models.knn.src.knn import KNNClassifier


def tune_k_with_cross_validation(X, y, k_range, cv=5, random_state=42):
    X_array = np.asarray(X, dtype=float)
    y_array = np.asarray(y)
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    cv_scores = []
    for k in k_range:
        fold_scores = []
        for train_indices, valid_indices in splitter.split(X_array, y_array):
            X_train_fold = X_array[train_indices]
            X_valid_fold = X_array[valid_indices]
            y_train_fold = y_array[train_indices]
            y_valid_fold = y_array[valid_indices]

            scaler = StandardScaler()
            X_train_fold_scaled = scaler.fit_transform(X_train_fold)
            X_valid_fold_scaled = scaler.transform(X_valid_fold)

            model = KNNClassifier(k=k)
            model.fit(X_train_fold_scaled, y_train_fold)
            y_pred = model.predict(X_valid_fold_scaled)
            fold_scores.append(accuracy_score(y_valid_fold, y_pred))

        cv_scores.append(float(np.mean(fold_scores)))

    best_index = int(np.argmax(cv_scores))
    return {
        "best_k": list(k_range)[best_index],
        "k_values": list(k_range),
        "cv_scores": cv_scores,
    }


def select_features_with_cross_validation(
    X,
    y,
    feature_columns,
    k_range,
    cv=5,
    random_state=42,
    max_candidate_features=8,
    max_subset_size=None,
):
    candidate_features = select_candidate_features_by_target_correlation(
        X,
        y,
        feature_columns,
        max_candidate_features,
    )
    if max_subset_size is None:
        max_subset_size = len(candidate_features)
    max_subset_size = min(max_subset_size, len(candidate_features))

    best_result = None
    evaluated_subsets = []
    for subset_size in range(1, max_subset_size + 1):
        for subset in itertools.combinations(candidate_features, subset_size):
            subset = list(subset)
            tuning_result = tune_k_with_cross_validation(
                X[subset],
                y,
                k_range=k_range,
                cv=cv,
                random_state=random_state,
            )
            best_score = max(tuning_result["cv_scores"])
            result = {
                "features": subset,
                "best_k": tuning_result["best_k"],
                "best_cv_accuracy": best_score,
                "cv_scores": tuning_result["cv_scores"],
                "k_values": tuning_result["k_values"],
            }
            evaluated_subsets.append(result)

            if best_result is None or best_score > best_result["best_cv_accuracy"]:
                best_result = result

    return {
        "selected_features": best_result["features"],
        "best_k": best_result["best_k"],
        "best_cv_accuracy": best_result["best_cv_accuracy"],
        "candidate_features": candidate_features,
        "evaluated_subset_count": len(evaluated_subsets),
        "top_subsets": sorted(
            evaluated_subsets,
            key=lambda item: item["best_cv_accuracy"],
            reverse=True,
        )[:10],
    }


def select_candidate_features_by_target_correlation(X, y, feature_columns, max_candidate_features):
    X_df = pd.DataFrame(X, columns=feature_columns)
    target = pd.Series(y, name="target")
    correlations = X_df.join(target).corr(numeric_only=True)["target"].drop(labels=["target"])
    ranked_features = correlations.abs().sort_values(ascending=False).index.tolist()
    return ranked_features[:max_candidate_features]


def save_feature_selection_result(selection_result, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(selection_result, file, indent=2)

    return output_path


def save_cv_plot(k_values, cv_scores, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, cv_scores, marker="o")
    plt.title("k-NN Cross-Validation Accuracy vs k")
    plt.xlabel("Number of Neighbors: k")
    plt.ylabel("Cross-Validated Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path


def save_correlation_heatmap(df, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    correlation = df.corr(numeric_only=True)
    fig_width = max(8, len(correlation.columns) * 0.8)
    fig_height = max(6, len(correlation.columns) * 0.7)

    plt.figure(figsize=(fig_width, fig_height))
    image = plt.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(image, fraction=0.046, pad=0.04)
    plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=45, ha="right")
    plt.yticks(range(len(correlation.columns)), correlation.columns)
    plt.title("Feature Correlation Heatmap")

    for row_index in range(len(correlation.columns)):
        for col_index in range(len(correlation.columns)):
            value = correlation.iloc[row_index, col_index]
            plt.text(col_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path


def get_feature_target_correlations(df, target_column):
    correlation = df.corr(numeric_only=True)
    if target_column not in correlation.columns:
        return {}

    target_correlation = correlation[target_column].drop(labels=[target_column])
    return {
        feature: float(value)
        for feature, value in target_correlation.sort_values(key=lambda series: series.abs(), ascending=False).items()
    }
