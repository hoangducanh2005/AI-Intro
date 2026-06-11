import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from models.k_nearest_neighbor.knn import KNeighborsClassifier as KNNClassifier
from models.k_nearest_neighbor.pso import BinaryPSOFeatureSelection

def tune_k_with_holdout(X_train_scaled, y_train, k_range=range(1, 30), test_size=0.2, random_state=42):
    # Set seed for reproducibility
    np.random.seed(random_state)
    
    # Split the scaled training set into sub-train and validation
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train_scaled,
        y_train,
        test_size=test_size,
        random_state=random_state,
    )
    
    k_list = list(k_range)
    scores = []
    
    for k in k_list:
        knn = KNNClassifier(k=k)
        knn.fit(X_train_sub, y_train_sub)
        y_pred_val = knn.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred_val)
        scores.append(accuracy)
        
    best_index = int(np.argmax(scores))
    return {
        "best_k": k_list[best_index],
        "k_values": k_list,
        "cv_scores": scores,  # Named cv_scores to align with interface
    }

def select_features_with_pso(
    X_train_scaled,
    y_train,
    feature_columns,
    best_k,
    num_particles=15,
    max_iter=30,
    random_state=42,
):
    # Set random seed to make the particle swarm optimization deterministic (seed=42)
    np.random.seed(random_state)
    
    # Split training set into sub-train and validation
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train_scaled,
        y_train,
        test_size=0.2,
        random_state=random_state,
    )
    
    knn_model = KNNClassifier(k=best_k)
    pso_selector = BinaryPSOFeatureSelection(
        num_features=X_train_scaled.shape[1],
        estimator=knn_model,
        num_particles=num_particles,
        max_iter=max_iter,
        verbose=False
    )
    
    # Run the PSO search
    best_feature_mask, best_val_accuracy = pso_selector.fit(
        X_train_sub, y_train_sub, X_val, y_val
    )
    
    selected_feature_indices = np.where(best_feature_mask == 1)[0]
    selected_features = [feature_columns[i] for i in selected_feature_indices]
    
    return {
        "selected_features": selected_features,
        "best_k": best_k,
        "best_cv_accuracy": best_val_accuracy,
        "candidate_features": feature_columns,
        "evaluated_subset_count": max_iter * num_particles,
        "top_subsets": [
            {
                "features": selected_features,
                "best_k": best_k,
                "best_cv_accuracy": best_val_accuracy
            }
        ]
    }

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

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, cv_scores, marker="o", color="deeppink")
    plt.title("Accuracy vs. Number of Neighbors (n)")
    plt.xlabel("Number of Neighbors (n)")
    plt.ylabel("Accuracy")
    plt.xticks(k_values)
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