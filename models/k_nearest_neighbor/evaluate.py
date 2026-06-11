from pathlib import Path
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

def evaluate_predictions(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
    }

def evaluate_classification_details(y_true, y_pred, target_names=None):
    if target_names is None:
        target_names = ["Class 0", "Class 1"]

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        zero_division=0,
        output_dict=True,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        zero_division=0,
    )

    return {
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "classification_report_text": report_text,
    }

def save_confusion_matrix_plot(y_true, y_pred, output_path, display_labels=None, title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if display_labels is None:
        display_labels = ["Class 0", "Class 1"]
    if title is None:
        title = "Confusion Matrix"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    
    # Thread-safe object-oriented plotting with identical size to Naive Bayes
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(cmap="Blues", ax=ax)
    ax.set_title(title)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path

def save_decision_boundary_plot(
    model,
    X_scaled,
    y,
    output_path,
    feature_names,
    k,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if X_scaled.shape[1] != 2:
        raise ValueError("Decision boundary plot requires exactly 2 scaled features")

    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Thread-safe object-oriented plotting with identical size to Naive Bayes
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
    
    try:
        import seaborn as sns
        sns.scatterplot(
            x=X_scaled[:, 0],
            y=X_scaled[:, 1],
            hue=y,
            palette="Set1",
            edgecolor="k",
            ax=ax,
        )
    except ModuleNotFoundError:
        scatter = ax.scatter(
            X_scaled[:, 0],
            X_scaled[:, 1],
            c=y,
            cmap=plt.cm.Set1,
            edgecolors="k",
        )
        ax.legend(*scatter.legend_elements(), title="Class")
        
    ax.set_title(f"Decision Boundary with Best k = {k}")
    ax.set_xlabel(f"{feature_names[0]} (scaled)")
    ax.set_ylabel(f"{feature_names[1]} (scaled)")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
