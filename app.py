from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import pickle
import json
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Import model và pso trực tiếp từ file người dùng
from models.k_nearest_neighbor.knn import KNeighborsClassifier
from models.k_nearest_neighbor.pso import BinaryPSOFeatureSelection
from models.naive_bayes.src.train import train_naive_bayes_model


st.set_page_config(page_title="Diabetes Prediction", layout="wide")


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
    return {
        "confusion_matrix": cm.tolist(),
    }


def save_confusion_matrix_plot_local(y_true, y_pred, output_path, cmap="Blues", title="Confusion Matrix"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1"])
    disp.plot(cmap=cmap, ax=ax)
    ax.set_title(title)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def save_decision_boundary_plot_local(model, X_scaled, y, output_path, feature_names, title="Decision Boundary"):
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
    except Exception:
        scatter = ax.scatter(
            X_scaled[:, 0],
            X_scaled[:, 1],
            c=y,
            cmap=plt.cm.Set1,
            edgecolors="k",
        )
        ax.legend(*scatter.legend_elements(), title="Class")
        
    ax.set_title(title)
    ax.set_xlabel(f"{feature_names[0]} (scaled)")
    ax.set_ylabel(f"{feature_names[1]} (scaled)")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


@st.cache_resource
def get_knn_result():
    # Load dataset
    dataset_path = Path(__file__).resolve().parent / "dataset" / "diabetes.csv"
    df = pd.read_csv(dataset_path)

    # Loại bỏ outliers giống test.ipynb
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR)))
    df = df[~outliers.any(axis=1)]

    feature_names = df.drop('Outcome', axis=1).columns
    FEATURE_COLUMNS = list(feature_names)

    X = df.drop('Outcome', axis=1).values
    y = df['Outcome'].values

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=666)

    # Scale dữ liệu
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train / Val split để chọn đặc trưng bằng PSO
    X_train_sub_scaled, X_val_scaled, y_train_sub, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=42
    )

    # Chạy PSO Feature Selection (k = 13 default)
    selected_k = 13
    knn_model = KNeighborsClassifier(n_neighbors=selected_k)
    np.random.seed(1)
    pso_selector = BinaryPSOFeatureSelection(
        num_features=X_train_scaled.shape[1],
        estimator=knn_model,
        num_particles=15,
        max_iter=30,
        verbose=False
    )
    best_feature_mask, best_val_accuracy = pso_selector.fit(
        X_train_sub_scaled, y_train_sub, X_val_scaled, y_val
    )

    selected_feature_indices = np.where(best_feature_mask == 1)[0]
    selected_feature_names = [FEATURE_COLUMNS[i] for i in selected_feature_indices]

    # Baseline Model (All features)
    knn_baseline = KNeighborsClassifier(n_neighbors=selected_k)
    knn_baseline.fit(X_train_scaled, y_train)
    y_pred_baseline = knn_baseline.predict(X_test_scaled)
    baseline_test_metrics = evaluate_predictions(y_test, y_pred_baseline)

    # PSO Model (Selected features)
    X_train_pso = X_train_scaled[:, selected_feature_indices]
    X_test_pso = X_test_scaled[:, selected_feature_indices]
    
    knn_pso = KNeighborsClassifier(n_neighbors=selected_k)
    knn_pso.fit(X_train_pso, y_train)
    y_pred_pso = knn_pso.predict(X_test_pso)
    test_metrics = evaluate_predictions(y_test, y_pred_pso)
    classification_details = evaluate_classification_details(y_test, y_pred_pso)

    # Save artifacts & plots
    model_dir = Path(__file__).resolve().parent / "models" / "k_nearest_neighbor" / "artifacts"
    log_dir = Path(__file__).resolve().parent / "models" / "k_nearest_neighbor" / "logs"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    confusion_matrix_plot_path = save_confusion_matrix_plot_local(
        y_test, y_pred_pso, model_dir / "confusion_matrix.png", cmap="Blues", title=f"KNN Confusion Matrix"
    )

    # Decision boundary trên 2 đặc trưng quan trọng đầu tiên
    boundary_data_X_train = X_train_scaled[:, selected_feature_indices[:2]]
    boundary_model = KNeighborsClassifier(n_neighbors=selected_k)
    boundary_model.fit(boundary_data_X_train, y_train)
    decision_boundary_plot_path = save_decision_boundary_plot_local(
        boundary_model,
        boundary_data_X_train,
        y_train,
        model_dir / "decision_boundary.png",
        feature_names=selected_feature_names[:2],
        title=f"KNN Decision Boundary (k={selected_k})"
    )

    # Lưu model và scaler
    model_path = model_dir / "knn_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": knn_pso,
            "scaler": scaler,
            "feature_columns": selected_feature_names,
            "class_labels": [0, 1]
        }, f)

    # Log results
    log_path = log_dir / "training_results.jsonl"
    log_record = {
        "timestamp": datetime.now().isoformat(),
        "params": {"k": selected_k, "max_iter": 30, "num_particles": 15},
        "test_metrics": test_metrics,
        "selected_features": selected_feature_names
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_record) + "\n")

    return {
        "model": knn_pso,
        "scaler": scaler,
        "feature_columns": selected_feature_names,
        "class_labels": [0, 1],
        "test_metrics": test_metrics,
        "baseline_test_metrics": baseline_test_metrics,
        "confusion_matrix_plot_path": confusion_matrix_plot_path,
        "decision_boundary_plot_path": decision_boundary_plot_path,
        "model_path": model_path,
        "log_path": log_path,
        "selected_features_path": model_dir / "selected_features.json",
        "cv_plot_path": model_dir / "knn_cv_accuracy.png"
    }


@st.cache_resource
def get_nb_result():
    return train_naive_bayes_model()


def get_relative_path(path_str):
    if not path_str:
        return ""
    try:
        path = Path(path_str).resolve()
        root = Path(__file__).resolve().parent
        return str(path.relative_to(root))
    except Exception:
        return str(path_str)


def predict_patient(result, input_values):
    model = result["model"]
    scaler = result["scaler"]
    feature_cols = result["feature_columns"]

    # Đảm bảo các feature tương thích
    all_features = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
    ]
    input_df = pd.DataFrame([input_values], columns=all_features)
    
    # Scale trên toàn bộ đặc trưng rồi lọc các đặc trưng được chọn
    input_scaled = scaler.transform(input_df)
    input_scaled_df = pd.DataFrame(input_scaled, columns=all_features)
    input_pso = input_scaled_df[feature_cols].values

    prediction = model.predict(input_pso)[0]
    probabilities = model.predict_proba(input_pso)[0]
    positive_index = list(result["class_labels"]).index(1)
    positive_probability = probabilities[positive_index]
    return prediction, positive_probability


st.title("Diabetes Prediction AI Model")
st.markdown("---")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page:", ["Home", "Make Prediction", "Model Performance"])
selected_model = st.sidebar.radio("Select Model:", ["K-Nearest Neighbors (KNN)", "Naive Bayes"])

knn_result = None
nb_result = None

if page == "Model Performance":
    knn_result = get_knn_result()
    nb_result = get_nb_result()
else:
    if selected_model == "K-Nearest Neighbors (KNN)":
        knn_result = get_knn_result()
    else:
        nb_result = get_nb_result()


if page == "Home":
    st.header("Welcome to Diabetes Prediction System")
    st.write(
        """
        This application uses two Machine Learning algorithms to predict diabetes risk
        from patient health metrics:
        
        1. **K-Nearest Neighbors (KNN)**: An instance-based classifier.
           - Optimized using **Binary Particle Swarm Optimization (PSO)** for feature selection.
           - Configured with $k = 13$ as the default parameter.
        2. **Naive Bayes**: A probabilistic classifier based on Bayes' theorem.
        
        Both models are trained on the `dataset/diabetes.csv` dataset, using standard preprocessing and scaling. 
        Use the sidebar to navigate to the prediction page or inspect the detailed performance comparison.
        """
    )
    st.info("Use the sidebar to make a prediction or inspect model performance.")

elif page == "Make Prediction":
    st.header(f"Make a Prediction (Active Model: {selected_model})")
    st.write("Enter patient health metrics to predict diabetes risk:")

    col1, col2, col3 = st.columns(3)

    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=17, value=0)
        glucose = st.number_input("Glucose", min_value=0, max_value=200, value=100)
        blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)

    with col2:
        skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
        insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
        bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0)

    with col3:
        diabetes_pedigree = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=3.0,
            value=0.5,
        )
        age = st.number_input("Age", min_value=18, max_value=100, value=33)

    if st.button("Predict", use_container_width=True):
        input_values = np.array(
            [
                pregnancies,
                glucose,
                blood_pressure,
                skin_thickness,
                insulin,
                bmi,
                diabetes_pedigree,
                age,
            ]
        )

        st.markdown("### Prediction Result")
        
        if selected_model == "K-Nearest Neighbors (KNN)":
            knn_pred, knn_prob = predict_patient(knn_result, input_values)
            knn_conf = knn_prob if knn_pred == 1 else 1 - knn_prob
            
            st.subheader("K-Nearest Neighbors (KNN)")
            if knn_pred == 1:
                st.error("Prediction: Positive (Diabetes)")
            else:
                st.success("Prediction: Negative (Normal)")
            st.metric("Confidence", f"{knn_conf * 100:.1f}%")
            st.caption(f"Selected Features used: {', '.join(knn_result['feature_columns'])}")
            
        else:
            nb_pred, nb_prob = predict_patient(nb_result, input_values)
            nb_conf = nb_prob if nb_pred == 1 else 1 - nb_prob
            
            st.subheader("Naive Bayes")
            if nb_pred == 1:
                st.error("Prediction: Positive (Diabetes)")
            else:
                st.success("Prediction: Negative (Normal)")
            st.metric("Confidence", f"{nb_conf * 100:.1f}%")
            st.caption(f"Selected Features used: {', '.join(nb_result['feature_columns'])}")

        st.markdown("---")
        st.info("This prediction is for study/demo purposes and is not a medical diagnosis.")

elif page == "Model Performance":
    st.header("Model Performance Metrics")

    metrics_df = pd.DataFrame(
        [
            {
                "Model": "KNN (All features, k=13)",
                **{name: round(value, 4) for name, value in knn_result["baseline_test_metrics"].items()},
            },
            {
                "Model": "KNN (PSO Selected features, k=13)",
                **{name: round(value, 4) for name, value in knn_result["test_metrics"].items()},
            },
            {
                "Model": "Naive Bayes (All features)",
                **{name: round(value, 4) for name, value in nb_result["test_metrics"].items()},
            },
        ]
    )

    st.subheader("Model Performance Comparison")
    st.dataframe(metrics_df, use_container_width=True)

    st.subheader("Metric Comparison")
    chart_data = metrics_df.set_index("Model")
    st.bar_chart(chart_data)

    st.subheader("Visual Model Evaluations")
    
    tab1, tab2 = st.tabs(["Confusion Matrices", "Decision Boundaries"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### KNN (k=13)")
            st.image(get_relative_path(knn_result["confusion_matrix_plot_path"]), use_column_width=True)
        with col2:
            st.markdown("##### Naive Bayes")
            st.image(get_relative_path(nb_result["confusion_matrix_plot_path"]), use_column_width=True)
            
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### KNN (k=13)")
            st.image(get_relative_path(knn_result["decision_boundary_plot_path"]), use_column_width=True)
        with col2:
            st.markdown("##### Naive Bayes")
            st.image(get_relative_path(nb_result["decision_boundary_plot_path"]), use_column_width=True)

    st.markdown("---")
    st.subheader("Artifact Locations")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**KNN Artifacts:**")
        st.caption(f"Saved model: {knn_result['model_path']}")
        st.caption(f"Confusion matrix plot: {knn_result['confusion_matrix_plot_path']}")
        st.caption(f"Decision boundary plot: {knn_result['decision_boundary_plot_path']}")
        st.caption(f"Training log: {knn_result['log_path']}")
    with col2:
        st.markdown("**Naive Bayes Artifacts:**")
        st.caption(f"Saved model: {nb_result['model_path']}")
        st.caption(f"Confusion matrix plot: {nb_result['confusion_matrix_plot_path']}")
        st.caption(f"Decision boundary plot: {nb_result['decision_boundary_plot_path']}")
        st.caption(f"Training log: {nb_result['log_path']}")
