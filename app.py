import json
import pickle
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

KNN_ARTIFACTS_DIR = Path("models/k_nearest_neighbor/artifacts")
KNN_LOGS_DIR = Path("models/k_nearest_neighbor/logs")
NB_ARTIFACTS_DIR = Path("models/naive_bayes/artifacts")
NB_LOGS_DIR = Path("models/naive_bayes/logs")

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


def save_confusion_matrix_plot_local(y_true, y_pred, output_path, cmap="Blues"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1"])
    # We set colorbar=False and no title to ensure perfect aspect ratio and layout symmetry.
    disp.plot(cmap=cmap, ax=ax, colorbar=False)
    ax.grid(False)
    fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    fig.savefig(output_path, dpi=100)
    plt.close(fig)
    return output_path


def save_decision_boundary_plot_local(model, X_scaled, y, output_path, feature_names):
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
        
    ax.set_xlabel(f"{feature_names[0]} (scaled)")
    ax.set_ylabel(f"{feature_names[1]} (scaled)")
    ax.grid(True)
    fig.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.12)
    fig.savefig(output_path, dpi=100)
    plt.close(fig)
    return output_path


@st.cache_resource
def get_knn_result():
    model_path = KNN_ARTIFACTS_DIR / "knn_model.pkl"
    log_path = KNN_LOGS_DIR / "training_results.jsonl"
    
    # Load model payload
    with open(model_path, "rb") as f:
        payload = pickle.load(f)
    
    # Load logs to get default/baseline parameters if needed
    records = []
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
                    
    # Find matching record in logs for k=13 (newest first)
    matched_record = None
    for record in reversed(records):
        if record.get("params", {}).get("k") == 13:
            matched_record = record
            break
            
    if matched_record is None and records:
        matched_record = records[-1]
        
    # Determine the target k (always 13 since it is fixed now)
    target_k = 13
        
    # Load test data and compute metrics dynamically to ensure they match target_k
    from models.k_nearest_neighbor.preprocess import prepare_diabetes_data
    from models.k_nearest_neighbor.evaluate import evaluate_predictions
    
    # Final model data (using selected features)
    data = prepare_diabetes_data(
        feature_columns=payload["feature_columns"],
        test_size=0.3,
        random_state=666,
    )
    
    # Update loaded model's k and evaluate
    model = payload["model"]
    model.n_neighbors = target_k
    model.k = target_k
    y_pred = model.predict(data["X_test"])
    test_metrics = evaluate_predictions(data["y_test"], y_pred)
    
    # Baseline model data (using all features)
    from models.k_nearest_neighbor.preprocess import FEATURE_COLUMNS
    all_feature_data = prepare_diabetes_data(
        feature_columns=FEATURE_COLUMNS,
        test_size=0.3,
        random_state=666,
    )
    from models.k_nearest_neighbor.knn import KNeighborsClassifier
    baseline_model = KNeighborsClassifier(k=target_k)
    baseline_model.fit(all_feature_data["X_train"], all_feature_data["y_train"])
    baseline_pred = baseline_model.predict(all_feature_data["X_test"])
    baseline_test_metrics = evaluate_predictions(all_feature_data["y_test"], baseline_pred)
    
    # Paths for plots corresponding to this target_k
    confusion_matrix_plot_path = KNN_ARTIFACTS_DIR / f"confusion_matrix_k{target_k}.png"
    decision_boundary_plot_path = KNN_ARTIFACTS_DIR / f"decision_boundary_k{target_k}.png"
    
    # Generate and save plots dynamically using local unified plotting functions
    save_confusion_matrix_plot_local(
        data["y_test"],
        y_pred,
        confusion_matrix_plot_path,
        cmap="Blues",
    )
    
    boundary_data = prepare_diabetes_data(
        feature_columns=payload["feature_columns"][:2],
        test_size=0.3,
        random_state=666,
    )
    boundary_model = KNeighborsClassifier(k=target_k)
    boundary_model.fit(boundary_data["X_train"], boundary_data["y_train"])
    save_decision_boundary_plot_local(
        boundary_model,
        boundary_data["X_train"],
        boundary_data["y_train"],
        decision_boundary_plot_path,
        feature_names=boundary_data["feature_columns"],
    )
        
    # Other standard paths
    cv_plot_path = KNN_ARTIFACTS_DIR / "knn_cv_accuracy.png"
    correlation_plot_path = KNN_ARTIFACTS_DIR / "feature_correlation_heatmap.png"
    selected_features_path = KNN_ARTIFACTS_DIR / "selected_features.json"
    
    result = {
        "model": model,
        "scaler": payload["scaler"],
        "feature_columns": payload["feature_columns"],
        "class_labels": payload["class_labels"],
        "k": target_k,
        "baseline_k": matched_record["baseline_tuning_result"]["best_k"] if (matched_record and matched_record.get("baseline_tuning_result") and matched_record["baseline_tuning_result"]) else target_k,
        "test_metrics": test_metrics,
        "baseline_test_metrics": baseline_test_metrics,
        "confusion_matrix_plot_path": confusion_matrix_plot_path,
        "decision_boundary_plot_path": decision_boundary_plot_path,
        "cv_plot_path": cv_plot_path,
        "correlation_plot_path": correlation_plot_path,
        "selected_features_path": selected_features_path,
        "model_path": model_path,
        "log_path": log_path,
    }
    return result


@st.cache_resource
def get_nb_result():
    model_path = NB_ARTIFACTS_DIR / "nb_model.pkl"
    log_path = NB_LOGS_DIR / "training_results.jsonl"
    
    # Load model payload
    with open(model_path, "rb") as f:
        payload = pickle.load(f)
        
    # Load logs
    records = []
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
                    
    matched_record = records[-1] if records else None
    
    if not matched_record:
        # If no record exists, run training to generate it
        train_results = train_naive_bayes_model()
        if log_path.exists():
            with open(log_path, "r", encoding="utf-8") as f:
                records = [json.loads(line.strip()) for line in f if line.strip()]
            matched_record = records[-1] if records else None
            
    # Load test data and compute metrics dynamically
    from models.naive_bayes.src.preprocess import prepare_diabetes_data
    from models.naive_bayes.src.evaluate import evaluate_predictions
    
    data = prepare_diabetes_data(
        feature_columns=payload["feature_columns"],
        test_size=0.2,
        random_state=42,
    )
    
    model = payload["model"]
    y_pred = model.predict(data["X_test"])
    test_metrics = evaluate_predictions(data["y_test"], y_pred)
    
    confusion_matrix_plot_path = NB_ARTIFACTS_DIR / "confusion_matrix.png"
    decision_boundary_plot_path = NB_ARTIFACTS_DIR / "decision_boundary.png"
    
    # Generate and save plots dynamically using local unified plotting functions
    save_confusion_matrix_plot_local(
        data["y_test"],
        y_pred,
        confusion_matrix_plot_path,
        cmap="Oranges",
    )
    
    boundary_data = prepare_diabetes_data(
        feature_columns=payload["feature_columns"][:2],
        test_size=0.2,
        random_state=42,
    )
    from models.naive_bayes.src.naive_bayes import NaiveBayesClassifier
    boundary_model_fit = NaiveBayesClassifier()
    boundary_model_fit.fit(boundary_data["X_train"], boundary_data["y_train"])
    
    save_decision_boundary_plot_local(
        boundary_model_fit,
        boundary_data["X_train"],
        boundary_data["y_train"],
        decision_boundary_plot_path,
        feature_names=boundary_data["feature_columns"],
    )
    
    return {
        "model": model,
        "scaler": payload["scaler"],
        "feature_columns": payload["feature_columns"],
        "class_labels": payload["class_labels"],
        "test_metrics": test_metrics,
        "confusion_matrix_plot_path": confusion_matrix_plot_path,
        "decision_boundary_plot_path": decision_boundary_plot_path,
        "model_path": model_path,
        "log_path": log_path,
    }


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
    
    # Kiểm tra xem scaler được fit trên bao nhiêu đặc trưng để biến đổi chính xác
    if hasattr(scaler, "n_features_in_") and scaler.n_features_in_ == len(feature_cols):
        # Nếu scaler chỉ được fit trên các đặc trưng được chọn (như KNN)
        input_subset = input_df[feature_cols]
        input_scaled = scaler.transform(input_subset)
    else:
        # Nếu scaler được fit trên toàn bộ đặc trưng (như Naive Bayes)
        input_scaled = scaler.transform(input_df)
        input_scaled_df = pd.DataFrame(input_scaled, columns=all_features)
        input_scaled = input_scaled_df[feature_cols].values

    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
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
    st.markdown(f"**Các đặc trưng được PSO lựa chọn cho mô hình KNN:** {', '.join(knn_result['feature_columns'])}")
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
        st.caption(f"Selected features file: {knn_result['selected_features_path']}")
        st.caption(f"Selected features: {', '.join(knn_result['feature_columns'])}")
        st.caption(f"Training log: {knn_result['log_path']}")
    with col2:
        st.markdown("**Naive Bayes Artifacts:**")
        st.caption(f"Saved model: {nb_result['model_path']}")
        st.caption(f"Confusion matrix plot: {nb_result['confusion_matrix_plot_path']}")
        st.caption(f"Decision boundary plot: {nb_result['decision_boundary_plot_path']}")
        st.caption(f"Training log: {nb_result['log_path']}")
