import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from models.k_nearest_neighbor.preprocess import FEATURE_COLUMNS
from models.k_nearest_neighbor.train import train_knn_model
from models.naive_bayes.src.train import train_naive_bayes_model

KNN_ARTIFACTS_DIR = Path("models/k_nearest_neighbor/artifacts")
KNN_LOGS_DIR = Path("models/k_nearest_neighbor/logs")
NB_ARTIFACTS_DIR = Path("models/naive_bayes/artifacts")
NB_LOGS_DIR = Path("models/naive_bayes/logs")

st.set_page_config(page_title="Diabetes Prediction", layout="wide")


@st.cache_resource
def get_knn_result(tune_k, k):
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
                    
    matched_record = None
    for record in reversed(records):
        log_params = record.get("params", {})
        if log_params.get("tune_k") == tune_k:
            if tune_k or log_params.get("k") == k:
                matched_record = record
                break
                
    if matched_record is None and records:
        matched_record = records[-1]
        
    # Determine the target k
    if k is not None:
        target_k = k
    elif matched_record:
        target_k = matched_record["params"]["k"]
    else:
        target_k = 13  # Default fallback if no logs/parameters exist
        
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
    
    # Generate and save plots if they don't exist
    if not confusion_matrix_plot_path.exists():
        from models.k_nearest_neighbor.evaluate import save_confusion_matrix_plot
        save_confusion_matrix_plot(
            data["y_test"],
            y_pred,
            confusion_matrix_plot_path,
            title=f"Confusion Matrix (k={target_k})",
        )
        
    if not decision_boundary_plot_path.exists():
        from models.k_nearest_neighbor.evaluate import save_decision_boundary_plot
        boundary_data = prepare_diabetes_data(
            feature_columns=payload["feature_columns"][:2],
            test_size=0.3,
            random_state=666,
        )
        boundary_model = KNeighborsClassifier(k=target_k)
        boundary_model.fit(boundary_data["X_train"], boundary_data["y_train"])
        save_decision_boundary_plot(
            boundary_model,
            boundary_data["X_train"],
            boundary_data["y_train"],
            decision_boundary_plot_path,
            feature_names=boundary_data["feature_columns"],
            k=target_k,
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
        "baseline_k": matched_record["baseline_tuning_result"]["best_k"] if (matched_record and matched_record.get("baseline_tuning_result")) else target_k,
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
    
    if matched_record:
        return {
            "model": payload["model"],
            "scaler": payload["scaler"],
            "feature_columns": payload["feature_columns"],
            "class_labels": payload["class_labels"],
            "test_metrics": matched_record["test_metrics"],
            "confusion_matrix_plot_path": Path(matched_record["confusion_matrix_plot_path"]) if matched_record.get("confusion_matrix_plot_path") else NB_ARTIFACTS_DIR / "confusion_matrix.png",
            "decision_boundary_plot_path": Path(matched_record["decision_boundary_plot_path"]) if matched_record.get("decision_boundary_plot_path") else NB_ARTIFACTS_DIR / "decision_boundary.png",
            "model_path": model_path,
            "log_path": log_path,
        }
    else:
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

    input_df = pd.DataFrame([input_values], columns=FEATURE_COLUMNS)
    input_df = input_df[result["feature_columns"]]
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    positive_index = list(result["class_labels"]).index(1)
    positive_probability = probabilities[positive_index]
    return prediction, positive_probability


st.title("Diabetes Prediction AI Model")
st.markdown("---")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page:", ["Home", "Make Prediction", "Model Performance"])
tune_k = st.sidebar.checkbox("Auto tune k", value=True)
k_value = st.sidebar.slider("K value", min_value=1, max_value=25, value=5, step=2, disabled=tune_k)

knn_result = get_knn_result(tune_k, None if tune_k else k_value)
nb_result = get_nb_result()

if page == "Home":
    st.header("Welcome to Diabetes Prediction System")
    st.write(
        """
        This application uses two Machine Learning algorithms to predict diabetes risk
        from patient health metrics:
        
        1. **K-Nearest Neighbors (KNN)**: An instance-based classifier.
        2. **Naive Bayes**: A probabilistic classifier based on Bayes' theorem.
        
        Both models are trained on the `dataset/diabetes.csv` dataset, using standard preprocessing and scaling. 
        Use the sidebar to navigate to the prediction page or inspect the detailed performance comparison.
        """
    )
    st.info("Use the sidebar to make a prediction or inspect model performance.")

elif page == "Make Prediction":
    st.header("Make a Prediction")
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

        st.markdown("### Prediction Comparison")
        knn_col, nb_col = st.columns(2)

        with knn_col:
            st.subheader("K-Nearest Neighbors (KNN)")
            knn_pred, knn_prob = predict_patient(knn_result, input_values)
            knn_conf = knn_prob if knn_pred == 1 else 1 - knn_prob
            if knn_pred == 1:
                st.error("Prediction: Positive (Diabetes)")
            else:
                st.success("Prediction: Negative (Normal)")
            st.metric("Confidence", f"{knn_conf * 100:.1f}%")

        with nb_col:
            st.subheader("Naive Bayes")
            nb_pred, nb_prob = predict_patient(nb_result, input_values)
            nb_conf = nb_prob if nb_pred == 1 else 1 - nb_prob
            if nb_pred == 1:
                st.error("Prediction: Positive (Diabetes)")
            else:
                st.success("Prediction: Negative (Normal)")
            st.metric("Confidence", f"{nb_conf * 100:.1f}%")

        st.markdown("---")
        st.info("This prediction is for study/demo purposes and is not a medical diagnosis.")

elif page == "Model Performance":
    st.header("Model Performance Metrics")

    metrics_df = pd.DataFrame(
        [
            {
                "Model": f"KNN (All features, k={knn_result['baseline_k']})",
                **{name: round(value, 4) for name, value in knn_result["baseline_test_metrics"].items()},
            },
            {
                "Model": f"KNN (Selected features, k={knn_result['k']})",
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
            st.markdown(f"##### KNN (k={knn_result['k']})")
            st.image(get_relative_path(knn_result["confusion_matrix_plot_path"]), use_column_width=True)
        with col2:
            st.markdown("##### Naive Bayes")
            st.image(get_relative_path(nb_result["confusion_matrix_plot_path"]), use_column_width=True)
            
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"##### KNN (k={knn_result['k']})")
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
        st.caption(f"CV plot: {knn_result['cv_plot_path']}")
        st.caption(f"Correlation plot: {knn_result['correlation_plot_path']}")
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
