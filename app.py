from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from models.knn.src.preprocess import FEATURE_COLUMNS
from models.knn.src.train import train_knn_model
from models.naive_bayes.src.train import train_naive_bayes_model


st.set_page_config(page_title="Diabetes Prediction", layout="wide")


@st.cache_resource
def get_knn_result(tune_k, k):
    return train_knn_model(k=k, tune_k=tune_k)


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
        st.caption(f"Training log: {knn_result['log_path']}")
    with col2:
        st.markdown("**Naive Bayes Artifacts:**")
        st.caption(f"Saved model: {nb_result['model_path']}")
        st.caption(f"Confusion matrix plot: {nb_result['confusion_matrix_plot_path']}")
        st.caption(f"Decision boundary plot: {nb_result['decision_boundary_plot_path']}")
        st.caption(f"Training log: {nb_result['log_path']}")
