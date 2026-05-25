import numpy as np
import pandas as pd
import streamlit as st

from models.knn.src.preprocess import FEATURE_COLUMNS
from models.knn.src.train import train_knn_model


st.set_page_config(page_title="Diabetes Prediction", layout="wide")


@st.cache_resource
def get_knn_result(tune_k, k):
    return train_knn_model(k=k, tune_k=tune_k)


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

if page == "Home":
    st.header("Welcome to Diabetes Prediction System")
    st.write(
        """
        This application uses a K-Nearest Neighbors model to predict diabetes risk
        from patient health metrics.

        The model is trained from `dataset/diabetes.csv`, uses a train/test split,
        tunes k with cross-validation on the training set, and standardizes features.
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

        prediction, positive_probability = predict_patient(knn_result, input_values)
        confidence = positive_probability if prediction == 1 else 1 - positive_probability

        result_col, confidence_col = st.columns(2)

        with result_col:
            if prediction == 1:
                st.error("Prediction: Positive")
            else:
                st.success("Prediction: Negative")

        with confidence_col:
            st.metric("Confidence", f"{confidence * 100:.1f}%")

        st.markdown("---")
        st.info("This prediction is for study/demo purposes and is not a medical diagnosis.")

elif page == "Model Performance":
    st.header("Model Performance Metrics")

    metrics_df = pd.DataFrame(
        [
            {
                "Model": "All features baseline",
                **{name: round(value, 4) for name, value in knn_result["baseline_test_metrics"].items()},
            },
            {
                "Model": "Selected features",
                **{name: round(value, 4) for name, value in knn_result["test_metrics"].items()},
            },
        ]
    )

    st.subheader(f"K-Nearest Neighbors Performance (k={knn_result['k']})")
    st.dataframe(metrics_df, use_container_width=True)

    deltas_df = pd.DataFrame(
        [
            {
                "Metric": metric,
                "Delta vs baseline": round(value, 4),
            }
            for metric, value in knn_result["metric_deltas"].items()
        ]
    )
    st.subheader("Selected Feature Impact")
    st.dataframe(deltas_df, use_container_width=True)

    st.subheader("Metric Comparison")
    chart_data = metrics_df.set_index("Model")
    st.bar_chart(chart_data)

    st.subheader("Selected Features")
    st.write(", ".join(knn_result["feature_columns"]))

    st.caption(f"Saved model: {knn_result['model_path']}")
    st.caption(f"CV plot: {knn_result['cv_plot_path']}")
    st.caption(f"Correlation plot: {knn_result['correlation_plot_path']}")
    st.caption(f"Confusion matrix plot: {knn_result['confusion_matrix_plot_path']}")
    st.caption(f"Decision boundary plot: {knn_result['decision_boundary_plot_path']}")
    st.caption(f"Selected features file: {knn_result['selected_features_path']}")
    st.caption(f"Training log: {knn_result['log_path']}")
