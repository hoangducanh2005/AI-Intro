# Diabetes Prediction AI System

A comprehensive machine learning system designed to predict diabetes risk from patient clinical metrics using **K-Nearest Neighbors (KNN)** and **Naive Bayes (NB)** classifiers. This project implements a custom KNN model from scratch, integrates **Binary Particle Swarm Optimization (BPSO)** for feature selection, and provides a **Streamlit Web Application** for interactive prediction and performance visualization.

---

## 🌟 Key Features

1. **Custom KNN Classifier (NumPy)**: 
   - Built from scratch using NumPy arrays to support custom distance metrics (Euclidean and Manhattan).
   - Custom tie-breaking logic using index classification ordering.
   - Built-in probability estimation (`predict_proba`) based on neighbor class proportions.

2. **Feature Selection using Binary PSO**:
   - Implements a Binary Particle Swarm Optimization (BPSO) metaheuristic to find the optimal subset of input features.
   - Designed with a multi-objective Fitness Function:
     $$
     \text{Fitness} = \alpha \cdot \text{Accuracy}_{\text{val}} + (1 - \alpha) \cdot \left(1 - \frac{N_{\text{selected}}}{N_{\text{total}}}\right)
     $$
     (where $\alpha = 0.99$ to prioritize diagnosis accuracy while encouraging feature pruning).
   - Successfully reduced feature space by **62.5%** (from 8 features to 3: *Glucose*, *DiabetesPedigreeFunction*, and *Age*).

3. **Standard Baseline Comparison**:
   - Compared against a probabilistic **Naive Bayes Classifier** implemented using probability distributions.
   - Comparative evaluation including accuracy, precision, recall, F1-score, and confusion matrices.

4. **Streamlit Web Interface**:
   - **Home**: Project introduction and summary of models.
   - **Make Prediction**: Interactive sliders and number inputs for clinical data to make real-time predictions with confidence scores.
   - **Model Performance**: Visual comparison of metrics, confusion matrices, and decision boundary plots.

---

## 📂 Project Structure

```
├── app.py                      # Main Streamlit web application
├── requirements.txt            # Python package dependencies
├── README.md                   # Project documentation
├── dataset/
│   ├── diabetes.csv            # Pima Indians Diabetes Dataset
│   └── test.ipynb              # Exploration, training, and visualization notebook
├── data_preprocessing/
│   └── preprocess.py           # Core preprocessing utilities
├── models/
│   ├── k_nearest_neighbor/
│   │   ├── knn.py              # Custom KNN Classifier implementation
│   │   ├── pso.py              # Binary PSO Feature Selector implementation
│   │   ├── preprocess.py       # Data preparation for KNN
│   │   ├── train.py            # Model fitting and artifact generation
│   │   ├── tune.py             # Hyperparameter k-tuning via Validation Hold-out
│   │   ├── evaluate.py         # Metrics calculation and plotting
│   │   ├── report.tex          # LaTeX research report (Ch. II & Ch. III)
│   │   ├── artifacts/          # KNN models, JSON logs, and figure files
│   │   └── images/             # Extracted PNG images for LaTeX inclusion
│   └── naive_bayes/
│       ├── main.py             # Training and evaluation runner for NB
│       ├── src/
│       │   ├── naive_bayes.py  # Naive Bayes Classifier implementation
│       │   ├── preprocess.py   # NB data preprocessing
│       │   ├── train.py        # NB training controller
│       │   └── evaluate.py     # NB evaluation helper
│       └── artifacts/          # NB models, JSON logs, and figures
```

---

## 🚀 Installation & Running

### 1. Prerequisites
Ensure you have Python 3.9+ installed.

### 2. Install Dependencies
Install all required libraries using the package manager:
```bash
pip install -r requirements.txt
```

### 3. Run the Web App
Launch the Streamlit web application locally:
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 📊 Experimental Results

Evaluating the models on a $30\%$ test partition of the cleaned **Pima Indians Diabetes** dataset (outliers removed via IQR, sample size = 639) yielded the following results:

| Model | Active Features | Test Accuracy | Precision | Recall (Class 1) | F1-Score (Class 1) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline KNN ($k=13$)** | 8 | 81.25% | 79.41% | 48.21% | 60.00% |
| **PSO-KNN ($k=13$)** | **3** | **82.81%** | 78.05% | **57.14%** | **65.98%** |
| **Naive Bayes** | 8 | 76.56% | 60.00% | 64.29% | 62.07% |

*Key Findings:*
- Running **BPSO** for feature selection reduced input parameters by 62.5%, selecting only **Glucose**, **DiabetesPedigreeFunction**, and **Age**.
- Despite using fewer features, **PSO-KNN** improved overall accuracy by **1.56%** and significantly increased Class 1 recall by **8.93%** (critical for medical screenings to minimize false negatives).

---

## ✍️ Authors

This project was developed as a final assignment for the Introduction to Artificial Intelligence course by:
- **Nguyễn Việt Dũng** (20235688)
- **Hoàng Đức Anh** (20235640)
- **Nguyễn Nhật Minh** (20235781)
- **Đỗ Khắc Hoàng** (20235721)

*Instructor:* **TS. Đỗ Tiến Dũng** (Đại học Bách Khoa Hà Nội)
