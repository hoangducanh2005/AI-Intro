# Diabetes Prediction AI Model

## Description
This project implements machine learning models to predict diabetes in patients based on various medical features. The system uses multiple classification algorithms to identify individuals at risk of diabetes with high accuracy.

## Algorithms Used
The project compares the performance of four machine learning algorithms:
- **K-Nearest Neighbors (KNN)** - Instance-based learning algorithm
- **Naive Bayes** - Probabilistic classifier based on Bayes' theorem
- **Decision Tree** - Tree-based classification algorithm
- **Random Forest** - Ensemble learning method using multiple decision trees

## Dataset
The project uses diabetes datasets containing patient medical records and health metrics as features to predict diabetes diagnosis.

## Project Structure
```
├── main.py                 # Main script to run the project
├── preprocessing/          # Data preprocessing and cleaning
├── models/                 # Model implementations (KNN, Decision Tree, etc.)
├── evaluation/             # Model evaluation and performance metrics
├── visualization/          # Data visualization utilities
├── dataset/                # Dataset files and exploration notebooks
└── requirements.txt        # Project dependencies
```

## Installation
Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Run the main script to train and evaluate all models:
```bash
python main.py
```

## Results
The models are evaluated using standard metrics including accuracy, precision, recall, and F1-score to determine the best performing algorithm for diabetes prediction.
