# Walkthrough - Binary PSO Feature Selection for KNN (Updated)

We have successfully implemented and refined the **Binary Particle Swarm Optimization (PSO)** for **Feature Selection** on the Diabetes Detection problem using our custom KNN classifier. Below is a summary of how we resolved the stochastic evaluation results and the final verified performance.

## Resolution of Stochastic/Identical Results

1. **Fitness Penalty for Feature Counts**: 
   Initially, when multiple feature subsets achieved the same highest validation accuracy (e.g. `0.8111`), the PSO had no preference toward selecting the smaller subset. This caused it to sometimes select all 8 features or larger sub-optimal masks, yielding identical results to the baseline model on the test set.
   * **Fix**: We modified the fitness function in [pso.py](file:///c:/Users/Admin/GIT/AI-Intro/models/k_nearest_neighbor/pso.py) to incorporate a penalty for the number of selected features:
     $$\text{Fitness} = \alpha \cdot \text{Accuracy} + (1 - \alpha) \cdot \left(1 - \frac{N_{\text{selected}}}{N_{\text{total}}}\right)$$
     where $\alpha = 0.99$. This heavily prioritizes validation accuracy but acts as a tie-breaker, rewarding particles that select fewer features.
2. **Deterministic Execution**:
   * **Fix**: We added `np.random.seed(42)` to the initialization step in [test.ipynb](file:///c:/Users/Admin/GIT/AI-Intro/models/k_nearest_neighbor/test.ipynb) to make the particle initialization and random velocity updates deterministic.

---

## Evaluation & Results

### 1. Selected Features
The PSO selector consistently and deterministically selects **3 out of 8** features:
* `Glucose`
* `DiabetesPedigreeFunction`
* `Age`

### 2. Performance Comparison on Test Set

| Metric | KNN (All 8 Features) | KNN + PSO (3 Selected Features) | Difference |
| :--- | :---: | :---: | :---: |
| **Test Accuracy** | 81.25% | **82.81%** | **+1.56%** |
| **Class 1 Precision** | **79.41%** | 78.05% | -1.36% |
| **Class 1 Recall** | 48.21% | **57.14%** | **+8.93%** |
| **Class 1 F1-Score** | 0.6000 | **0.6598** | **+0.0598** |

## UI Verification

Below is the screenshot of the model performance dashboard in the Streamlit UI showing the corrected KNN with PSO metrics matching the Jupyter notebook:

![Model Performance Comparison](file:///C:/Users/Admin/.gemini/antigravity-ide/brain/b2a6bb86-6fe0-4898-99b3-9c56d01e0a5f/knn_pso_metrics_1781169928410.png)

> [!TIP]
> By reducing the number of features from 8 to 3, the model's complexity is reduced, and the generalization accuracy on the test set increases from **81.25%** to **82.81%**. More importantly, the **recall** for detecting diabetic patients improved significantly from **48.21%** to **57.14%** and the F1-Score increased to **0.6598**.
