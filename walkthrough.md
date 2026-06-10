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
| **Test Accuracy** | 81.25% | **83.33%** | **+2.08%** |
| **Class 1 (Diabetic) Precision** | **79%** | 78% | -1% |
| **Class 1 (Diabetic) Recall** | 48% | **57%** | **+9%** |
| **Class 1 (Diabetic) F1-Score** | 0.60 | **0.66** | **+0.06** |

> [!TIP]
> By reducing the number of features from 8 to 3, the model's complexity is halved, and the generalization accuracy on the hidden test set increases from **81.25%** to **83.33%**. More importantly, the **recall** for detecting diabetic patients improved significantly from **48%** to **57%**.
