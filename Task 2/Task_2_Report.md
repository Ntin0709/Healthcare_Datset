# Report: Identifying High-Risk Patients Based on Medical Conditions

## Overview
This report summarizes the development, evaluation, and findings of a machine learning pipeline designed to identify high-risk patients based on medical conditions, age, and other clinical features. The provided Jupyter Notebook (`Task_2_Training.ipynb`) implements data preprocessing, exploratory data analysis (EDA), model training, and evaluation, culminating in actionable insights for patient care. 

Three models—Baseline Random Forest, Tuned Random Forest, and XGBoost—were tested, with the **Baseline Random Forest recommended as the best performer**.

---

## 1. Data Preprocessing Steps
- **Dataset Loading**: Loaded `healthcare_dataset.csv` using pandas, ensuring non-negative billing amounts.
- **Missing Value Handling**: Filled missing values with mode (categorical) or median (numerical) for columns like `Age`, `Gender`, and `Test Results`.
- **Feature Engineering**:
  - Converted `Date of Admission` and `Discharge Date` to datetime, calculating `Length of Stay` and its log-transformed version (`Length of Stay Log`).
  - Created `Age Bin` categories (`<50`, `50-70`, `>70`) with one-hot encoding.
  - Assigned `Condition Severity` scores (e.g., Cancer: 3, Diabetes: 2) and computed `Age_HighRisk` as an interaction term.
  - One-hot encoded `Medical Condition`.
  - Defined `Risk` target: High-risk if (**Cancer or Diabetes AND Age > 70**) OR **abnormal test results**.

---

## 2. Model Development and Evaluation
- **Features**: Included `Age`, `Length of Stay Log`, `Condition Severity`, `Age_HighRisk`, and one-hot encoded variables.
- **Train-Test Split**: 70% training, 30% testing with a random seed of 42.
- **Models**:
  - **Experiment 1: Baseline Random Forest**: Used balanced class weights, 100 estimators.
  - **Experiment 2: Tuned Random Forest**: Employed GridSearchCV for hyperparameter tuning (`n_estimators`, `max_depth`, `min_samples_split`) with recall scoring and a **0.3 prediction threshold**.
  - **Experiment 3: XGBoost**: Applied a `scale_pos_weight` of 1.5 for class imbalance.
- **Evaluation Metrics**: Classification report, ROC-AUC score, confusion matrix for each model.

---

## 3. Visualizations
### EDA Plots:
- **Class Distribution** (bar plot).
- **Risk by Medical Condition** (count plot).
- **Age and Length of Stay by Risk** (box plots).
- **Condition Severity vs. Risk** (box plot).
- **Correlation Matrix** (heatmap).

### Model Visualizations:
- **Confusion Matrices** (heatmaps) for each model.
- **ROC Curve Comparison** across models.
- **Feature Importance**: Table for Baseline Random Forest.

---

## 4. Insights and Recommendations
- **Best Model Saved**: `baseline_rf_model.pkl` for future use.
- **Final Recommendation**: Baseline Random Forest offers a balanced trade-off between **recall, accuracy, and ROC-AUC**.

---

## EDA Findings
### 1. Class Distribution:
- 61% **low-risk (Risk=0)**, 39% **high-risk (Risk=1)**, indicating **moderate class imbalance**.

### 2. Risk by Medical Condition:
- **Cancer and Diabetes** exhibit higher risk prevalence, validating their designation as high-risk conditions.

### 3. Age and Risk:
- High-risk patients are **older (median age higher for Risk=1)**, especially beyond 70.

### 4. Length of Stay:
- **Longer stays (log-transformed)** correlate with higher risk, suggesting complications.

### 5. Correlations:
- `Age`, `Age_HighRisk`, and **condition-specific features** (e.g., `Condition_Cancer`) show **positive correlations** with risk.

**Insight**: **Age (>70), Cancer/Diabetes, and abnormal test results are strong predictors of high risk**, supporting the target definition.

---

## Model Performance Comparison

### Experiment 1: Baseline Random Forest
- **Metrics**:  
  - **Accuracy**: 0.58  
  - **High-risk recall**: 0.50  
  - **ROC-AUC**: 0.635  
- **Output**: Balanced performance with moderate recall and precision.
- **Confusion Matrix**: Reasonable **true positives** (high-risk detection) with some false positives.

### Experiment 2: Tuned Random Forest (Threshold 0.3)
- **Metrics**:  
  - **Accuracy**: 0.39  
  - **High-risk recall**: Improved (higher due to threshold adjustment)  
  - **ROC-AUC**: Comparable to baseline  
- **Output**: High recall at the **cost of low accuracy and excessive false positives**.
- **Best Params**: Tuned via GridSearchCV (`n_estimators=200`, `max_depth=20`, `min_samples_split=5`).

### Experiment 3: XGBoost
- **Metrics**:  
  - **Accuracy**: 0.64  
  - **High-risk recall**: 0.34  
  - **ROC-AUC**: 0.63  
- **Output**: Higher accuracy but **lower recall for high-risk cases** compared to Baseline RF.

### ROC Curve Comparison
- **Baseline RF**: AUC = **0.635** (highest).
- **Tuned RF**: Similar AUC but **impractical due to low accuracy**.
- **XGBoost**: AUC = **0.63**, slightly lower than Baseline RF.

**Winner**: **Baseline Random Forest**, balancing **recall (0.50)** and **ROC-AUC (0.635)** without sacrificing too much accuracy (**0.58**).

---

## Implications for Patient Care and Targeted Interventions
### 1. Key Predictors
- **Age, Length of Stay, and Cancer/Diabetes** conditions drive risk, confirmed by EDA and feature importance.

### 2. Targeted Interventions
- Prioritize **elderly patients (>70) with Cancer or Diabetes** for **early screening and intervention**.
- Monitor patients with **abnormal test results and extended hospital stays** closely.

### 3. Resource Allocation
- **Baseline RF model avoids over-prediction** (unlike Tuned RF), ensuring **efficient use of medical resources**.

### 4. Practical Deployment
- The saved model (`baseline_rf_model.pkl`) can be **integrated into hospital systems** for real-time risk assessment.

### Business Insight
Implementing this model can **reduce adverse outcomes** by focusing resources on the **most vulnerable patients**, improving **care quality** and potentially **lowering costs** associated with untreated high-risk cases.

---

## Conclusion
The **Baseline Random Forest model** offers the **best trade-off** between **accuracy, recall, and ROC-AUC**, making it suitable for identifying high-risk patients. 

The **EDA and feature importance** underscore the **clinical relevance** of **age and specific conditions**, aligning with **medical intuition**.
