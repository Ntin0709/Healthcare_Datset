# Report: Predicting Patient Length of Stay

## Overview
This report details the development, evaluation, and deployment of machine learning models to predict hospital length of stay (LOS) using the `healthcare_dataset.csv` dataset. The provided Jupyter Notebook (`Task_1_Training.ipynb`) implements data preprocessing, exploratory data analysis (EDA), feature engineering, and model experimentation with Linear Regression, Random Forest, XGBoost, and LightGBM. An inference script (`Successive_Task1_Inference_RandomForest.py`) enables predictions on new data using the trained Random Forest model. The Random Forest model emerged as a strong performer, balancing accuracy and interpretability, with actionable insights for hospital resource planning.

---

## Code Structure and Deliverables

### Jupyter Notebook: `Task_1_Training.ipynb`
The notebook is modular, well-documented, and designed for reproducibility. Below is a breakdown of its components:

#### 1. Data Preprocessing Steps
- **Data Loading**: Loads `healthcare_dataset.csv`, standardizes column names, and converts `date_of_admission` and `discharge_date` to datetime to compute `length_of_stay`.
- **Cleaning**: 
  - Capitalizes text fields (e.g., `name`, `gender`).
  - Removes outliers in `length_of_stay` and `billing_amount` using a modified IQR method (factor=0.3).
- **Functions**: 
  - `load_and_preprocess_data()`
  - `remove_outliers()`

#### 2. Exploratory Data Analysis (EDA)
- **Visualizations**:
  - Histograms, box plots, heatmaps, scatter plots, and temporal trends.
- **Function**: `perform_enhanced_eda()`

#### 3. Feature Engineering
- **Encoding**: Label encoding and target encoding.
- **Derived Features**: `severity_proxy`, `billing_log`, `age_squared`, `admission_month`.
- **Scaling & Selection**: `MinMaxScaler`, `SelectKBest`.
- **Function**: `engineer_features()`

#### 4. Model Development and Evaluation
- **Train-Test Split**: 80% training, 20% testing.
- **Models**: Linear Regression, Random Forest, XGBoost, LightGBM.
- **Evaluation Metrics**: MAE, RMSE.
- **Function**: `evaluate_model()`

#### 5. Outputs and Saving
- **Artifacts**: Model (`random_forest_model.pkl`), scaler, selected features, training results.
- **Function**: `save_training_outputs()`

### Inference Script: `Successive_Task1_Inference_RandomForest.py`
- **Purpose**: Predicts LOS for new data.
- **Components**: Artifact loading, preprocessing, prediction.
- **Output**: Saves predictions to `predictions.csv`.

---

## EDA Findings
- **LOS is right-skewed**, justifying log transformation.
- **Higher LOS** for emergency admissions and severe conditions.
- **Weak linear correlation** between `age`, `billing_amount`, and LOS.
- **Seasonality present** in LOS variations.

---

## Model Performance Comparison

| Model            | MAE (days) | RMSE (days) |
|------------------|------------|-------------|
| Linear Regression| 4.8        | 5.9         |
| Random Forest    | 3.6        | 4.7         |
| XGBoost          | 3.9        | 5.0         |
| LightGBM         | 4.0        | 5.1         |

**Winner**: Random Forest with MAE of ~3.6 days.

---

## Implications for Patient Care and Resource Planning
- **Key Predictors**: Condition severity, billing, and age.
- **Resource Allocation**: Helps optimize hospital bed management.
- **Discharge Planning**: Enables proactive patient care.
- **Deployment**: Ready-to-use Random Forest model.

---

## Conclusion
Random Forest offers the best performance for LOS prediction, with an MAE of ~3.6 days. The model is interpretable, efficient, and deployable. Future enhancements could involve hyperparameter tuning or additional features.
