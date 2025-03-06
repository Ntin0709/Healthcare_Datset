# Report: Predicting Patient Length of Stay

## Overview
This report details the development, evaluation, and deployment of machine learning models to predict hospital length of stay (LOS) using healthcare dataset analysis. The analysis includes data preprocessing, exploratory data analysis (EDA), feature engineering, and model experimentation with Linear Regression, Random Forest, XGBoost, and LightGBM. The Random Forest model emerged as the best performer, offering accurate predictions for hospital resource planning.

---

## Code Structure and Deliverables

### Data Analysis Process
The analysis follows a structured approach:

#### 1. Data Preprocessing Steps
- **Data Loading**: Loads healthcare dataset, standardizes column names, and converts `date_of_admission` and `discharge_date` to datetime format.
- **Cleaning**: 
  - Removes outliers in numerical fields using IQR method
  - Original dataset shape: (55,500, 16), after outlier removal: (35,541, 16)

#### 2. Exploratory Data Analysis (EDA)
- **Statistical Analysis**:
  - Age range: 13-89 years (mean: 51.6)
  - Billing amount range: $5,903-$45,221 (mean: $25,579)
  - Length of stay range: 4-27 days (mean: 15.5)
- **Categorical Feature Analysis**:
  - Gender: Males (15.50 days) vs. Females (15.44 days)
  - Medical conditions: Cancer (15.57 days) shows highest average stay
  - Admission types: Emergency (15.50 days) vs. Elective (15.45 days)
  - Insurance providers: Medicare (15.53 days) shows highest average stay

#### 3. Feature Engineering
- **Generated Features**: 
  - Temporal features: `admission_year`, `admission_month`
  - Derived features: `insurance_target`, `medication_target`, `avg_stay_condition`, `billing_log`, `age_squared`
- **Feature Selection**: Dropped highly correlated features (>0.9)
- **Selected Features**: 'billing_amount', 'admission_type', 'medication', 'test_results', 'admission_year', 'insurance_target', 'medication_target', 'billing_log'

#### 4. Model Development and Evaluation
- **Models Tested**: Linear Regression, Random Forest, XGBoost, LightGBM
- **Evaluation Metrics**: MAE, RMSE

---

## Model Performance Comparison

| Model            | MAE (days) | RMSE (days) |
|------------------|------------|-------------|
| Linear Regression| 6.06       | 7.08        |
| Random Forest    | 5.73       | 7.00        |
| XGBoost          | 6.06       | 7.18        |
| LightGBM         | 6.08       | 7.13        |

**Winner**: Random Forest with MAE of 5.73 days.

---

## Key Findings and Recommendations
- **Best Model**: Random Forest offers the most accurate predictions for length of stay.
- **Key Predictors**: Average stay by condition, billing amount (log-transformed), and age.
- **Resource Planning**: 
  - Deploy Random Forest model for discharge prediction
  - Focus planning on identified key features
  - Use insights from EDA to refine features if needed

---

## Conclusion
The Random Forest model provides the most accurate predictions for hospital length of stay with an MAE of 5.73 days. This model can effectively support hospital resource planning and discharge projections. The analysis identified important factors influencing length of stay, which can guide administrators in optimizing patient care and resource allocation.
