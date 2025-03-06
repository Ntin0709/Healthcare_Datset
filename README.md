# Healthcare Dataset Documentation

This repository contains machine learning models for healthcare predictions across two main tasks:

## Task 1: Hospital Length of Stay Prediction

### Overview
This component uses machine learning algorithms to predict the length of stay for hospital patients. The model helps healthcare providers better allocate resources and plan for patient care.

### Requirements
- Python 3.8+
- Libraries:
  - pandas
  - numpy
  - scikit-learn
  - xgboost
  - lightgbm
  - matplotlib
  - seaborn
  - joblib

### Running the Code
1. **Training**: 
   - Open and run `Task_1/Task_1_Training.ipynb`

2. **Inference**: 
   - Run `Task_1/inference.py`

3. **Model Outputs**: 
   - Extract `Task_1/model_outputs.zip` to a directory named `model_outputs` to use the pre-trained models

4. **Documentation**:
   - The complete analysis and results are documented in `Task_1/Report_1.md`

## Task 2: High-Risk Patient Prediction

### Overview
This component focuses on identifying high-risk patients based on various medical conditions using advanced machine learning techniques. The model aims to enable early intervention for patients who may require additional care.

### Requirements
- Python 3.8+
- Libraries:
  - pandas
  - numpy
  - scikit-learn
  - xgboost
  - matplotlib
  - seaborn
  - joblib

### Running the Code
1. **Training**: 
   - Open and run `Task_2/Task_2_Training.ipynb`

2. **Inference**: 
   - Run `Task_2/inference.py`

3. **Model Outputs**: 
   - Extract `Task_2/baseline_rf_model.zip` to a directory named `model_outputs` to use the pre-trained models

4. **Documentation**:
   - The complete analysis and methodology are documented in `Task_2/Report_2.md`

## Repository Structure

```
healthcare-dataset/
├── Task_1/
│   ├── Task_1_Training.ipynb
│   ├── inference.py
│   ├── model_outputs.zip
│   └── Report_1.md
└── Task_2/
    ├── Task_2_Training.ipynb
    ├── inference.py
    ├── baseline_rf_model.zip
    └── Report_2.md
```

## Getting Started

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn joblib
   ```
3. Extract the model files from the zip archives
4. Run the inference scripts or notebooks to make predictions
