# Healthcare_Datset

```
high-risk-patient-prediction/
├── data/
│   └── healthcare_dataset.csv  # Input dataset (assumed, not provided here)
├── notebooks/
│   └── Task_2_Training.ipynb  # Main notebook with all code
├── scripts/
│   └── inference.py           # Placeholder for inference script (not fully implemented in doc)
├── models/
│   └── baseline_rf_model.pkl  # Saved Baseline RF model
├── reports/
│   └── Final_Report.md        # This report in markdown format
│   └── Final_Report.pdf       # Optional PDF version
├── README.md                  # Instructions for running the code
└── .gitignore                 # Ignore large files, virtual envs, etc.
```

 High-Risk Patient Prediction

 Overview
This repository contains code and analysis for identifying high-risk patients based on medical conditions using machine learning.

 Requirements
- Python 3.8+
- Libraries: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, joblib

 Setup
1. Clone the repository:```markdown
#
   ```bash
   git clone <repository-link>
   cd high-risk-patient-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place `healthcare_dataset.csv` in the `data/` folder.

 Running the Code
- Open `notebooks/Task_2_Training.ipynb` in Jupyter Notebook or Colab and run all cells.
- The trained model is saved as `models/baseline_rf_model.pkl`.

 Outputs
- Visualizations and model evaluations are displayed in the notebook.
- Final report is in `reports/Final_Report.md`.

 License
MIT License
```
