import pandas as pd
import numpy as np
import joblib

# Define preprocessing function
def preprocess_data(new_data):
    """
    Preprocess new patient data to match training features.
    Args:
        new_data (pd.DataFrame): Raw input data with required columns.
    Returns:
        pd.DataFrame: Preprocessed features ready for prediction.
    """
    # Required columns
    required_cols = ['Age', 'Gender', 'Medical Condition', 'Test Results', 
                     'Date of Admission', 'Discharge Date']
    
    # Check for missing columns
    missing_cols = [col for col in required_cols if col not in new_data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Copy to avoid modifying original
    data = new_data.copy()

    # Handle missing values
    for col in required_cols:
        if data[col].isnull().sum() > 0:
            if data[col].dtype == 'object':
                data[col].fillna(data[col].mode()[0], inplace=True)
            else:
                data[col].fillna(data[col].median(), inplace=True)

    # Feature Engineering
    data['Date of Admission'] = pd.to_datetime(data['Date of Admission'])
    data['Discharge Date'] = pd.to_datetime(data['Discharge Date'])
    data['Length of Stay'] = (data['Discharge Date'] - data['Date of Admission']).dt.days
    data['Length of Stay Log'] = np.log1p(data['Length of Stay'])

    # Age bins
    bins = [0, 50, 70, 150]
    labels = ['lt_50', '50_70', 'gt_70']
    data['Age Bin'] = pd.cut(data['Age'], bins=bins, labels=labels, right=False)
    age_bin_encoded = pd.get_dummies(data['Age Bin'], prefix='Age')

    # Condition severity and interaction
    high_risk_conditions = ['Cancer', 'Diabetes']
    condition_severity = {'Cancer': 3, 'Diabetes': 2, 'Asthma': 1, 'Obesity': 1, 'Hypertension': 1, 'Arthritis': 0}
    data['Condition Severity'] = data['Medical Condition'].map(condition_severity).fillna(0)
    data['Age_HighRisk'] = data['Age'] * data['Medical Condition'].isin(high_risk_conditions).astype(int)

    # One-hot encode Medical Condition
    condition_encoded = pd.get_dummies(data['Medical Condition'], prefix='Condition')
    
    # Combine features
    data = pd.concat([data, condition_encoded, age_bin_encoded], axis=1)

    # Define expected features (from training)
    expected_features = [
        'Age', 'Length of Stay Log', 'Condition Severity', 'Age_HighRisk',
        'Condition_Arthritis', 'Condition_Asthma', 'Condition_Cancer', 
        'Condition_Diabetes', 'Condition_Hypertension', 'Condition_Obesity',
        'Age_lt_50', 'Age_50_70', 'Age_gt_70'
    ]

    # Ensure all expected features are present
    for feature in expected_features:
        if feature not in data.columns:
            data[feature] = 0  # Add missing feature with default value 0

    # Select only expected features in correct order
    return data[expected_features]

# Load the saved model
model = joblib.load('/home/npci-admin/Healthcare_Datset/Task_2/baseline_rf_model.pkl')
print("Model loaded successfully from 'baseline_rf_model.pkl'")

# Example inference function
def predict_risk(new_data):
    """
    Predict risk for new patient data.
    Args:
        new_data (pd.DataFrame): Raw input data.
    Returns:
        np.array: Predicted risk (0 = low-risk, 1 = high-risk).
        np.array: Prediction probabilities.
    """
    try:
        # Preprocess the data
        X_new = preprocess_data(new_data)
        
        # Predict risk and probabilities
        predictions = model.predict(X_new)
        probabilities = model.predict_proba(X_new)[:, 1]
        
        return predictions, probabilities
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None

# Example usage
if __name__ == "__main__":
    # Sample new data (replace with actual new patient data)
    sample_data = pd.DataFrame({
        'Age': [75, 45],
        'Gender': ['Male', 'Female'],
        'Medical Condition': ['Cancer', 'Asthma'],
        'Test Results': ['Abnormal', 'Normal'],
        'Date of Admission': ['2024-01-01', '2024-02-01'],
        'Discharge Date': ['2024-01-10', '2024-02-05']
    })

    # Predict
    predictions, probabilities = predict_risk(sample_data)
    
    if predictions is not None:
        print("\nPredictions:")
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            print(f"Patient {i+1}: Risk = {'High' if pred == 1 else 'Low'} (Probability: {prob:.2f})")